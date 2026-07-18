"""这个文件做什么：RAG 的"检索引擎" —— 负责把文字变成向量，并按语义找出最相关的内容。

功能特点：
- 本地向量化：用 fastembed 在本地把文本转成向量，不依赖任何付费 embedding API
- build_index：把 ingest 切好的文本块向量化、做 L2 归一化，连同原文一起存成 index/index.pkl（首次会联网下载模型）
- load_index：在需要时把已保存的索引从磁盘读回内存
- search：把用户问题向量化，与索引做余弦相似度计算，返回相似度最高的 top-k 个片段

这个文件让"问一个问题 → 找到文档里最相关的几句"成为可能，是检索的核心。"""
import pickle

import numpy as np

from .config import EMBED_MODEL, INDEX_FILE
from .ingest import ingest

_embedder = None


def get_embedder():
    """懒加载本地向量模型，返回单例（第一次调用会下载模型）。"""
    global _embedder
    if _embedder is None:
        from fastembed import TextEmbedding

        print(f"[embed] 加载本地向量模型：{EMBED_MODEL}（首次会下载，请稍候）")
        _embedder = TextEmbedding(model_name=EMBED_MODEL)
    return _embedder


def build_index() -> int:
    """读取 data/ 文档 → 切块 → 向量化 → 保存索引。返回块数量。"""
    docs = ingest()
    if not docs:
        return 0
    texts = [d["text"] for d in docs]
    emb = get_embedder()
    vectors = [np.asarray(v, dtype="float32") for v in emb.embed(texts)]
    matrix = np.vstack(vectors)
    # L2 归一化，方便用点积算余弦相似度
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.clip(norms, 1e-9, None)
    with open(INDEX_FILE, "wb") as f:
        pickle.dump({"docs": docs, "matrix": matrix}, f)
    print(f"[index] 已写入 {INDEX_FILE}（{len(docs)} 块）")
    return len(docs)


def load_index():
    """加载已保存的索引文件（文档块 + 向量矩阵），返回 dict。"""
    if not INDEX_FILE.exists():
        raise FileNotFoundError("索引不存在，请先运行：python -m src.cli ingest")
    with open(INDEX_FILE, "rb") as f:
        return pickle.load(f)


def search(query: str, top_k: int = 4) -> list:
    """返回与 query 最相关的 top_k 个 {doc, score}。"""
    data = load_index()
    docs = data["docs"]
    matrix = data["matrix"]
    q = np.asarray(list(get_embedder().embed([query]))[0], dtype="float32")
    q = q / np.linalg.norm(q)
    sims = matrix @ q  # 已归一化，点积即余弦
    k = min(top_k, len(docs))
    idx = np.argsort(-sims)[:k]
    return [{"doc": docs[i], "score": float(sims[i])} for i in idx]
