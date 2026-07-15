"""本地向量化与检索（不依赖付费 embedding API）。

- 用 fastembed 在本地把文本变成向量
- 用 numpy 做余弦相似度检索 top-k
- 索引（文档块 + 向量矩阵）持久化到 index/index.pkl
"""
import pickle

import numpy as np

from .config import EMBED_MODEL, INDEX_FILE
from .ingest import ingest

_embedder = None


def get_embedder():
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
