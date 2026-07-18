"""这个文件做什么：RAG 流水线的"喂料"环节 —— 把原始文档读进来并切成小段。

功能分解：
- load_file：按扩展名读取单个文件（PDF 用 pypdf 抽文字，TXT/MD 直接读），其它类型跳过
- chunk_text：按字符数用"滑动窗口 + 重叠"切块，避免长文被硬切断，对中文很友好
- ingest：扫描 data/ 下所有 PDF/TXT/MD，逐个读取并切块，给每块标注来源文件名与序号

输出是一组 {"text", "source", "chunk_id"} 字典，交给 embeddings 去向量化。"""
from pathlib import Path

from .config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP


def load_file(path: Path) -> str:
    """读取单个文件，返回纯文本。"""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    if suffix in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="ignore")
    # 其他类型跳过
    return ""


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """把长文本切成有重叠的块。"""
    text = text.strip()
    if not text:
        return []
    chunks = []
    step = max(1, size - overlap)
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start += step
    return chunks


def ingest() -> list:
    """扫描 data/ 下所有文档，返回 [{text, source, chunk_id}, ...]。"""
    docs = []
    files = []
    for ext in ("*.pdf", "*.txt", "*.md"):
        files.extend(DATA_DIR.glob(ext))
    if not files:
        print(f"[ingest] 在 {DATA_DIR} 没找到 pdf/txt/md 文件")
        return docs
    for f in sorted(files):
        print(f"[ingest] 读取 {f.name}")
        text = load_file(f)
        for i, c in enumerate(chunk_text(text)):
            docs.append({"text": c, "source": f.name, "chunk_id": f"{f.stem}-{i}"})
    print(f"[ingest] 共切出 {len(docs)} 个文本块")
    return docs
