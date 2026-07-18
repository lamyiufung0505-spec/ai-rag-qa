"""这个文件做什么：从 data/ 目录加载文档并切成文本块，供后续向量化与检索使用。

支持 PDF / TXT / MD。
切块策略：按字符长度滑动窗口（带重叠），对中文也很友好。
"""
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
