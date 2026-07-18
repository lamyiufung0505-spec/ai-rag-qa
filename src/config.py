"""这个文件做什么：集中管理全局配置，包括目录路径、DeepSeek API 密钥、向量模型、文本切块与检索参数。"""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# 目录
DATA_DIR = BASE_DIR / "data"  # 用户文档存放目录（PDF / TXT / MD）
INDEX_DIR = BASE_DIR / "index"  # 向量索引存放目录
INDEX_FILE = INDEX_DIR / "index.pkl"  # 索引文件名（pickle 格式）
INDEX_DIR.mkdir(exist_ok=True)  # 自动创建索引目录

# 模型
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")  # 本地向量化模型名
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")  # DeepSeek API 密钥（从 .env 读取）
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")  # DeepSeek 接口地址
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")  # 生成答案用的模型

# 切块 / 检索
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))  # 文本切块长度（字符数）
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))  # 相邻块重叠长度（字符数）
TOP_K = int(os.getenv("TOP_K", "4"))  # 默认检索的段落数
