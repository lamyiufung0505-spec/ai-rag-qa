"""这个文件做什么：全局配置中心，集中存放整个项目用到的所有"常量"和"可调参数"。

它负责：
- 目录路径：data/（你的文档放这里）、index/（向量索引存这里）
- 模型与密钥：从 .env 读取 DeepSeek 的 API key、接口地址、模型名，以及本地向量化模型名
- 切块与检索参数：文本切块长度、块间重叠字符数、默认检索段落数（top_k）

其它所有模块（ingest / embeddings / qa / cli / app）都从这里 import 配置，
因此改这一个文件就能调整全局行为，不用到处分头改代码。"""
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
