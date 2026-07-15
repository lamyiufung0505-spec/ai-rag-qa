# 本地 RAG 文档问答（ai-rag-qa）

一个**完全本地运行**的检索增强生成（RAG）问答工具：把你的文档（PDF / TXT / Markdown）放进 `data/`，它就能基于这些文档回答你的问题。

- **向量化本地完成**（fastembed），不依赖任何付费 embedding API
- **答案用 DeepSeek 生成**（你有 key 就能用）
- **文档不出本机**，隐私友好
- 自带**命令行**与**网页界面**两种用法

---

## 1. 准备

```bash
# 创建虚拟环境（用你机器上的 python 3.11+ 即可）
python -m venv .venv
.venv\Scripts\activate        # Windows
#   source .venv/bin/activate # macOS/Linux

pip install -r requirements.txt
```

在 `.env` 里填入你的 DeepSeek key（已帮你从之前的项目复用）：

```
DEEPSEEK_API_KEY=你的key
```

> 想对**纯中文**文档效果更好，可在 `.env` 加一行：
> `EMBED_MODEL=BAAI/bge-small-zh-v1.5`
> 默认是 `intfloat/multilingual-e5-small`（中英文都行）。

---

## 2. 放入你的文档

把任意 PDF / TXT / MD 文件丢进 `data/` 目录（里面已经放了一个 `sample.txt` 示例）。

---

## 3. 建立索引

```bash
python -m src.cli ingest
```

首次运行会下载本地向量模型（约 100~300MB，之后缓存），请稍候。

---

## 4. 提问（命令行）

```bash
python -m src.cli ask "这个工具怎么用？" --show-context
```

`--show-context` 会在答案后列出参考来源文件名。

---

## 5. 提问（网页界面，更直观）

```bash
streamlit run src/app.py
```

打开浏览器：左侧点「读取 data/ 并重建索引」，右侧输入问题即可。

---

## 6. 项目结构

```
ai-rag-qa/
├─ data/            # 放你的文档（PDF/TXT/MD）
├─ index/           # 自动生成的向量索引（不入库）
├─ src/
│  ├─ config.py     # 路径 / 密钥 / 参数
│  ├─ ingst.py     # 文档加载与切块
│  ├─ embeddings.py # 本地向量化 + 余弦检索 + 持久化
│  ├─ qa.py        # 调 DeepSeek 生成答案
│  ├─ cli.py        # 命令行入口
│  └─ app.py        # Streamlit 网页界面
├─ .env             # DeepSeek key（已填好）
└─ requirements.txt
```

---

## 7. 常见问题

- **改了文档要重跑吗？** 要。重新 `python -m src.cli ingest` 即可更新索引。
- **支持多大文档？** 本版用 numpy 做检索，个人知识库（几百页以内）完全够用；海量文档可换成 FAISS。
- **能换别的模型吗？** 向量模型在 `.env` 的 `EMBED_MODEL` 改；生成模型改 `DEEPSEEK_MODEL`（也支持任何 OpenAI 兼容接口）。
