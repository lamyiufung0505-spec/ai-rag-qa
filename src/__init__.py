"""这个文件做什么：标识 ai-rag-qa 是一个 Python 包（package），让整个项目能被当作模块引用。

它的作用：
- 使 src/ 下的各模块之间能用相对导入（如 from .config import ...）互相引用，避免路径混乱
- 允许外部代码用「from src import embeddings」这样的方式直接调用本项目功能

它本身不含任何业务逻辑，只是一个"包的身份标记"，但缺少它 Python 就不把 src 当包处理。"""
