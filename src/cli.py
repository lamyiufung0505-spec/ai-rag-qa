"""这个文件做什么：命令行入口，让你不写代码也能在终端里使用这个项目。

提供两个子命令：
- python -m src.cli ingest    → 读取 data/ 文档并建立向量索引（底层调 embeddings.build_index）
- python -m src.cli ask "问题" → 基于索引回答（底层调 qa.ask），支持 --top-k 和 --show-context

用 argparse 解析参数，是项目的"无界面"使用方式，适合批量处理或在脚本里调用。"""
import argparse

from . import embeddings
from .qa import ask


def main():
    """命令行主入口：解析 ingest / ask 子命令并调度对应逻辑。"""
    p = argparse.ArgumentParser(description="本地 RAG 文档问答")
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("ingest", help="读取 data/ 下的文档并建立索引")

    a = sub.add_parser("ask", help="基于文档回答问题")
    a.add_argument("question", help="你的问题")
    a.add_argument("--top-k", type=int, default=4, help="检索的段落数")
    a.add_argument("--show-context", action="store_true", help="在答案后附上参考来源")

    args = p.parse_args()
    if args.cmd == "ingest":
        n = embeddings.build_index()
        print(f"完成，共索引 {n} 个文本块。")
    elif args.cmd == "ask":
        print(ask(args.question, args.top_k, args.show_context))
    else:
        p.print_help()


if __name__ == "__main__":
    main()
