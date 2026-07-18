"""这个文件做什么：命令行入口，让用户通过终端指令控制 RAG 流程（ingest 建立索引 / ask 提问）。

    python -m src.cli ingest          # 读取 data/ 并建立索引
    python -m src.cli ask "你的问题"   # 基于文档回答
"""
import argparse

from . import embeddings
from .qa import ask


def main():
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
