"""这个文件做什么：根据用户问题检索最相关的文档片段，再调用 DeepSeek 大模型生成基于上下文的答案。

核心原则：只依据【上下文】回答，不知道就明说，不编造。
"""
from openai import OpenAI

from .config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from .embeddings import search

SYSTEM_PROMPT = (
    "你是一个严谨的文档问答助手。只能依据下面提供的【上下文】回答问题，"
    "不要编造上下文之外的信息。如果上下文里找不到答案，就明确告诉用户"
    "“根据提供的文档无法回答这个问题”。回答请使用与用户问题相同的语言。"
)


def build_user_prompt(query: str, hits: list) -> str:
    ctx = "\n\n".join(
        f"[来源 {h['doc']['source']}]\n{h['doc']['text']}" for h in hits
    )
    return f"【上下文】\n{ctx}\n\n【问题】{query}"


def ask(query: str, top_k: int = 4, show_context: bool = False) -> str:
    if not DEEPSEEK_API_KEY:
        return "❌ 未设置 DEEPSEEK_API_KEY，请在 .env 里填入你的 DeepSeek key。"

    hits = search(query, top_k)
    if not hits:
        return "⚠️ 索引为空，请先运行：python -m src.cli ingest"

    user = build_user_prompt(query, hits)
    try:
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        resp = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
        )
        answer = resp.choices[0].message.content
    except Exception as e:  # noqa: BLE001
        return f"❌ 调用 DeepSeek 失败：{e}\n（请检查 key 是否有效、网络是否可用）"

    if show_context:
        srcs = ", ".join(sorted({h["doc"]["source"] for h in hits}))
        ctx_block = "\n\n".join(
            f"▸ [{h['doc']['source']}] (相似度 {h['score']:.3f})\n{h['doc']['text']}"
            for h in hits
        )
        return (
            f"{answer}\n\n"
            f"🔎 参考来源：{srcs}\n\n"
            f"---\n\n📄 检索到的原文片段：\n\n{ctx_block}"
        )
    return answer
