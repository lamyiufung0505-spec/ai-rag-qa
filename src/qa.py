"""这个文件做什么：RAG 的"大脑 / 生成层" —— 拿到问题后，检索相关片段并调用大模型生成答案。

工作流程：
1. search：先调 embeddings.search 取出与问题最相关的若干原文片段
2. build_user_prompt：把这些片段拼成带【上下文】的提示词
3. ask：调用 DeepSeek，严格只依据【上下文】作答；上下文里找不到就明说"无法回答"，绝不编造
4. 可选 show_context：在答案后附上参考来源文件名与检索到的原文片段，方便核对

它是连接"检索"与"回答"的最后一公里，决定最终输出质量。"""
from openai import OpenAI

from .config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from .embeddings import search

SYSTEM_PROMPT = (
    "你是一个严谨的文档问答助手。只能依据下面提供的【上下文】回答问题，"
    "不要编造上下文之外的信息。如果上下文里找不到答案，就明确告诉用户"
    "“根据提供的文档无法回答这个问题”。回答请使用与用户问题相同的语言。"
)


def build_user_prompt(query: str, hits: list) -> str:
    """把检索到的片段拼成给 DeepSeek 的提示词（上下文 + 问题）。"""
    ctx = "\n\n".join(
        f"[来源 {h['doc']['source']}]\n{h['doc']['text']}" for h in hits
    )
    return f"【上下文】\n{ctx}\n\n【问题】{query}"


def ask(query: str, top_k: int = 4, show_context: bool = False) -> str:
    """执行一次 RAG 问答：检索相关片段 → 调 DeepSeek → 返回答案。"""
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
