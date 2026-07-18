"""这个文件做什么：Streamlit 网页界面，提供在浏览器里运行的图形化交互（项目的"有界面"使用方式）。

布局与功能：
- 左侧栏：一个按钮"读取 data/ 并重建索引"，点击即调 embeddings.build_index()
- 主区域：输入框填问题、滑块选检索段落数、勾选是否显示原文，点"回答"后调 qa.ask() 实时显示答案
- 展开区：以可折叠形式列出检索到的原文片段与相似度，方便溯源核对

比 CLI 更直观，适合非技术用户直接上手。运行：streamlit run src/app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 把项目根目录加入模块搜索路径，确保能 import src

from src import embeddings
from src.qa import ask

import streamlit as st

st.set_page_config(page_title="本地 RAG 文档问答", page_icon="📚")
st.title("📚 本地 RAG 文档问答")

with st.sidebar:
    st.header("① 建立索引")
    if st.button("读取 data/ 并重建索引"):
        with st.spinner("正在向量化（首次会下载模型）…"):
            n = embeddings.build_index()
        st.success(f"索引完成：{n} 个文本块")
    st.caption("把你的 PDF / TXT / MD 放进 data/ 目录即可")  # 左侧栏：点击按钮后读取 data/ 并重建索引

st.header("② 提问")  # 主区域：输入问题后调 ask() 生成答案
q = st.text_input("输入你的问题：")
top_k = st.slider("检索段落数", 1, 8, 4)
show_ctx = st.checkbox("显示检索到的原文片段", value=True)
if st.button("回答", type="primary"):
    if not q.strip():
        st.warning("请先输入问题")
    else:
        with st.spinner("思考中…"):
            ans = ask(q, top_k=top_k, show_context=False)
            hits = embeddings.search(q, top_k=top_k) if show_ctx else []
        st.markdown(ans)
        if show_ctx and hits:
            with st.expander(f"🔎 检索到的原文片段（{len(hits)} 段）", expanded=True):
                for i, h in enumerate(hits, 1):
                    st.caption(f"#{i} 来源：{h['doc']['source']} ｜ 相似度：{h['score']:.3f}")
                    st.markdown(h["doc"]["text"])
