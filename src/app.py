"""Streamlit 网页界面：左侧建索引，右侧提问。

运行：streamlit run src/app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
    st.caption("把你的 PDF / TXT / MD 放进 data/ 目录即可")

st.header("② 提问")
q = st.text_input("输入你的问题：")
top_k = st.slider("检索段落数", 1, 8, 4)
if st.button("回答", type="primary"):
    if not q.strip():
        st.warning("请先输入问题")
    else:
        with st.spinner("思考中…"):
            ans = ask(q, top_k=top_k, show_context=True)
        st.markdown(ans)
