"""Streamlit chat UI for the arxiv RAG backend."""

import requests
import streamlit as st

from arxiv_rag.config import settings

st.set_page_config(page_title="Arxiv AI Papers RAG Chat", page_icon="📄")
st.title("Arxiv AI Papers RAG Chat")
st.caption("Ask questions about AI/ML arXiv papers. Answers are grounded in retrieved papers with cited sources.")

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(sources: list[dict]) -> None:
    with st.expander(f"Sources ({len(sources)})"):
        for src in sources:
            st.markdown(
                f"**[{src['title']}]({src['pdf_url']})**  \n"
                f"{src['authors']}  \n"
                f"Published: {src['published']} · arXiv: {src['entry_id']} · relevance: {src['relevance_score']:.3f}"
            )


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"])

question = st.chat_input("Ask about arXiv AI papers...")
if question:
    st.session_state.messages.append({"role": "user", "content": question, "sources": None})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching papers and generating an answer..."):
            try:
                resp = requests.post(f"{settings.backend_url}/chat", json={"question": question}, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                answer, sources = data["answer"], data.get("sources", [])
            except requests.exceptions.RequestException as exc:
                answer, sources = f"Error contacting the backend: {exc}", []
        st.markdown(answer)
        if sources:
            render_sources(sources)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
