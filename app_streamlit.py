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


def send_feedback(conversation_id: int, value: int) -> None:
    try:
        requests.post(
            f"{settings.backend_url}/feedback",
            json={"conversation_id": conversation_id, "feedback": value},
            timeout=10,
        ).raise_for_status()
    except requests.exceptions.RequestException as exc:
        st.toast(f"Couldn't save feedback: {exc}", icon="⚠️")
        return
    for msg in st.session_state.messages:
        if msg.get("conversation_id") == conversation_id:
            msg["feedback"] = value


def render_feedback(msg: dict, key: str) -> None:
    conversation_id = msg.get("conversation_id")
    if conversation_id is None:
        return

    if msg.get("feedback") == 1:
        st.caption("👍 Thanks for the feedback!")
    elif msg.get("feedback") == -1:
        st.caption("👎 Thanks for the feedback!")
    else:
        up, down, _ = st.columns([1, 1, 8])
        up.button("👍", key=f"up-{key}", on_click=send_feedback, args=(conversation_id, 1))
        down.button("👎", key=f"down-{key}", on_click=send_feedback, args=(conversation_id, -1))


for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"])
        if msg["role"] == "assistant":
            render_feedback(msg, key=str(i))

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
                answer, sources, conversation_id = data["answer"], data.get("sources", []), data.get("conversation_id")
            except requests.exceptions.RequestException as exc:
                answer, sources, conversation_id = f"Error contacting the backend: {exc}", [], None
        st.markdown(answer)
        if sources:
            render_sources(sources)

        new_msg = {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "conversation_id": conversation_id,
            "feedback": None,
        }
        render_feedback(new_msg, key=str(len(st.session_state.messages)))

    st.session_state.messages.append(new_msg)
