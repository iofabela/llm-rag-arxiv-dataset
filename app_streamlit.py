"""Streamlit chat UI for the arxiv RAG backend."""

import logging

import requests
import streamlit as st

from arxiv_rag.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

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


def ask_backend(question: str) -> tuple[str, list[dict], int | None, bool]:
    """POST the question to the backend, returning (message, sources, conversation_id, is_error)."""
    try:
        resp = requests.post(f"{settings.backend_url}/chat", json={"question": question}, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["answer"], data.get("sources", []), data.get("conversation_id"), False
    except requests.exceptions.HTTPError as exc:
        detail = None
        try:
            detail = exc.response.json().get("detail")
        except (ValueError, AttributeError):
            pass
        logger.error("Backend returned HTTP %s for question %r: %s", exc.response.status_code, question, detail or exc)
        message = detail or "The backend ran into a problem answering that question. Please try again in a moment."
        return message, [], None, True
    except requests.exceptions.RequestException as exc:
        logger.error("Could not reach backend at %s for question %r: %s", settings.backend_url, question, exc)
        message = "Couldn't reach the backend. Please check that it's running and try again."
        return message, [], None, True


for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg.get("is_error"):
            st.error(msg["content"])
        else:
            st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"])
        if msg["role"] == "assistant" and not msg.get("is_error"):
            render_feedback(msg, key=str(i))

question = st.chat_input("Ask about arXiv AI papers...")
if question:
    st.session_state.messages.append({"role": "user", "content": question, "sources": None})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching papers and generating an answer..."):
            answer, sources, conversation_id, is_error = ask_backend(question)

        if is_error:
            st.error(answer)
        else:
            st.markdown(answer)
        if sources:
            render_sources(sources)

        new_msg = {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "conversation_id": conversation_id,
            "feedback": None,
            "is_error": is_error,
        }
        if not is_error:
            render_feedback(new_msg, key=str(len(st.session_state.messages)))

    st.session_state.messages.append(new_msg)
