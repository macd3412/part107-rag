"""Streamlit web UI for the Part 107 & Remote ID Drone FAQ RAG."""

import streamlit as st

from src.rag.agent import AgenticPart107RAG
from src.rag.pipeline import Part107RAG

st.set_page_config(page_title="Part 107 Drone FAQ", page_icon="🚁", layout="centered")


@st.cache_resource
def get_agent_rag() -> AgenticPart107RAG:
    """Load and cache the multi-step agentic RAG."""
    return AgenticPart107RAG()


@st.cache_resource
def get_single_rag() -> Part107RAG:
    """Load and cache the single-pass RAG."""
    return Part107RAG()


st.title("Part 107 & Remote ID Drone FAQ")
st.caption(
    "Grounded in 14 CFR Part 107, Part 89, referenced Parts 48/61/91, and FAA AC 107-2A. "
    "Educational prototype only—always verify current FAA rules and LAANC before flying."
)

EXAMPLES = [
    "I moved. Do I have to tell the FAA?",
    "What is the maximum altitude and groundspeed under Part 107?",
    "What lighting is required to fly a small UAS at night?",
    "What are my options for complying with Remote ID?",
]

with st.sidebar:
    use_agent = st.toggle("Agentic multi-step search", value=True)
    st.subheader("Try an example")
    selected_example = None
    for q in EXAMPLES:
        if st.button(q, use_container_width=True):
            selected_example = q

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"Retrieved Sources ({len(msg['sources'])})"):
                for s in msg["sources"]:
                    st.markdown(f"- **[{s['citation']}]({s['url']})** — {s['section_title']}")

prompt = st.chat_input("Ask a Part 107 or Remote ID question...") or selected_example

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching FAA regulations..."):
            if use_agent:
                answer, sources = get_agent_rag().ask(prompt, verbose=False)
            else:
                answer, sources = get_single_rag().ask(prompt)
        st.markdown(answer)
        with st.expander(f"Retrieved Sources ({len(sources)})"):
            for s in sources:
                st.markdown(f"- **[{s['citation']}]({s['url']})** — {s['section_title']}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
