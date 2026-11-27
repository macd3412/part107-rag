import streamlit as st

from src.rag.pipeline import Part107RAG

st.set_page_config(page_title="Part 107 Drone FAQ", page_icon="🚁", layout="centered")

@st.cache_resource
def get_rag() -> Part107RAG:
    return Part107RAG()

st.title("Part 107 & Remote ID Drone FAQ")
st.caption(
    "Grounded in 14 CFR Part 107, Part 89, and FAA Advisory Circular AC 107-2A. "
    "Educational prototype only—always verify current FAA rules and LAANC before flying."
)

EXAMPLES = [
    "What is the maximum altitude and groundspeed under Part 107?",
    "What lighting is required to fly a small UAS at night?",
    "Who is eligible to apply to establish a FRIA?",
    "What are the Category 2 operations over people requirements?",
]

with st.sidebar:
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
        with st.spinner("Searching Part 107, Part 89 & AC 107-2A..."):
            rag = get_rag()
            answer, sources = rag.ask(prompt)
        st.markdown(answer)
        with st.expander(f"Retrieved Sources ({len(sources)})"):
            for s in sources:
                st.markdown(f"- **[{s['citation']}]({s['url']})** — {s['section_title']}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
