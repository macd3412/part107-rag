"""Streamlit web UI for the Part 107 & Remote ID Drone FAQ RAG."""

import time

import streamlit as st
from streamlit_geolocation import streamlit_geolocation

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
    "Grounded in 14 CFR Part 107, Part 89, referenced Parts 48/61/91, FAA AC 107-2A, "
    "and live FAA TFRs (tfr.faa.gov). Educational prototype only—verify NOTAMs & LAANC before flight."
)

EXAMPLES = [
    "Are there any active TFRs near me right now?",
    "I moved. Do I have to tell the FAA?",
    "What is the maximum altitude and groundspeed under Part 107?",
    "What are my options for complying with Remote ID?",
]

with st.sidebar:
    use_agent = st.toggle("Agentic multi-step search", value=True)
    st.subheader("Your Location (Live TFRs)")
    loc = streamlit_geolocation()
    lat = loc.get("latitude") if isinstance(loc, dict) else None
    lon = loc.get("longitude") if isinstance(loc, dict) else None
    if lat is not None and lon is not None:
        st.success(f"GPS active: {lat:.4f}, {lon:.4f}")
    user_state = st.text_input(
        "Fallback State (2-letter code)",
        value="MA",
        max_chars=2,
        help="Used when GPS is not shared.",
    ).strip().upper()
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
        if msg.get("latency_s") is not None:
            saved_min = max(0.0, 10.0 - msg["latency_s"] / 60.0)
            st.caption(
                f"Answered in {msg['latency_s']:.1f}s "
                f"(~{saved_min:.1f} min saved vs. 10-min manual lookup)"
            )
        if msg.get("sources"):
            with st.expander(f"Retrieved Sources ({len(msg['sources'])})"):
                for s in msg["sources"]:
                    st.markdown(f"- **[{s['citation']}]({s['url']})** — {s['section_title']}")

prompt = st.chat_input("Ask a Part 107, Remote ID, or live TFR question...") or selected_example

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if lat is not None and lon is not None:
        loc_prefix = f"[User GPS: lat={lat:.4f}, lon={lon:.4f}, state={user_state}] "
    elif user_state:
        loc_prefix = f"[User state: {user_state}] "
    else:
        loc_prefix = ""

    with st.chat_message("assistant"):
        with st.spinner("Searching FAA regulations & live TFRs..."):
            t0 = time.perf_counter()
            if use_agent:
                answer, sources = get_agent_rag().ask(loc_prefix + prompt, verbose=False)
            else:
                answer, sources = get_single_rag().ask(prompt)
            latency_s = time.perf_counter() - t0
        st.markdown(answer)
        saved_min = max(0.0, 10.0 - latency_s / 60.0)
        st.caption(
            f"Answered in {latency_s:.1f}s "
            f"(~{saved_min:.1f} min saved vs. 10-min manual lookup)"
        )
        with st.expander(f"Retrieved Sources ({len(sources)})"):
            for s in sources:
                st.markdown(f"- **[{s['citation']}]({s['url']})** — {s['section_title']}")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "latency_s": latency_s,
        }
    )
