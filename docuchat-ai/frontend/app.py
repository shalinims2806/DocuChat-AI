"""
frontend/app.py  —  Streamlit UI for DocuChat AI
Run with:  streamlit run frontend/app.py
"""

import streamlit as st
import requests

API_URL = "http://localhost:8000"

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="DocuChat AI",
    page_icon="📄",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    .stChatMessage { border-radius: 12px; }
    .source-box {
        background: #f0f2f6;
        border-left: 4px solid #4f8bf9;
        padding: 8px 12px;
        border-radius: 6px;
        margin-bottom: 6px;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────
st.title("📄 DocuChat AI")
st.caption("Upload your PDFs and chat with them — powered by **Llama3** running locally on your machine.")

# ── Sidebar — upload & document list ─────────────────────────
with st.sidebar:
    st.header("📂 Upload a Document")
    uploaded = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded:
        if st.button("⚡ Process PDF", type="primary", use_container_width=True):
            with st.spinner(f"Embedding '{uploaded.name}'… this takes ~30s for the first run"):
                try:
                    resp = requests.post(
                        f"{API_URL}/upload",
                        files={"file": (uploaded.name, uploaded, "application/pdf")},
                        timeout=300,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        st.success(data["message"])
                        st.info(f"📦 {data['chunks_created']} chunks stored in vector DB")
                        st.rerun()
                    else:
                        st.error(f"Error: {resp.json().get('detail', 'Unknown error')}")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot reach backend. Is `uvicorn main:app` running?")

    st.divider()
    st.header("📋 Uploaded Documents")
    try:
        docs_resp = requests.get(f"{API_URL}/documents", timeout=5)
        docs      = docs_resp.json().get("documents", [])
        if docs:
            for doc in docs:
                st.markdown(f"📄 `{doc}`")
        else:
            st.info("No documents yet. Upload a PDF above.")
    except Exception:
        st.warning("⚠️ Backend not reachable.")

    st.divider()
    if st.button("🗑️ Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Chat interface ────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Welcome message
if not st.session_state.messages:
    st.info("👋 Upload a PDF in the sidebar, then ask me anything about it!")

# Render previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📎 {len(msg['sources'])} source(s) used"):
                for s in msg["sources"]:
                    st.markdown(
                        f"<div class='source-box'>"
                        f"<b>{s['doc_name']}</b> — Page {s['page']}<br>"
                        f"<i>{s['excerpt']}</i>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

# Chat input
if prompt := st.chat_input("Ask a question about your documents…"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get answer from backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking… (Llama3 is running locally)"):
            try:
                resp = requests.post(
                    f"{API_URL}/chat",
                    json={"question": prompt},
                    timeout=120,
                )
                if resp.status_code == 200:
                    data    = resp.json()
                    answer  = data["answer"]
                    sources = data["sources"]

                    st.markdown(answer)

                    if sources:
                        with st.expander(f"📎 {len(sources)} source(s) used"):
                            for s in sources:
                                st.markdown(
                                    f"<div class='source-box'>"
                                    f"<b>{s['doc_name']}</b> — Page {s['page']}<br>"
                                    f"<i>{s['excerpt']}</i>"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )

                    st.session_state.messages.append({
                        "role":    "assistant",
                        "content": answer,
                        "sources": sources,
                    })
                else:
                    err = resp.json().get("detail", "Unknown error")
                    st.error(f"Backend error: {err}")

            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot reach backend. Is `uvicorn main:app` running?")
            except requests.exceptions.Timeout:
                st.warning("⏳ Llama3 is taking long — it's still thinking. Refresh in a moment.")
