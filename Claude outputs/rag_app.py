import os
import tempfile
import streamlit as st
from groq import Groq
from dotenv import load_dotenv
import chromadb

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocuChat AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    #MainMenu { visibility: hidden; }
    header    { visibility: hidden; }
    footer    { visibility: hidden; }

    * { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }

    h1, h2 { color: #1E293B; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1E293B, #0F172A) !important;
    }
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span {
        color: #F1F5F9 !important;
    }
    [data-testid="stSidebar"] .stMarkdown p { color: #CBD5E1 !important; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #4F46E5, #06B6D4) !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 20px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(79,70,229,0.3) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(79,70,229,0.4) !important;
    }

    /* Chat messages */
    .stChatMessage { border-radius: 12px !important; }

    /* Empty state card */
    .empty-state {
        text-align: center;
        padding: 60px 20px;
        background: #F8FAFC;
        border-radius: 16px;
        border: 2px dashed #CBD5E1;
        margin: 20px 0;
        color: #64748B;
    }

    /* Source badge */
    .source-tag {
        display: inline-block;
        background: #EEF2FF;
        color: #4F46E5;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)


# ─── Cached Embedding Model ──────────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️ Loading embedding model (first run only)…")
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ─── Session State ───────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "chroma_client": chromadb.EphemeralClient(),
        "vectorstore": None,
        "documents": [],      # list of filenames already processed
        "chat_history": [],   # list of {role, content, sources?}
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_state()


# ─── Helpers ─────────────────────────────────────────────────────────────────
def get_api_key():
    """Read Groq key from Streamlit secrets (Cloud) or .env (local)."""
    try:
        key = st.secrets.get("GROQ_API_KEY")
        if key:
            return key
    except Exception:
        pass
    return os.getenv("GROQ_API_KEY")


def process_pdf(uploaded_file):
    """Extract + chunk a PDF; returns list of LangChain Documents."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    try:
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_documents(pages)
        for chunk in chunks:
            chunk.metadata["source"] = uploaded_file.name
        return chunks
    finally:
        os.unlink(tmp_path)


def add_to_vectorstore(chunks):
    """Add document chunks to the in-memory Chroma store."""
    embeddings = load_embeddings()
    if st.session_state.vectorstore is None:
        st.session_state.vectorstore = Chroma(
            client=st.session_state.chroma_client,
            collection_name="docuchat",
            embedding_function=embeddings,
        )
        st.session_state.vectorstore.add_documents(chunks)
    else:
        st.session_state.vectorstore.add_documents(chunks)


def get_answer(question: str):
    """Retrieve relevant docs and ask Groq."""
    if st.session_state.vectorstore is None:
        return "Please upload a PDF document first.", []

    api_key = get_api_key()
    if not api_key:
        return (
            "❌ GROQ_API_KEY not found. "
            "Add it under Settings → Secrets in Streamlit Cloud.",
            [],
        )

    # Retrieve top-4 chunks
    retriever = st.session_state.vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )
    docs = retriever.invoke(question)
    if not docs:
        return "I couldn't find relevant information in the uploaded documents.", []

    context = "\n\n---\n\n".join(d.page_content for d in docs)
    sources = list({d.metadata.get("source", "Unknown") for d in docs})

    prompt = f"""You are a helpful assistant that answers questions based only on the provided document context.

Context extracted from uploaded documents:
{context}

User question: {question}

Instructions:
- Answer ONLY using the context above.
- If the answer is not in the context, say "I couldn't find this information in the uploaded documents."
- Be concise, clear, and accurate.
- Quote or cite specific passages when helpful.

Answer:"""

    try:
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=1,
            max_completion_tokens=2048,
            top_p=1,
            stream=False,
            stop=None,
        )
        return completion.choices[0].message.content, sources

    except Exception as e:
        err = str(e)
        if "401" in err or "invalid" in err.lower():
            return "❌ Invalid Groq API Key. Please check your Streamlit secrets.", []
        if "429" in err or "rate" in err.lower():
            return "⏳ Rate limit reached — please wait a moment and try again.", []
        return f"❌ Error: {err}", []


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📄 DocuChat AI")
    st.markdown("*Ask questions about your PDFs*")
    st.markdown("---")

    st.markdown("### 📤 Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        for uf in uploaded_files:
            if uf.name not in st.session_state.documents:
                with st.spinner(f"Processing **{uf.name}**…"):
                    try:
                        chunks = process_pdf(uf)
                        add_to_vectorstore(chunks)
                        st.session_state.documents.append(uf.name)
                        st.success(f"✅ {uf.name} ready!")
                    except Exception as e:
                        st.error(f"Error: {e}")

    if st.session_state.documents:
        st.markdown("---")
        st.markdown("### 📚 Loaded Documents")
        for doc in st.session_state.documents:
            st.markdown(f"&nbsp;&nbsp;📄 {doc}")

        st.markdown("")
        if st.button("🗑️ Clear Everything", use_container_width=True):
            for key in ["vectorstore", "documents", "chat_history"]:
                st.session_state[key] = None if key == "vectorstore" else []
            st.session_state.chroma_client = chromadb.EphemeralClient()
            st.rerun()

    st.markdown("---")
    st.markdown("### ℹ️ How It Works")
    st.markdown("""
- Upload one or more PDFs
- Ask any question about their content
- AI searches the documents and answers accurately

**Powered by:**
- 🤖 Groq (LLM inference)
- 🔢 HuggingFace Embeddings
- 🗄️ ChromaDB (vector search)
- 🦜 LangChain RAG
""")


# ─── Main ────────────────────────────────────────────────────────────────────
st.markdown("## 📄 DocuChat AI")
st.markdown("*Upload your PDFs — then ask anything about them*")
st.markdown("---")

if not st.session_state.documents:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="empty-state">
            <div style="font-size:4rem;margin-bottom:16px">📂</div>
            <h3 style="color:#1E293B;margin-bottom:8px">No Documents Yet</h3>
            <p style="color:#64748B;font-size:1rem">
                Upload PDF files using the sidebar on the left to get started.
            </p>
        </div>
        """, unsafe_allow_html=True)

else:
    # Render existing chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("sources"):
                tags = " ".join(
                    f'<span class="source-tag">📄 {s}</span>'
                    for s in msg["sources"]
                )
                st.markdown(
                    f"<div style='margin-top:6px'>{tags}</div>",
                    unsafe_allow_html=True,
                )

    # Chat input
    if question := st.chat_input("Ask a question about your documents…"):
        st.session_state.chat_history.append(
            {"role": "user", "content": question}
        )
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching documents…"):
                answer, sources = get_answer(question)
            st.write(answer)
            if sources:
                tags = " ".join(
                    f'<span class="source-tag">📄 {s}</span>'
                    for s in sources
                )
                st.markdown(
                    f"<div style='margin-top:6px'>{tags}</div>",
                    unsafe_allow_html=True,
                )

        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#94A3B8;font-size:0.85rem;padding:10px'>
⚡ Powered by <b>Groq</b> + <b>LangChain</b> + <b>ChromaDB</b> | RAG Document Q&A
</div>
""", unsafe_allow_html=True)
