from pathlib import Path

# ── Directories ──────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
CHROMA_DIR = BASE_DIR / "chroma_db"

UPLOAD_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)

# ── Ollama (local model) ──────────────────────────────────────
OLLAMA_MODEL    = "llama3"
OLLAMA_BASE_URL = "http://localhost:11434"

# ── Ollama Embeddings (no sentence-transformers needed) ───────
EMBEDDING_MODEL = "nomic-embed-text"

# ── RAG tuning ────────────────────────────────────────────────
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200
TOP_K_RESULTS = 4
