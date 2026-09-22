"""
pdf_processor.py
Loads a PDF → splits into chunks → embeds via Ollama → stores in ChromaDB.
No sentence-transformers or torch required.
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from config import CHROMA_DIR, EMBEDDING_MODEL, OLLAMA_BASE_URL, CHUNK_SIZE, CHUNK_OVERLAP


def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


def process_pdf(file_path: str, doc_name: str) -> int:
    """Ingest a PDF into ChromaDB. Returns number of chunks stored."""
    loader = PyPDFLoader(file_path)
    pages  = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(pages)

    for chunk in chunks:
        chunk.metadata["doc_name"] = doc_name

    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=get_embeddings(),
        collection_name="docuchat",
    )
    vectorstore.add_documents(chunks)

    return len(chunks)


def get_vectorstore() -> Chroma:
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=get_embeddings(),
        collection_name="docuchat",
    )
