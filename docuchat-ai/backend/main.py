"""
main.py  —  FastAPI backend for DocuChat AI
Run with:  uvicorn main:app --reload --port 8000
"""

import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pdf_processor import process_pdf
from rag_engine import query_documents
from config import UPLOAD_DIR


app = FastAPI(title="DocuChat AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    sources: list


# ── Routes ────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "DocuChat AI is running 🚀"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Accept a PDF, embed it, and store in ChromaDB."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    dest = UPLOAD_DIR / file.filename
    with open(dest, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    chunks = process_pdf(str(dest), file.filename)

    return {
        "message":        f"✅ '{file.filename}' processed successfully.",
        "chunks_created": chunks,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Answer a question using the RAG pipeline."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    return query_documents(req.question)


@app.get("/documents")
def list_documents():
    """List all uploaded PDF files."""
    files = [f.name for f in UPLOAD_DIR.iterdir() if f.suffix.lower() == ".pdf"]
    return {"documents": files, "count": len(files)}


@app.delete("/documents/{filename}")
def delete_document(filename: str):
    """Remove a document from the upload folder (does not remove from vector DB)."""
    target = UPLOAD_DIR / filename
    if not target.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    target.unlink()
    return {"message": f"'{filename}' deleted."}
