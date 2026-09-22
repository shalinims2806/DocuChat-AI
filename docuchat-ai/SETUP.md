# DocuChat AI — Setup Guide

## Prerequisites
- Python 3.10+
- VS Code
- [Ollama](https://ollama.com) installed

---

## Step 1 — Install Ollama & pull Llama3

Download Ollama from https://ollama.com and install it.
Then open a terminal and run:

```bash
ollama pull llama3
```

This downloads Llama3 (~4.7 GB). Do this once. After it finishes:

```bash
ollama serve
```

Keep this terminal open — Ollama must be running for the app to work.

---

## Step 2 — Open the project in VS Code

```bash
cd docuchat-ai
code .
```

---

## Step 3 — Create a virtual environment

In the VS Code terminal (Ctrl + `):

```bash
python -m venv venv
```

Activate it:
- **Windows:**  `venv\Scripts\activate`
- **Mac/Linux:** `source venv/bin/activate`

---

## Step 4 — Install dependencies

```bash
pip install -r requirements.txt
```

> First run downloads the embedding model (~90 MB). This is automatic.

---

## Step 5 — Start the backend (Terminal 1)

```bash
cd backend
uvicorn main:app --reload --port 8000
```

You should see: `Uvicorn running on http://127.0.0.1:8000`

---

## Step 6 — Start the frontend (Terminal 2)

Open a second VS Code terminal, activate the venv again, then:

```bash
streamlit run frontend/app.py
```

Your browser will open at `http://localhost:8501` 🎉

---

## How to use

1. Upload a PDF using the sidebar
2. Click **Process PDF** — wait for the success message
3. Type a question in the chat box
4. Get an answer with page-level source citations

---

## Project structure

```
docuchat-ai/
├── backend/
│   ├── config.py          # Settings (model name, paths, chunk size)
│   ├── pdf_processor.py   # PDF loading, chunking, embedding
│   ├── rag_engine.py      # RAG chain with Llama3 + ChromaDB
│   └── main.py            # FastAPI routes
├── frontend/
│   └── app.py             # Streamlit chat UI
├── uploads/               # Uploaded PDFs stored here
├── chroma_db/             # Vector embeddings stored here
├── requirements.txt
└── SETUP.md
```

---

## Switching models

Edit `backend/config.py` and change:
```python
OLLAMA_MODEL = "llama3"   # try "mistral" or "phi3" for faster responses
```

Then `ollama pull mistral` to download it.
