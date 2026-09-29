# 📄 DocuChat AI

An intelligent document Q&A application powered by LangChain, ChromaDB, and Groq LLM. Upload your PDFs and ask questions about their content!

## 🚀 Live App

**[Try DocuChat AI Now!](https://docuchat-ai-app.streamlit.app/)**

## ✨ Features

- 📤 **Upload Multiple PDFs** - Process one or more documents at once
- 🤖 **AI-Powered Q&A** - Ask questions and get accurate answers based on your documents
- 🔍 **Intelligent Search** - Uses vector embeddings and ChromaDB for semantic search
- 📊 **Source Attribution** - Know exactly which documents the answers come from
- 🎨 **Beautiful UI** - Modern, responsive Streamlit interface

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **LLM**: Groq (OpenAI GPT-OSS-120B)
- **Vector DB**: ChromaDB
- **Embeddings**: HuggingFace (all-MiniLM-L6-v2)
- **Document Processing**: LangChain
- **PDF Parsing**: PyPDF

## 📦 Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/shalinims2806/DocuChat-AI.git
cd docuchat-ai
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file and add your Groq API key:
```bash
GROQ_API_KEY=your_groq_api_key_here
```

5. Run the app:
```bash
streamlit run app.py
```

## 🔑 Getting a Groq API Key

1. Visit [Groq Console](https://console.groq.com)
2. Sign up/login
3. Navigate to API Keys
4. Create a new API key
5. Add it to your `.env` file

## 📝 Usage

1. **Upload PDFs** - Use the sidebar to upload one or more PDF files
2. **Ask Questions** - Type your question in the chat input
3. **Get Answers** - The AI will search your documents and provide accurate answers
4. **View Sources** - See which documents contributed to the answer

## 🏗️ Project Structure

```
docuchat-ai/
├── app.py                 # Main Streamlit application
├── backend/
│   ├── config.py         # Configuration settings
│   ├── main.py           # Backend server (FastAPI)
│   ├── pdf_processor.py  # PDF processing logic
│   └── rag_engine.py     # RAG chain implementation
├── frontend/
│   └── app.py            # Alternative frontend (Streamlit)
├── chroma_db/            # Vector database storage
├── uploads/              # Temporary file uploads
├── requirements.txt      # Python dependencies
└── SETUP.md             # Setup instructions
```

## 🔧 Configuration

Edit `backend/config.py` to customize:
- `CHUNK_SIZE` - Size of text chunks
- `CHUNK_OVERLAP` - Overlap between chunks
- `TOP_K_RESULTS` - Number of documents to retrieve
- Embedding model and LLM settings

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) - UI framework
- [LangChain](https://www.langchain.com/) - LLM orchestration
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [Groq](https://groq.com/) - LLM inference
- [HuggingFace](https://huggingface.co/) - Embeddings

---

Made with ❤️ by [Shalini](https://github.com/shalinims2806)
