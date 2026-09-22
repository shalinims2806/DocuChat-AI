"""
rag_engine.py
RAG chain using LangChain 0.3.x LCEL style (no deprecated RetrievalQA).
"""

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from pdf_processor import get_vectorstore
from config import OLLAMA_MODEL, OLLAMA_BASE_URL, TOP_K_RESULTS


PROMPT_TEMPLATE = """You are a helpful, precise document assistant.
Use ONLY the context below to answer the question.
If the answer is not in the context, say: "I couldn't find that in the uploaded documents."
Always mention which document and page your answer comes from.

Context:
{context}

Question: {question}

Answer (concise, cite the source):"""


def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def query_documents(question: str) -> dict:
    llm = OllamaLLM(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,
    )

    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_RESULTS},
    )

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    # Retrieve source docs for citations
    source_docs = retriever.invoke(question)

    # LCEL chain
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(question)

    # Deduplicate sources
    seen    = set()
    sources = []
    for doc in source_docs:
        key = (doc.metadata.get("doc_name", ""), doc.metadata.get("page", ""))
        if key not in seen:
            seen.add(key)
            sources.append({
                "doc_name": doc.metadata.get("doc_name", "Unknown"),
                "page":     doc.metadata.get("page", "N/A"),
                "source":   doc.metadata.get("source", ""),
                "excerpt":  doc.page_content[:250] + "…",
            })

    return {"answer": answer, "sources": sources}
