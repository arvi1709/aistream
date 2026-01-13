import os
from typing import Tuple, Optional, List

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "faiss_index")


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def build_vectorstore(
    kb_path: str,
    embeddings: Optional[GoogleGenerativeAIEmbeddings] = None,
    persist: bool = True,
) -> FAISS:
    """Builds and saves a FAISS vector store from the knowledge base file."""
    if embeddings is None:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    _ensure_dir(INDEX_DIR)

    if os.path.exists(os.path.join(INDEX_DIR, "index.faiss")):
        try:
            return FAISS.load_local(INDEX_DIR, embeddings)
        except Exception:
            pass  # fallback to rebuild

    with open(kb_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = [p.strip() for p in text.split("\n\n") if p.strip()]
    docs = [Document(page_content=c, metadata={"source": kb_path}) for c in chunks]

    vectorstore = FAISS.from_documents(docs, embeddings)

    if persist:
        try:
            vectorstore.save_local(INDEX_DIR)
        except Exception:
            pass

    return vectorstore


def _format_sources(docs: List[Document]) -> str:
    # Helper function to format the retrieved documents for the prompt
    parts = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        excerpt = d.page_content.strip()
        if len(excerpt) > 300:
            excerpt = excerpt[:297] + "..."
        parts.append(f"Source: {src}\n\"{excerpt}\"")
    return "\n\n".join(parts)


def answer_with_rag(
    question: str,
    vectorstore: FAISS,
    llm: Optional[ChatGoogleGenerativeAI] = None,
) -> Tuple[str, list]:
    if llm is None:
        # Use Gemini 2.5 Flash model for RAG answers
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # RAG chain to answer questions based on retrieved context
    template = """
    You are a helpful sales assistant for AutoStream.
    Use the following context to answer the user's question in a conversational way.
    Summarize the information to provide a clear answer.
    If the context doesn't contain the answer, just say you couldn't find the information.
    
    Context:
    {context}

    Question: {question}
    
    Helpful Answer:
    """
    prompt = PromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever | _format_sources, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    try:
        answer = rag_chain.invoke(question)
        docs = retriever.invoke(question)
        
        # If the LLM gives an empty response, fall back to showing the raw context
        if not answer or answer.strip() == "":
            context = _format_sources(docs)
            fallback = "I don't have a direct answer, but here is some information I found:\n\n" + context
            return fallback, docs
        
        return answer, docs
    except Exception as e:
        # If the chain fails, fall back to showing the raw context
        docs = retriever.get_relevant_documents(question)
        context = _format_sources(docs)
        answer = f"I encountered an error. Here is the information I was able to find:\n\n{context}"
        return answer, docs
