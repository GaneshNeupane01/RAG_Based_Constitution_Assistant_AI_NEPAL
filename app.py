import os
import streamlit as st
from qdrant_client import QdrantClient
from langchain_qdrant import (
    QdrantVectorStore,
    RetrievalMode,
    FastEmbedSparse
)
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from langchain_groq import ChatGroq
from data.chunks import split_markdown_into_chunks   
from dotenv import load_dotenv


load_dotenv()


st.set_page_config(
    page_title="Nepal Constitution AI",
    page_icon="🧑‍⚖️",
    layout="wide"
)

st.title("🧑‍⚖️ Nepal Constitution – AI Legal Assistant")
st.caption("Hybrid RAG (Dense + BM25) + Cross-Encoder Reranking")
st.write("✅ App booted successfully.")


# Cached Heavy Stuff
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

@st.cache_resource
def load_sparse_embeddings():
    return FastEmbedSparse(model_name="Qdrant/bm25")

@st.cache_resource
def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

@st.cache_resource
def load_vector_store():
    embeddings = load_embeddings()
    sparse_embeddings = load_sparse_embeddings()
    client = QdrantClient(path="./qdrant_db")

    return QdrantVectorStore(
        client=client,
        collection_name="nepal_law",
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID
    )

@st.cache_resource
def load_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2,
        max_tokens=600
    )



if not os.path.exists("./qdrant_db"):
    with st.spinner("📚 Building vector DB... this may take a while"):
        final_chunks = split_markdown_into_chunks("./data/converted_law.md")
        vector_store = QdrantVectorStore.from_documents(
            documents = final_chunks,
            embedding=load_embeddings(),
            sparse_embedding=load_sparse_embeddings(),
            path="./qdrant_db",
            collection_name="nepal_law",
            retrieval_mode=RetrievalMode.HYBRID,
        )
    st.success("✅ Vector DB built successfully. Please rerun the app.")


    

# User Input
query = st.text_input(
    "Ask a constitutional or legal question:",
    placeholder="e.g. What does Article 275 say about local governance?"
)

# Reranking
def rerank(query, docs, top_k=8):
    reranker = load_reranker()
    pairs = [(query, d.page_content) for d in docs]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [doc for doc, _ in ranked[:top_k]]


if query:
    with st.spinner("🔍 Searching constitution..."):
        vector_store = load_vector_store()
        retrieved = vector_store.similarity_search(query, k=20)
        reranked = rerank(query, retrieved)

        context = "\n\n".join(
            f"[Source {i+1}]\n{doc.page_content}"
            for i, doc in enumerate(reranked)
        )

    prompt = f"""
You are a constitutional law assistant for Nepal.

RULES:
- Use ONLY the provided context.
- Do NOT invent articles, clauses, or interpretations.
- If the answer is not found, say so explicitly.
- Use formal, neutral legal language.
- Reference article/section numbers when mentioned.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

    with st.spinner("🧠 Generating answer..."):
        llm = load_llm()
        response = llm.invoke(prompt)

    st.markdown("### ✅ Answer")
    st.write(response.content)

    with st.expander("📚 Retrieved Constitutional Sources"):
        for i, doc in enumerate(reranked):
            st.markdown(f"**Source {i+1}**")
            st.write(doc.page_content)
            st.markdown("---")
