import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI

from retreival import RAGRetriever

SYSTEM_PROMPT = "You are a helpful assistant. Use the provided context to answer questions accurately and concisely."


def get_ai_client() -> OpenAI:
    """Creates OpenAI-compatible client pointed at the Hugging Face router."""
    load_dotenv()
    token = os.getenv("HF_API_TOKEN")
    if not token:
        raise RuntimeError("HF_API_TOKEN is missing from environment variables.")

    return OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=token,
    )


def rag_with_sources(
    query: str,
    retriever: RAGRetriever,
    client: OpenAI,
    model: str,
    session_id: str,  # <-- 1. Added session_id parameter
    top_k: int,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Retrieves context via hybrid + rerank, generates answer, and returns formatted metadata."""
    if not model or not model.strip():
        raise ValueError("A valid model identifier is mandatory.")

    # 2. Pass session_id to retrieve only from this user's namespace & BM25 model
    results = retriever.retrieve(query=query, session_id=session_id, top_k=top_k)

    if chat_history is None:
        chat_history = []

    if not results:
        return {
            "answer": "No relevant context found in uploaded documents.",
            "sources": []
        }

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for msg in chat_history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    context = "\n\n".join([doc["content"] for doc in results])

    prompt = f"""Context:
{context}

Question: {query}
Answer:"""

    messages.append({"role": "user", "content": prompt})

    print(f"\n--- EXECUTING LLM ---")
    print(f"Target Model: {model.strip()}")
    print(f"Session ID: {session_id}")
    print(f"Retrieved Chunks: {len(results)}")
    print(f"---------------------\n")

    response = client.chat.completions.create(
        model=model.strip(),
        messages=messages,
    )

    sources = []
    for doc in results:
        meta = doc["metadata"]
        sources.append({
            "source_file": meta.get("source", "Unknown"),
            "page": meta.get("page", None),
            "similarity_score": round(doc.get("cross_encoder_score", doc.get("pinecone_score", 0.0)), 4),
            "snippet": doc["content"][:250] + ("..." if len(doc["content"]) > 250 else ""),
            "full_metadata": meta
        })

    return {
        "answer": response.choices[0].message.content,
        "sources": sources
    }
