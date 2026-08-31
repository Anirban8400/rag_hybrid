import os
import shutil
import tempfile  # <-- NEW: For isolated file uploads
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager
import uvicorn

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from retreival import (
    PineVectorStore,
    EmbeddingMan,
    RAGRetriever,
    process_pdfs,
    split_docs,
)
from llm import get_ai_client, rag_with_sources

# Global references initialized on startup
rag_retriever: Optional[RAGRetriever] = None
ai_client = None
vector_store: Optional[PineVectorStore] = None
embedding_man: Optional[EmbeddingMan] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_retriever, ai_client, vector_store, embedding_man
    print("Initializing complete Hybrid RAG Pipeline...")

    vector_store = PineVectorStore()
    embedding_man = EmbeddingMan()
    rag_retriever = RAGRetriever(vector_store=vector_store, embedding_manager=embedding_man)
    ai_client = get_ai_client()

    print(f"Pipeline ready. Vectors in Pinecone: {vector_store.count()}")
    yield
    print("Shutting down RAG service...")


app = FastAPI(title="Hybrid RAG API", lifespan=lifespan)


# Pydantic Schemas 

class ChatMessage(BaseModel):
    role: str
    content: str


class AskRequest(BaseModel):
    text: str
    model: str
    session_id: str  # <-- NEW: Required field
    top_k: int = 4
    chat_history: List[ChatMessage] = []


class SourceDoc(BaseModel):
    source_file: str
    page: Optional[int] = None
    similarity_score: float
    snippet: str
    full_metadata: dict


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceDoc]


#  Endpoints 

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "vector_store_chunks": vector_store.count() if vector_store else 0
    }


# NEW: Route now requires session_id in the URL path
@app.post("/upload/{session_id}")
def upload_pdfs(session_id: str, files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    # NEW: Creates a unique, strictly isolated temporary directory for THIS user's request
    with tempfile.TemporaryDirectory() as temp_dir:
        dir_path = Path(temp_dir)

        try:
            # 1. Reset ONLY this user's namespace, not the global database
            vector_store.clear_namespace(session_id)

            # 2. Save uploaded files to the isolated folder
            saved_paths = []
            for file in files:
                dest_path = dir_path / file.filename
                with open(dest_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                saved_paths.append(dest_path)

            # 3. Process, embed, and store
            docs = process_pdfs(str(dir_path))
            chunks = split_docs(docs)

            if not chunks:
                raise HTTPException(status_code=400, detail="No text could be extracted.")

            texts = [doc.page_content for doc in chunks]
            dense_embs = embedding_man.create_embeddings(texts)
            
            # 4. Pass session_id to properly isolate BM25 and Pinecone
            vector_store.add_documents(chunks, dense_embs, texts, session_id)

            return {
                "status": "success",
                "files_processed": len(saved_paths),
                "chunks_created": len(chunks)
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}") from e
        # The 'with' block automatically deletes the temp_dir when finished.


@app.post("/ask_with_source", response_model=AskResponse)
def ask_text(request: AskRequest):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Query text must not be empty.")

    if not request.model or not request.model.strip():
        raise HTTPException(status_code=400, detail="Model selection is mandatory.")

    try:
        hist_dicts = [msg.model_dump() for msg in request.chat_history]
        answer = rag_with_sources(
            query=request.text.strip(),
            retriever=rag_retriever,
            client=ai_client,
            model=request.model.strip(),
            top_k=request.top_k,
            session_id=request.session_id,  # <-- NEW: Pass session down to the LLM/Retriever
            chat_history=hist_dicts,
        )
        return AskResponse(**answer)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG execution failed: {exc}") from exc


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)