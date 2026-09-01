# import os
# import shutil
# import tempfile 
# from pathlib import Path
# from typing import List, Optional
# from contextlib import asynccontextmanager
# import uvicorn

# from fastapi import FastAPI, UploadFile, File, HTTPException
# from pydantic import BaseModel

# from retreival import (
#     PineVectorStore,
#     EmbeddingMan,
#     RAGRetriever,
#     process_pdfs,
#     split_docs,
# )
# from llm import get_ai_client, rag_with_sources

# # Global references initialized on startup
# rag_retriever: Optional[RAGRetriever] = None
# ai_client = None
# vector_store: Optional[PineVectorStore] = None
# embedding_man: Optional[EmbeddingMan] = None


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     global rag_retriever, ai_client, vector_store, embedding_man
#     print("Initializing complete Hybrid RAG Pipeline...")

#     vector_store = PineVectorStore()
#     embedding_man = EmbeddingMan()
#     rag_retriever = RAGRetriever(vector_store=vector_store, embedding_manager=embedding_man)
#     ai_client = get_ai_client()

#     print(f"Pipeline ready. Vectors in Pinecone: {vector_store.count()}")
#     yield
#     print("Shutting down RAG service...")


# app = FastAPI(title="Hybrid RAG API", lifespan=lifespan)


# # Pydantic Schemas 

# class ChatMessage(BaseModel):
#     role: str
#     content: str


# class AskRequest(BaseModel):
#     text: str
#     model: str
#     session_id: str  # <-- NEW: Required field
#     top_k: int = 4
#     chat_history: List[ChatMessage] = []


# class SourceDoc(BaseModel):
#     source_file: str
#     page: Optional[int] = None
#     similarity_score: float
#     snippet: str
#     full_metadata: dict


# class AskResponse(BaseModel):
#     answer: str
#     sources: List[SourceDoc]


# #  Endpoints 

# @app.get("/health")
# def health_check():
#     return {
#         "status": "healthy",
#         "vector_store_chunks": vector_store.count() if vector_store else 0
#     }


# # Route now requires session_id in the URL path
# @app.post("/upload/{session_id}")
# def upload_pdfs(session_id: str, files: List[UploadFile] = File(...)):
#     if not files:
#         raise HTTPException(status_code=400, detail="No files uploaded.")

#     # Creates a unique, strictly isolated temporary directory for THIS user's request
#     with tempfile.TemporaryDirectory() as temp_dir:
#         dir_path = Path(temp_dir)

#         try:
#             vector_store.clear_namespace(session_id)

#             # Save uploaded files to the isolated folder
#             saved_paths = []
#             for file in files:
#                 dest_path = dir_path / file.filename
#                 with open(dest_path, "wb") as buffer:
#                     shutil.copyfileobj(file.file, buffer)
#                 saved_paths.append(dest_path)

#             # Process, embed, and store
#             docs = process_pdfs(str(dir_path))
#             chunks = split_docs(docs)

#             if not chunks:
#                 raise HTTPException(status_code=400, detail="No text could be extracted.")

#             texts = [doc.page_content for doc in chunks]
#             dense_embs = embedding_man.create_embeddings(texts)
            
#             # Pass session_id to properly isolate BM25 and Pinecone
#             vector_store.add_documents(chunks, dense_embs, texts, session_id)

#             return {
#                 "status": "success",
#                 "files_processed": len(saved_paths),
#                 "chunks_created": len(chunks)
#             }

#         except Exception as e:
#             raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}") from e
#         # automatically deletes the temp_dir when finished.


# @app.post("/ask_with_source", response_model=AskResponse)
# def ask_text(request: AskRequest):
#     if not request.text or not request.text.strip():
#         raise HTTPException(status_code=400, detail="Query text must not be empty.")

#     if not request.model or not request.model.strip():
#         raise HTTPException(status_code=400, detail="Model selection is mandatory.")

#     try:
#         hist_dicts = [msg.model_dump() for msg in request.chat_history]
#         answer = rag_with_sources(
#             query=request.text.strip(),
#             retriever=rag_retriever,
#             client=ai_client,
#             model=request.model.strip(),
#             top_k=request.top_k,
#             session_id=request.session_id,  # Pass session down to the LLM/Retriever
#             chat_history=hist_dicts,
#         )
#         return AskResponse(**answer)
#     except Exception as exc:
#         raise HTTPException(status_code=500, detail=f"RAG execution failed: {exc}") from exc


# if __name__ == "__main__":
#     uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)



import os
import shutil
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager
import uvicorn
import bcrypt
import jwt
from jwt.exceptions import ExpiredSignatureError, InvalidTokenError

from fastapi import Depends, FastAPI, UploadFile, File, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from retreival import (
    PineVectorStore,
    EmbeddingMan,
    RAGRetriever,
    process_pdfs,
    split_docs,
)
from llm import get_ai_client, rag_with_sources

# --- NEW: Import Database Functions ---
from database import (
    create_session,
    create_user,
    get_history,
    get_session,
    get_user_by_email,
    get_user_by_id,
    init_db,
    list_sessions,
    save_message,
    update_session,
)

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "480"))
MAX_PASSWORD_BYTES = 72
bearer_scheme = HTTPBearer()

# Global references initialized on startup
rag_retriever: Optional[RAGRetriever] = None
ai_client = None
vector_store: Optional[PineVectorStore] = None
embedding_man: Optional[EmbeddingMan] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_retriever, ai_client, vector_store, embedding_man
    print("Initializing complete Hybrid RAG Pipeline...")

    require_jwt_secret()

    # --- NEW: Initialize MongoDB Database ---
    init_db()

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
    session_id: str
    top_k: int = Field(..., ge=1, le=10)
    chat_history: List[ChatMessage] = Field(default_factory=list)


class SourceDoc(BaseModel):
    source_file: str
    page: Optional[int] = None
    similarity_score: float
    snippet: str
    full_metadata: dict


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceDoc]


class LoginRequest(BaseModel):
    email: str
    password: str


class SessionSummary(BaseModel):
    session_id: str
    label: str
    created_at: datetime
    updated_at: datetime


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    sessions: List[SessionSummary]


def require_jwt_secret():
    if not JWT_SECRET_KEY:
        raise RuntimeError("JWT_SECRET_KEY environment variable is missing.")


def create_access_token(user_id: str) -> str:
    require_jwt_secret()
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRE_MINUTES)
    return jwt.encode({"sub": user_id, "exp": expires_at}, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> dict:
    require_jwt_secret()
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        user_id = payload.get("sub")
        if not user_id:
            raise InvalidTokenError("Token has no subject.")
    except (ExpiredSignatureError, InvalidTokenError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token is invalid or expired.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User no longer exists.")
    return user


def get_owned_session(user_id: str, session_id: str) -> dict:
    session = get_session(user_id, session_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found.")
    return session


def label_from_query(query: str) -> str:
    words = query.split()
    if not words:
        return "New chat"
    label = " ".join(words[:3])
    return f"{label}{'…' if len(words) > 3 else ''}"


def password_bytes(password: str) -> bytes:
    encoded_password = password.encode("utf-8")
    if len(encoded_password) > MAX_PASSWORD_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"Password must not exceed {MAX_PASSWORD_BYTES} bytes.",
        )
    return encoded_password


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password_bytes(password), bcrypt.gensalt(rounds=12)).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password_bytes(password), password_hash.encode("utf-8"))
    except ValueError:
        return False


#  Endpoints 

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "vector_store_chunks": vector_store.count() if vector_store else 0
    }


@app.post("/login", response_model=LoginResponse)
def login(request: LoginRequest):
    email = request.email.strip().lower()
    password = request.password
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="A valid email address is required.")
    if not password:
        raise HTTPException(status_code=400, detail="A password is required.")

    user = get_user_by_email(email)
    if user is None:
        user_id = str(uuid.uuid4())
        user = create_user(user_id, email, hash_password(password))
    elif not verify_password(password, user["password_hash"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password.")

    return LoginResponse(
        access_token=create_access_token(user["user_id"]),
        user_id=user["user_id"],
        sessions=list_sessions(user["user_id"]),
    )


@app.get("/sessions", response_model=List[SessionSummary])
def fetch_sessions(current_user: dict = Depends(get_current_user)):
    return list_sessions(current_user["user_id"])


@app.post("/sessions", response_model=SessionSummary, status_code=status.HTTP_201_CREATED)
def new_session(current_user: dict = Depends(get_current_user)):
    session = create_session(current_user["user_id"], str(uuid.uuid4()))
    return SessionSummary(**{key: session[key] for key in SessionSummary.model_fields})


@app.get("/history/{session_id}")
def fetch_history(session_id: str, current_user: dict = Depends(get_current_user)):
    try:
        get_owned_session(current_user["user_id"], session_id)
        messages = get_history(current_user["user_id"], session_id)
        return {"messages": messages}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {str(e)}")


@app.post("/upload/{session_id}")
def upload_pdfs(
    session_id: str,
    files: List[UploadFile] = File(...),
    current_user: dict = Depends(get_current_user),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    get_owned_session(current_user["user_id"], session_id)

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


@app.post("/ask_with_source", response_model=AskResponse)
def ask_text(request: AskRequest, current_user: dict = Depends(get_current_user)):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Query text must not be empty.")

    if not request.model or not request.model.strip():
        raise HTTPException(status_code=400, detail="Model selection is mandatory.")

    try:
        session = get_owned_session(current_user["user_id"], request.session_id)

        # --- NEW: Save USER Message to MongoDB ---
        save_message(
            user_id=current_user["user_id"],
            session_id=request.session_id,
            role="user",
            content=request.text.strip()
        )
        if session["label"] == "New chat":
            update_session(
                current_user["user_id"], request.session_id, label_from_query(request.text.strip())
            )
        else:
            update_session(current_user["user_id"], request.session_id)

        # Enforce the same conversation window for every client, including
        # clients other than the Streamlit UI.
        hist_dicts = [msg.model_dump() for msg in request.chat_history[-10:]]
        answer = rag_with_sources(
            query=request.text.strip(),
            retriever=rag_retriever,
            client=ai_client,
            model=request.model.strip(),
            top_k=request.top_k,
            session_id=request.session_id,
            chat_history=hist_dicts,
        )

        # --- NEW: Save ASSISTANT Message to MongoDB ---
        save_message(
            user_id=current_user["user_id"],
            session_id=request.session_id,
            role="assistant",
            content=answer["answer"],
            sources=answer.get("sources", [])
        )

        return AskResponse(**answer)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG execution failed: {exc}") from exc


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
