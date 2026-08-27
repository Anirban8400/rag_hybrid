import os
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
import requests
from dotenv import load_dotenv

load_dotenv()

import numpy as np
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parent


# --- 1. Document Ingestion & Splitting ---

def process_pdfs(pdf_directory: str):
    """Loads all PDFs in a directory into LangChain Documents."""
    pdf_dir = Path(pdf_directory)
    pdf_documents = []
    pdf_files = list(pdf_dir.glob("**/*.pdf"))
    print(f"Found {len(pdf_files)} PDF file(s) in '{pdf_directory}'")

    for pdf_file in pdf_files:
        try:
            loader = PDFPlumberLoader(str(pdf_file))
            documents = loader.load()

            for doc in documents:
                doc.metadata["source"] = pdf_file.name
                doc.metadata["file_type"] = "pdf"

            pdf_documents.extend(documents)
            print(f"Loaded: {pdf_file.name}")
        except Exception as e:
            print(f"Error loading {pdf_file.name}: {e}")

    print(f"Total PDF pages loaded: {len(pdf_documents)}")
    return pdf_documents


def split_docs(documents, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Splits loaded documents into overlapping chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)


# --- 2. Dense Embeddings (Hugging Face) ---

class EmbeddingMan:
    def __init__(self, model_name: str = "google/embeddinggemma-300m"):
        self.model_name = model_name
        self.api_url = f"https://router.huggingface.co/hf-inference/models/{self.model_name}/pipeline/feature-extraction"

        token = os.environ.get("HF_API_TOKEN")
        if not token:
            raise ValueError("HF_API_TOKEN environment variable is missing.")

        self.headers = {"Authorization": f"Bearer {token}"}

    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        print(f"Fetching dense embeddings for {len(texts)} chunks from Hugging Face...")
        payload = {"inputs": texts}
        response = requests.post(self.api_url, headers=self.headers, json=payload)

        if response.status_code != 200:
            raise Exception(f"Hugging Face API Failed: {response.status_code} - {response.text}")

        embeddings_list = response.json()
        return np.array(embeddings_list)


# --- 3. Pinecone Hybrid Vector Store (Dense + BM25 Sparse) ---

class PineVectorStore:
    def __init__(self, index_name: str = "pdf-hybrid-index"):
        self.index_name = index_name
        pinecone_api_key = os.environ.get("PINECONE_API_KEY")
        if not pinecone_api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables.")

        self.pc = Pinecone(api_key=pinecone_api_key)
        print("Loading BM25 encoder dictionary...")
        self.bm25 = BM25Encoder.default()

    def ensure_index_exists(self, dimension: int):
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        if self.index_name not in existing_indexes:
            print(f"Creating Pinecone hybrid index '{self.index_name}'...")
            self.pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric="dotproduct",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        return self.pc.Index(self.index_name)

    def count(self) -> int:
        try:
            index = self.pc.Index(self.index_name)
            stats = index.describe_index_stats()
            return stats.total_vector_count or 0
        except Exception:
            return 0

    def clear_database(self):
        try:
            index = self.pc.Index(self.index_name)
            index.delete(delete_all=True)
            print(f"Cleared all vectors from index '{self.index_name}'.")
        except Exception as e:
            print(f"Note on clearing index: {e}")

    def add_documents(self, documents: List[Any], dense_embeddings: np.ndarray):
        if len(documents) != len(dense_embeddings):
            raise ValueError("Document count does not match dense embedding count.")

        dimension = len(dense_embeddings[0])
        index = self.ensure_index_exists(dimension)

        vectors_to_upsert = []
        print("Generating BM25 sparse vectors and preparing upsert payloads...")

        for i, (doc, dense_emb) in enumerate(zip(documents, dense_embeddings)):
            sparse_emb = self.bm25.encode_documents(doc.page_content)

            metadata = {
                "source": str(doc.metadata.get("source", "unknown")),
                "file_type": str(doc.metadata.get("file_type", "pdf")),
                "page": doc.metadata.get("page", None),
                "doc_index": i,
                "content_length": len(doc.page_content),
                "text": doc.page_content
            }

            vectors_to_upsert.append({
                "id": f"doc_{uuid.uuid4().hex[:8]}_{i}",
                "values": dense_emb.tolist(),
                "sparse_values": sparse_emb,
                "metadata": metadata
            })

        batch_size = 100
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i + batch_size]
            index.upsert(vectors=batch)
            print(f"Upserted batch {i // batch_size + 1}")


VectorStore = PineVectorStore


# --- 4. Hybrid Retriever + Cloud API Cross-Encoder Reranker ---

class RAGRetriever:
    def __init__(
        self,
        vector_store: PineVectorStore,
        embedding_manager: EmbeddingMan,
        reranker_model_name: str = "BAAI/bge-reranker-v2-m3"
    ):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        self.reranker_model_name = reranker_model_name
        
        # Connect directly to Hugging Face Inference API for the reranker
        self.api_url = f"https://router.huggingface.co/hf-inference/models/{self.reranker_model_name}"
        token = os.environ.get("HF_API_TOKEN")
        if not token:
            raise ValueError("HF_API_TOKEN is missing. Required for Cloud Reranking.")
        self.headers = {"Authorization": f"Bearer {token}"}

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        candidate_top_k: int = 10
    ) -> List[Dict[str, Any]]:
        index = self.vector_store.pc.Index(self.vector_store.index_name)

        # 1. Generate Query Embeddings (Dense & Sparse)
        query_dense = self.embedding_manager.create_embeddings([query])[0].tolist()
        query_sparse = self.vector_store.bm25.encode_queries(query)

        # 2. Hybrid Search on Pinecone
        hybrid_response = index.query(
            vector=query_dense,
            sparse_vector=query_sparse,
            top_k=candidate_top_k,
            include_metadata=True
        )

        matches = hybrid_response.get("matches", [])
        if not matches:
            return []

        # 3. Cloud API Cross-Encoder Reranking
        print(f"Reranking candidates via Hugging Face API ({self.reranker_model_name})...")
        
        # Format payload for Hugging Face Cross-Encoder text classification
        payload = {
            "inputs": [{"text": query, "text_pair": match["metadata"]["text"]} for match in matches]
        }
        
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            api_results = response.json()
            
            cross_scores = []
            for res in api_results:
                # HF classification API usually returns a list of dictionaries per input pair
                # We extract the score safely, assuming the model returns a direct score or standard label format.
                if isinstance(res, list) and len(res) > 0 and "score" in res[0]:
                    # Find the score for 'LABEL_1' (relevant), fallback to the highest class score
                    positive_score = next((cls["score"] for cls in res if cls.get("label") == "LABEL_1"), res[0]["score"])
                    cross_scores.append(float(positive_score))
                elif isinstance(res, dict) and "score" in res:
                    cross_scores.append(float(res["score"]))
                elif isinstance(res, float):
                    cross_scores.append(res)
                else:
                    cross_scores.append(0.0)
                    
        except Exception as e:
            print(f"Reranker API failed or timed out: {e}. Falling back to default Pinecone scores.")
            # If the API fails (e.g. rate limits or model is loading), fallback gracefully to Pinecone scores
            cross_scores = [float(match.get("score", 0.0)) for match in matches]

        candidates = []
        for match, score in zip(matches, cross_scores):
            candidates.append({
                "id": match["id"],
                "content": match["metadata"]["text"],
                "metadata": match["metadata"],
                "pinecone_score": float(match.get("score", 0.0)),
                "cross_encoder_score": float(score)
            })

        # Sort descending by cross-encoder score
        candidates.sort(key=lambda x: x["cross_encoder_score"], reverse=True)
        return candidates[:top_k]


def ingest_directory(vector_store: PineVectorStore, embedding_manager: EmbeddingMan, directory: str) -> int:
    """Convenience helper: load -> split -> embed -> upsert."""
    all_docs = process_pdfs(directory)
    chunks = split_docs(all_docs)
    if chunks:
        texts = [doc.page_content for doc in chunks]
        embeddings = embedding_manager.create_embeddings(texts)
        vector_store.add_documents(chunks, embeddings)
    return len(chunks) if chunks else 0