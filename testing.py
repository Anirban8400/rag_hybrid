import os
import time
from langchain_core.documents import Document
from retreival import EmbeddingMan, PineVectorStore, RAGRetriever

def run_test():
    # 1. Initialize Components with a dedicated test index
    TEST_INDEX_NAME = "test-hybrid-rag"
    
    print("--- 1. Initializing Components ---")
    embedding_manager = EmbeddingMan()
    vector_store = PineVectorStore(index_name=TEST_INDEX_NAME)
    retriever = RAGRetriever(vector_store, embedding_manager)

    # 2. Create Controlled Dummy Chunks
    # Case A: Technical code / acronym (tests BM25 sparse matching)
    # Case B: Semantic concept (tests Dense embedding matching)
    # Case C: Irrelevant noise / distractor
    dummy_docs = [
        Document(
            page_content="Error code ERR_9021: Database connection timed out on cluster alpha-east.",
            metadata={"source": "system_logs.txt", "category": "technical"}
        ),
        Document(
            page_content="When the database server becomes unresponsive, network latency spikes and queries fail.",
            metadata={"source": "architecture_guide.txt", "category": "conceptual"}
        ),
        Document(
            page_content="To bake chocolate chip cookies, preheat your oven to 180°C and mix butter with brown sugar.",
            metadata={"source": "cookbook.txt", "category": "recipes"}
        ),
        Document(
            page_content="Project Titan roadmap: Quarterly target is to migrate legacy microservices to Kubernetes.",
            metadata={"source": "roadmap.txt", "category": "planning"}
        )
    ]

    # 3. Ingest Dummy Documents
    print(f"\n--- 2. Ingesting {len(dummy_docs)} Dummy Chunks ---")
    texts = [doc.page_content for doc in dummy_docs]
    dense_embeddings = embedding_manager.create_embeddings(texts)
    vector_store.add_documents(dummy_docs, dense_embeddings)

    # Allow a brief moment for Pinecone to index newly upserted vectors
    print("Waiting 5 seconds for Pinecone index to settle...")
    time.sleep(5)

    # 4. Run Test Queries
    test_queries = [
        "What does ERR_9021 indicate?",                   # Keyword / Acronym test
        "How do I fix database network latency failures?", # Semantic meaning test
    ]

    print("\n--- 3. Executing Retrieval & Reranking Tests ---")
    for q in test_queries:
        print(f"\n==========================================")
        print(f"QUERY: \"{q}\"")
        print(f"==========================================")
        
        # Retrieve candidate_top_k=4 (all chunks) and final_top_k=2 (top 2 reranked)
        results = retriever.retrieve(query=q, final_top_k=2, candidate_top_k=4)

        for rank, res in enumerate(results, start=1):
            print(f"\n[Rank {rank}]")
            print(f"  Content:             {res['content']}")
            print(f"  Source:              {res['metadata'].get('source')}")
            print(f"  Pinecone Hybrid:     {res['pinecone_hybrid_score']:.4f}")
            print(f"  Cross-Encoder Score: {res['cross_encoder_score']:.4f}")

if __name__ == "__main__":
    run_test()