"""
Stage 3: Vector Store & Retrieval
Loads chunks_with_embeddings.json into ChromaDB and provides a retrieval function.
Run this file directly to build the store and try a few test queries.
"""

import json
from pathlib import Path
import chromadb
from fastembed import TextEmbedding

CHUNKS_PATH = Path("data/processed/chunks_with_embeddings.json")
CHROMA_DIR = "data/chroma_db"
COLLECTION_NAME = "rag_corpus"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_chunks():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_store(chunks):
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # fresh collection each time this script is run (safe for iterating on the pipeline)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    print(f"Inserting {len(chunks)} chunks into ChromaDB...")
    # Chroma wants string ids, lists of embeddings, documents, and metadatas
    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=[c["embedding"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[{"source": c["source"], "page": c["page"], "language": c["language"]} for c in chunks],
    )
    print("Insert complete.\n")
    return collection


def retrieve(query: str, collection, model, k: int = 5):
    query_embedding = list(model.embed([query]))[0].tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=k)

    hits = []
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        hits.append({
            "text": doc,
            "source": meta["source"],
            "page": meta["page"],
            "language": meta["language"],
            "distance": dist,
        })
    return hits


def print_hits(query, hits):
    print(f"\nQuery: {query}")
    for i, hit in enumerate(hits, start=1):
        preview = hit["text"][:120].replace("\n", " ")
        print(f"  {i}. [{hit['language']}] {hit['source']} p{hit['page']} (dist={hit['distance']:.3f})")
        print(f"     {preview}...")


def main():
    chunks = load_chunks()
    model = TextEmbedding(model_name=MODEL_NAME)
    collection = build_store(chunks)

    # A few manual sanity-check queries - mix of English and Urdu,
    # deliberately testing cross-lingual retrieval (Urdu query -> may hit English chunks, etc.)
    test_queries = [
        "How do I register for a digital ID?",
        "پیدائش کا اندراج کیسے کریں؟",   # "how to register a birth?"
        "What is online safety for children?",
        "شناختی کارڈ کے لیے کیا دستاویزات درکار ہیں؟",  # "what documents needed for ID card?"
    ]

    for q in test_queries:
        hits = retrieve(q, collection, model, k=3)
        print_hits(q, hits)


if __name__ == "__main__":
    main()
