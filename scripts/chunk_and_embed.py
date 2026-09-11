"""
Stage 2: Chunking & Embedding
Splits corpus.json into retrieval-sized chunks and generates multilingual
embedding vectors for each chunk. Output feeds directly into ChromaDB (Step 3).
"""

import json
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastembed import TextEmbedding

CORPUS_PATH = Path("data/processed/corpus.json")
OUT_PATH = Path("data/processed/chunks_with_embeddings.json")

# Multilingual model, no PyTorch dependency - understands Urdu and English
# in the same vector space. E5 models expect a "passage: " / "query: " prefix
# on the text they embed for best retrieval quality.
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

CHUNK_SIZE = 400        # characters per chunk (kept smaller since Urdu script is denser)
CHUNK_OVERLAP = 80      # overlap between consecutive chunks


def load_corpus() -> list[dict]:
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def chunk_records(records: list[dict]) -> list[dict]:
    """Split each page's text into overlapping chunks, keeping metadata attached."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "۔", ".", " ", ""],  # ۔ = Urdu full stop
    )

    chunks = []
    chunk_id = 0
    for record in records:
        pieces = splitter.split_text(record["text"])
        for piece in pieces:
            if len(piece.strip()) < 20:  # skip tiny fragments
                continue
            chunks.append({
                "chunk_id": f"chunk_{chunk_id:05d}",
                "text": piece.strip(),
                "source": record["source"],
                "page": record["page"],
                "language": record["language"],
            })
            chunk_id += 1
    return chunks


def embed_chunks(chunks: list[dict], model: TextEmbedding) -> list[dict]:
    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks (this may take a minute)...")
    embeddings = list(model.embed(texts))

    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding.tolist()

    return chunks


def main():
    print("Loading corpus...")
    records = load_corpus()
    print(f"Loaded {len(records)} page-records.\n")

    print("Chunking...")
    chunks = chunk_records(records)
    print(f"Created {len(chunks)} chunks.\n")

    lang_counts = {}
    for c in chunks:
        lang_counts[c["language"]] = lang_counts.get(c["language"], 0) + 1
    print(f"Chunk language breakdown: {lang_counts}\n")

    print(f"Loading embedding model: {MODEL_NAME} (first run will download it)...")
    model = TextEmbedding(model_name=MODEL_NAME)

    chunks = embed_chunks(chunks, model)

    print(f"\nSaving to {OUT_PATH}...")
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)

    print(f"Done. {len(chunks)} chunks with embeddings saved.")
    print(f"Embedding dimension: {len(chunks[0]['embedding'])}")


if __name__ == "__main__":
    main()
