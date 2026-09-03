"""
Stage 4: LLM Integration
Combines the retrieval pipeline (Step 3) with an LLM (Groq) to generate grounded,
language-matched answers with a hallucination guardrail.
"""

import os
from dotenv import load_dotenv
from groq import Groq
import chromadb
from sentence_transformers import SentenceTransformer
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0
load_dotenv()

CHROMA_DIR = "data/chroma_db"
COLLECTION_NAME = "rag_corpus"
EMBED_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "openai/gpt-oss-120b"  # current Groq general-purpose model (as of testing)

LANGUAGE_BOOST = 3.0
CANDIDATE_POOL = 15
TOP_K = 5

SYSTEM_PROMPT_TEMPLATE = """You are a helpful assistant answering questions about NADRA citizen \
services, digital ID, birth/death registration, and online safety - based ONLY on the \
provided context below.

Rules:
1. Answer in the SAME language as the user's question. If the question is in Urdu, answer \
   fully in Urdu. If in English, answer in English.
2. Use ONLY information from the context below. Do not use outside knowledge.
3. If the context does not contain enough information to answer the question, say clearly:
   - In English: "I don't have enough information in the provided documents to answer that."
   - In Urdu: "میرے پاس اس سوال کا جواب دینے کے لیے کافی معلومات دستیاب نہیں ہیں۔"
4. Keep answers concise and directly address the question.
5. Do not mention "the context" or "the documents" explicitly in your answer - just answer \
   naturally as if you know this information.

Context:
{context}
"""


def detect_query_language(query: str) -> str:
    try:
        lang = detect(query)
        return lang if lang in ("ur", "en") else "unknown"
    except Exception:
        return "unknown"


def retrieve(query, collection, embed_model, k=TOP_K):
    query_embedding = embed_model.encode([query])[0].tolist()
    query_lang = detect_query_language(query)

    results = collection.query(query_embeddings=[query_embedding], n_results=CANDIDATE_POOL)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    candidates = []
    for doc, meta, dist in zip(docs, metas, dists):
        adjusted = dist - LANGUAGE_BOOST if (query_lang != "unknown" and meta["language"] == query_lang) else dist
        candidates.append((adjusted, doc, meta))

    candidates.sort(key=lambda x: x[0])
    top = candidates[:k]
    return [{"text": doc, "source": meta["source"], "page": meta["page"]} for _, doc, meta in top]


def build_context(chunks) -> str:
    parts = []
    for c in chunks:
        parts.append(f"[Source: {c['source']}, page {c['page']}]\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def answer_query(query: str, collection, embed_model, groq_client):
    chunks = retrieve(query, collection, embed_model)
    context = build_context(chunks)

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)

    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        temperature=0.2,  # low temperature = more grounded, less creative
        max_tokens=500,
    )

    answer = response.choices[0].message.content
    sources = list({f"{c['source']} (p{c['page']})" for c in chunks})

    return {"answer": answer, "sources": sources}


def main():
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("ERROR: GROQ_API_KEY not found. Check your .env file.")
        return

    groq_client = Groq(api_key=groq_api_key)
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = chroma_client.get_collection(COLLECTION_NAME)

    test_queries = [
        "How do I register for a digital ID?",
        "پیدائش کے اندراج کے لیے کون سے دستاویزات درکار ہیں؟",
        "What is the capital of France?",  # should trigger the "I don't know" guardrail
    ]

    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        result = answer_query(q, collection, embed_model, groq_client)
        print(f"A: {result['answer']}")
        print(f"Sources: {result['sources']}")


if __name__ == "__main__":
    main()
