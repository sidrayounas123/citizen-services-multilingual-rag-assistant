"""
Stage 5: FastAPI Backend
Wraps the RAG pipeline (retrieval + LLM generation) as a REST API.
Run with: uvicorn main:app --reload   (from the rag-assistant root folder)
"""

import os
import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from groq import Groq
import chromadb
from fastembed import TextEmbedding
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0
load_dotenv()

# ---- Config ----
CHROMA_DIR = "data/chroma_db"
COLLECTION_NAME = "rag_corpus"
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "openai/gpt-oss-120b"
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
5. Do not mention "the context" or "the documents" explicitly - just answer naturally.

Context:
{context}
"""

# ---- Logging ----
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("rag_api")

# ---- Globals populated at startup ----
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading embedding model, ChromaDB, and Groq client...")
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise RuntimeError("GROQ_API_KEY not set - check your .env file")

    state["groq_client"] = Groq(api_key=groq_api_key)
    state["embed_model"] = TextEmbedding(model_name=EMBED_MODEL_NAME)
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    state["collection"] = chroma_client.get_collection(COLLECTION_NAME)
    logger.info("Startup complete.")
    yield
    state.clear()


app = FastAPI(title="Multilingual RAG Assistant API", lifespan=lifespan)

# Allow the React dev server (different port) to call this API.
# For production deployment, replace "*" with your actual deployed frontend URL.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- Pydantic models ----
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)


class SourceChunk(BaseModel):
    source: str
    page: int


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    language_detected: str
    latency_ms: int


class HealthResponse(BaseModel):
    status: str


# ---- RAG logic (same as Step 4, adapted for reuse) ----
def detect_query_language(query: str) -> str:
    try:
        lang = detect(query)
        return lang if lang in ("ur", "en") else "unknown"
    except Exception:
        return "unknown"


def retrieve(query: str, k: int = TOP_K):
    embed_model = state["embed_model"]
    collection = state["collection"]

    query_embedding = list(embed_model.embed([query]))[0].tolist()
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
    chunks = [{"text": doc, "source": meta["source"], "page": meta["page"]} for _, doc, meta in top]
    return chunks, query_lang


def build_context(chunks) -> str:
    parts = [f"[Source: {c['source']}, page {c['page']}]\n{c['text']}" for c in chunks]
    return "\n\n---\n\n".join(parts)


# ---- Routes ----
@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    start = time.time()

    try:
        chunks, query_lang = retrieve(request.question)
        context = build_context(chunks)
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)

        response = state["groq_client"].chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.question},
            ],
            temperature=0.2,
            max_tokens=500,
        )
        answer = response.choices[0].message.content
        sources = list({f"{c['source']} (p{c['page']})" for c in chunks})

        latency_ms = int((time.time() - start) * 1000)
        logger.info(
            f"query='{request.question[:60]}' lang={query_lang} "
            f"chunks={len(chunks)} latency_ms={latency_ms}"
        )

        return QueryResponse(
            answer=answer,
            sources=sources,
            language_detected=query_lang,
            latency_ms=latency_ms,
        )

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Serve the built React frontend (run `npm run build` in frontend/ first).
# Mounted last and at "/" so API routes above take priority.
app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="frontend")
