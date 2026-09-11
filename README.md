# Citizen Services Multilingual RAG Assistant

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about NADRA digital ID, birth/death registration, family registration, and online safety — in **both Urdu and English** — grounded in official citizen-service documents.

🔗 **Live demo:** \[citizen-services-multilingual-rag-assistant-production.up.railway.app]

\---

## Overview

Most RAG tutorials are English-only. This project handles a genuinely harder problem: retrieving and answering accurately across **two languages with different scripts (Latin and Nastaliq Urdu)**, using real government citizen-service PDFs as the knowledge base.

Built end-to-end: data pipeline → embeddings → vector search → LLM generation → tested API → deployed web app.

## Features

* **Bilingual Q\&A** — ask in Urdu or English, get an answer in the same language
* **Grounded answers with citations** — every answer includes its source document and page number
* **Hallucination guardrail** — the model explicitly says when it doesn't have enough information, instead of making things up
* **Cross-lingual retrieval** — a question asked in one language can still surface relevant content in the other
* **RTL-aware chat UI** — Urdu text renders correctly right-to-left

## Architecture

```
PDFs (Urdu + English)
      │
      ▼
Text extraction (PyMuPDF) + OCR fallback (Tesseract) for scanned pages
      │
      ▼
Per-page language detection (langdetect)
      │
      ▼
Chunking (LangChain RecursiveCharacterTextSplitter, \~400 chars)
      │
      ▼
Multilingual embeddings (paraphrase-multilingual-MiniLM-L12-v2)
      │
      ▼
ChromaDB vector store + soft language-boost re-ranking
      │
      ▼
Groq LLM (openai/gpt-oss-120b) — grounded, language-matched generation
      │
      ▼
FastAPI backend  ──────►  React frontend (RTL-aware chat UI)
```

## Tech Stack

|Layer|Technology|
|-|-|
|Text extraction|PyMuPDF, Tesseract OCR + Poppler|
|Language detection|langdetect|
|Chunking|LangChain|
|Embeddings|fastembed (ONNX runtime, no PyTorch) — sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2|
|Vector store|ChromaDB|
|LLM|Groq API (openai/gpt-oss-120b)|
|Backend|FastAPI, Pydantic, pytest|
|Frontend|React (Vite), CSS|
|Deployment|Docker, Railway|

## Engineering Highlights

A few real problems this project ran into and how they were solved — included here because working through them was most of the actual engineering:

**1. Urdu text extraction was silently broken.**
The initial pipeline (`pdfplumber`) extracted Urdu text with characters in the wrong order — even from clean, non-scanned PDFs. This wasn't obvious until inspecting raw output. Switched to PyMuPDF, which fixed body text ordering (headings occasionally still reverse due to how PDF text runs are encoded).

**2. One source PDF was a scanned image, not real text.**
Detected via a near-zero language-detection success rate on that file. Added a Tesseract OCR fallback (`urd+eng` combined language model) specifically for that document.

**3. Hard language filtering broke cross-lingual retrieval.**
An early attempt at improving Urdu retrieval accuracy filtered search to same-language chunks only. This *looked* like an improvement in isolated tests, but it silently broke retrieval for documents that only existed in one language — an English query about a Urdu-only document would find nothing at all. Replaced it with a soft re-ranking boost (widen the candidate pool, mildly prefer same-language matches, never exclude cross-lingual results).

**4. Retrieval evaluation surfaced a real data imbalance.**
Built a 12-question bilingual evaluation set and measured hit-rate@5. One document (`ssoauthentication.pdf`) was consistently missed — it turned out to have only 7 chunks versus 100+ for other documents, so it was statistically drowned out. Documented rather than "fixed" with an artificial workaround, since this is a realistic constraint of small corpora.

**5. A pinned LLM model was deprecated mid-project.**
`llama-3.3-70b-versatile` returned a 404 from the Groq API partway through development — Groq had deprecated it. Swapped to `openai/gpt-oss-120b`.

**6. The deployed app crash-looped on Out-of-Memory errors — and the fix wasn't obvious.**
After deployment, the container kept restarting every \~10 seconds with no visible error — a classic silent OOM kill signature. The embedding stack (`sentence-transformers` + PyTorch) was exceeding the host's 1GB free-tier memory limit. Tried, in order: reducing PyTorch thread counts (no effect), switching to an ONNX backend within `sentence-transformers` (reduced memory but still crashed, and introduced an offline-mode bug), and using a quantized ONNX model variant (helped, but PyTorch was still being loaded internally as a dependency). The actual fix was switching to `fastembed`, a library that uses ONNX runtime directly with no PyTorch dependency at all — using the *same* embedding model, so retrieval quality was unaffected. This also required re-generating all 605 chunk embeddings and rebuilding the vector store with the new library.

## Evaluation

A 12-question bilingual test set (English + Urdu, including cross-lingual pairs) was used to measure retrieval quality.

**Hit-rate@5: 75% (9/12)**

Remaining gaps, with root causes identified:

* The OCR-extracted Urdu document has measurably weaker embeddings than natively-extracted text, so it's occasionally outranked by cleaner English content on the same topic.
* A small (7-chunk) document is under-represented against larger documents (100+ chunks) in the same corpus.

## Project Structure

```
├── main.py                  # FastAPI backend
├── test\_main.py              # pytest suite
├── requirements.txt
├── Dockerfile                 # multi-stage: builds frontend, packages with backend
├── scripts/
│   ├── preprocess\_v2.py       # PDF text extraction + language detection
│   ├── ocr\_fix.py              # OCR fallback for scanned PDFs
│   ├── chunk\_and\_embed.py     # chunking + embedding generation
│   ├── build\_vectorstore.py   # ChromaDB indexing + retrieval test
│   ├── evaluate\_retrieval.py  # hit-rate@k evaluation harness
│   └── llm\_integration.py     # standalone RAG + LLM test script
├── frontend/
│   └── src/App.jsx            # React chat UI
└── data/
    └── chroma\_db/              # persisted vector store
```

## Running Locally

```bash
# Backend
python -m venv venv
venv\\Scripts\\activate          # Windows
pip install -r requirements.txt
# add a .env file with: GROQ\_API\_KEY=your\_key\_here
uvicorn main:app --reload

# Frontend (separate terminal)
cd frontend
npm install
npm run dev
```

Run tests:

```bash
pytest test\_main.py -v
```

## Future Improvements

* Improve Urdu OCR quality (Nastaliq-specific model or a cloud OCR API) to close the retrieval gap on scanned documents
* Add authentication and rate limiting for public deployment
* Expand the evaluation set and add automated CI testing on every push
* Add conversation memory for multi-turn follow-up questions

\---

Built by Sidra Younas as a self-directed project to strengthen full-stack AI engineering skills — model integration, backend, and frontend — beyond the scope of a single FYP.

