"""
Stage 3b: Retrieval Evaluation
Runs a set of hand-written test questions (English + Urdu) against the vector store
and measures hit-rate@k - i.e. how often the correct source document appears in the
top-k retrieved chunks. This is the number to report in your README/resume.
"""

import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DIR = "data/chroma_db"
COLLECTION_NAME = "rag_corpus"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
K = 5  # how many chunks to retrieve per query

# ---- Test set ----
# Each entry: a question + the source PDF(s) that would be a CORRECT retrieval.
# "expected_source" can be a single filename or a list - some topics (e.g. birth/death
# registration) are covered by both the KP and Punjab guides, so either counts as a hit.
# Mix of English and Urdu, including cross-lingual pairs (same topic asked in both languages).
TEST_SET = [
    {"query": "How do I register for a digital ID?", "expected_source": "digital_ID_Citizen_Guide.pdf"},
    {"query": "ڈیجیٹل آئی ڈی کے لیے رجسٹریشن کیسے کریں؟", "expected_source": "digital_ID_Citizen_Guide_Urdu.pdf"},
    {"query": "What documents are needed for birth registration?",
     "expected_source": ["Birth_Death_Public_Guide_KP.pdf", "Birth_Death_Public_Guide_Punjab.pdf"]},
    {"query": "پیدائش کے اندراج کے لیے کون سے دستاویزات درکار ہیں؟",
     "expected_source": ["Birth_Death_Public_Guide_KP.pdf", "Birth_Death_Public_Guide_Punjab.pdf"]},
    {"query": "How can children stay safe online?", "expected_source": "2025-04-15-Safety-Guied-English.pdf"},
    {"query": "بچوں کو انٹرنیٹ پر محفوظ کیسے رکھا جائے؟", "expected_source": "2025-04-15-online_safety_guide_urdu_08-03-2024.pdf"},
    {"query": "What is single sign-on authentication?", "expected_source": "ssoauthentication.pdf"},
    {"query": "How do I get a Family Registration Certificate?", "expected_source": "FRC Guide v2.pdf"},
    {"query": "What happens to a death registered in a hospital?",
     "expected_source": ["Birth_Death_Public_Guide_KP.pdf", "Birth_Death_Public_Guide_Punjab.pdf"]},
    {"query": "کیا ڈیجیٹل آئی ڈی محفوظ ہے؟", "expected_source": "digital_ID_Citizen_Guide_Urdu.pdf"},
    {"query": "What counts as harmful content online?", "expected_source": "2025-04-15-Safety-Guied-English.pdf"},
    {"query": "آن لائن نقصان دہ مواد کیا ہے؟", "expected_source": "2025-04-15-online_safety_guide_urdu_08-03-2024.pdf"},
]
# Adjust "expected_source" values above if any don't match your actual filenames -
# check data/processed/corpus.json for exact filenames used.


def load_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(COLLECTION_NAME)


from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0


def detect_query_language(query: str) -> str:
    try:
        lang = detect(query)
        return lang if lang in ("ur", "en") else "unknown"
    except Exception:
        return "unknown"


LANGUAGE_BOOST = 3.0  # subtracted from distance for same-language matches (lower distance = better)
CANDIDATE_POOL = 15   # retrieve this many candidates before re-ranking down to k


def retrieve(query, collection, model, k=K, language_aware=True):
    query_embedding = model.encode([query])[0].tolist()
    query_lang = detect_query_language(query) if language_aware else "unknown"

    results = collection.query(query_embeddings=[query_embedding], n_results=CANDIDATE_POOL)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    candidates = []
    for doc, meta, dist in zip(docs, metas, dists):
        adjusted = dist - LANGUAGE_BOOST if (query_lang != "unknown" and meta["language"] == query_lang) else dist
        candidates.append((adjusted, meta["source"]))

    candidates.sort(key=lambda x: x[0])
    return [source for _, source in candidates[:k]]


def main():
    model = SentenceTransformer(MODEL_NAME)
    collection = load_collection()

    hits = 0
    print(f"Running {len(TEST_SET)} test queries (k={K})...\n")

    for item in TEST_SET:
        query = item["query"]
        expected = item["expected_source"]
        expected_list = expected if isinstance(expected, list) else [expected]
        retrieved_sources = retrieve(query, collection, model)

        is_hit = any(exp in retrieved_sources for exp in expected_list)
        hits += is_hit

        status = "HIT " if is_hit else "MISS"
        print(f"[{status}] {query}")
        print(f"        expected (any of): {expected_list}")
        print(f"        got: {retrieved_sources}\n")

    hit_rate = hits / len(TEST_SET) * 100
    print("=" * 50)
    print(f"Hit-rate@{K}: {hits}/{len(TEST_SET)} = {hit_rate:.1f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()
