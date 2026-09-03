"""
Basic tests for the RAG API. Run with: pytest test_main.py -v
(from the rag-assistant root folder, with the venv activated)

Note: these tests call the real Groq API and real ChromaDB store, so they
require a valid .env and a built vector store (Steps 1-3 completed).
"""

import pytest
from fastapi.testclient import TestClient
from main import app


@pytest.fixture(scope="module")
def client():
    # Using TestClient as a context manager triggers the app's lifespan
    # (startup/shutdown) events - without this, state['embed_model'] etc.
    # never get populated and every /query call fails.
    with TestClient(app) as c:
        yield c


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_returns_200(client):
    response = client.post("/query", json={"question": "How do I register for a digital ID?"})
    assert response.status_code == 200


def test_query_response_shape(client):
    response = client.post("/query", json={"question": "How do I register for a digital ID?"})
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert "language_detected" in data
    assert "latency_ms" in data
    assert isinstance(data["sources"], list)


def test_query_english_answer_is_english_source(client):
    response = client.post("/query", json={"question": "How do I get a Family Registration Certificate?"})
    data = response.json()
    assert any("FRC" in s for s in data["sources"])


def test_query_urdu_question_returns_sources(client):
    response = client.post("/query", json={"question": "پیدائش کے اندراج کے لیے کون سے دستاویزات درکار ہیں؟"})
    data = response.json()
    assert response.status_code == 200
    assert len(data["sources"]) > 0


def test_query_out_of_scope_triggers_guardrail(client):
    response = client.post("/query", json={"question": "What is the capital of France?"})
    data = response.json()
    assert "don't have enough information" in data["answer"].lower() or \
           "کافی معلومات" in data["answer"]


def test_query_empty_question_is_rejected(client):
    response = client.post("/query", json={"question": ""})
    assert response.status_code == 422


def test_query_missing_field_is_rejected(client):
    response = client.post("/query", json={})
    assert response.status_code == 422
