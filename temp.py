import pytest
from fastapi.testclient import TestClient
from api import app  # your FastAPI app with both endpoints
from fuzzywuzzy import fuzz

client = TestClient(app)

def compute_similarity(response: str, ground_truth: str) -> int:
    """Compute fuzzy similarity score between two texts."""
    return fuzz.token_set_ratio(response.lower(), ground_truth.lower())


# --------------------------------------
# TEST: fetchPreSummarizedResponse
# --------------------------------------

@pytest.mark.parametrize("files,context,ground_truth_response", [
    (
        ["python_intro.pdf", "llm_basics.pdf"],
        "Focus on Python as a programming language",
        "Python is a high-level, interpreted language used widely for AI and data science."
    ),
])
def test_fetch_pre_summarized(files, context, ground_truth_response):
    payload = {
        "files_selected": files,
        "context_selected": context
    }

    response = client.post("/fetchPreSummarizedResponse", json=payload)
    assert response.status_code == 200

    result = response.json()["response"]
    similarity = compute_similarity(result, ground_truth_response)
    print(f"Similarity score: {similarity}")

    assert similarity > 75, f"Low similarity score ({similarity}).\nExpected: {ground_truth_response}\nGot: {result}"


# --------------------------------------
# TEST: fetchResponse
# --------------------------------------

@pytest.mark.parametrize("query,files,context,ground_truth_response", [
    (
        "What is a vector database?",
        ["vectordb_intro.pdf"],
        "Explain in simple terms",
        "A vector database stores data as numerical vectors for similarity-based search, useful in AI and semantic search."
    ),
    (
        "Explain what is RAG in AI?",
        ["rag_concept.pdf"],
        "No extra context",
        "RAG stands for Retrieval-Augmented Generation, a technique where a model retrieves relevant documents before answering."
    )
])
def test_fetch_response(query, files, context, ground_truth_response):
    payload = {
        "user_query": query,
        "files_selected": files,
        "context_selected": context
    }

    response = client.post("/fetchResponse", json=payload)
    assert response.status_code == 200

    result = response.json()["response"]
    similarity = compute_similarity(result, ground_truth_response)
    print(f"Similarity score: {similarity}")

    assert similarity > 75, f"Low similarity score ({similarity}).\nExpected: {ground_truth_response}\nGot: {result}"


# pip install pytest pytest-html fuzzywuzzy[speedup]
# pytest test/test_api.py --html=test_report.html --self-contained-html
