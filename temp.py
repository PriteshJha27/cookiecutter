# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
import api

# ─── Dummy Stubs ────────────────────────────────────────────────────────────────

class DummyPre:
    def __init__(self):
        self.initial_state = {}
    def execute(self):
        return {"final_result": "pre_out"}

class DummyFollow:
    def __init__(self):
        self.initial_state = {}
    def execute(self):
        return {"final_result": "follow_out", "retrieved_results": "ret"}

class DummyEncoder:
    def encode(self, text: str) -> str:
        return f"enc({text})"

class DummyHealth:
    def __init__(self, cfg):
        pass
    def check_system_health(self):
        return None

class DummyS3:
    def __init__(self):
        pass
    def download_file(self, *args, **kwargs):
        return None

# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def stub_everything(monkeypatch):
    # Stub out orchestrators & encoder
    monkeypatch.setattr(api, "LanggraphPreSummOrchestrator", DummyPre)
    monkeypatch.setattr(api, "LanggraphFollowUpOrchestrator", DummyFollow)
    monkeypatch.setattr(api, "Base64Encoding", DummyEncoder)

    # Stub out health check, config loader, and S3 utils
    monkeypatch.setattr(api, "HealthChecker", DummyHealth)
    monkeypatch.setattr(
        api,
        "load_config",
        lambda: {
            "data": {"bm25_path": "./tmp", "s3_folder": "sf"},
            "vectorstore": {"index_save_path": "./tmp"},
            "project": {"author": "a", "usecase": "u"},
        },
    )
    monkeypatch.setattr(api, "S3Utils", DummyS3)

    # start each test with no sessions
    api.active_sessions.clear()

@pytest.fixture
def client():
    return TestClient(api.app)

# ─── initialize_session (/fetchPreSummarizedResponse) ─────────────────────────

def test_initialize_session_success(client):
    payload = {
        "session_id": "s1",
        "files_selected": [],
        "contexts_selected": []
    }
    resp = client.post("/fetchPreSummarizedResponse", json=payload)
    assert resp.status_code == 200
    # JSONResponse(content="pre_out") → body is the bare string
    assert resp.json() == "pre_out"
    # side-effect: session is stored
    assert "s1" in api.active_sessions
    assert api.active_sessions["s1"]["state"] == "initialized"

def test_initialize_session_missing_fields(client):
    # session_id is required by Pydantic → 422
    resp = client.post("/fetchPreSummarizedResponse", json={
        "files_selected": [], "contexts_selected": []
    })
    assert resp.status_code == 422

# ─── execute_query (/fetchResponse) ────────────────────────────────────────────

def test_fetch_response_success(client):
    payload = {
        "session_id": "s2",
        "user_query": "hello",
        "files_selected": ["doc1"],
        "contexts_selected": [],
        "query_type": "doc"
    }
    resp = client.post("/fetchResponse", json=payload)
    assert resp.status_code == 200

    data = resp.json()
    # we stubbed DummyFollow + DummyEncoder
    assert data["response"] == "enc(follow_out)"
    assert isinstance(data["sources"], list) and len(data["sources"]) == 1

    src = data["sources"][0]
    assert src["source_content"] == "ret"
    # source_title is "_".join(files_selected)
    assert src["source_title"] == "doc1"

def test_fetch_response_invalid_payload(client):
    # missing user_query etc → 422
    resp = client.post("/fetchResponse", json={"session_id": "s3"})
    assert resp.status_code == 422

# ─── session status & deletion ────────────────────────────────────────────────

def test_get_session_status_not_found(client):
    resp = client.get("/sessions/doesnt_exist/status")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Session not found"

def test_get_session_status_after_init(client):
    # initialize one first
    client.post("/fetchPreSummarizedResponse", json={
        "session_id": "s4", "files_selected": [], "contexts_selected": []
    })
    resp = client.get("/sessions/s4/status")
    assert resp.status_code == 200
    assert resp.json() == {"status": "initialized"}

def test_delete_session_not_found(client):
    resp = client.delete("/sessions/none")
    assert resp.status_code == 404

def test_delete_session_success(client):
    # manually seed one
    api.active_sessions["to_delete"] = {"state": "initialized"}
    resp = client.delete("/sessions/to_delete")
    assert resp.status_code == 200
    assert resp.json() == {"status": "deleted"}
    assert "to_delete" not in api.active_sessions
