"""Tests for FastAPI web search API and Jinja2/htmx UI."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from findex.models import Index
from findex.web.app import create_app
from findex.web.deps import get_index


@pytest.fixture
def web_client(session_index: Index) -> Iterator[TestClient]:
    application = create_app()
    application.dependency_overrides[get_index] = lambda: session_index
    with TestClient(application) as client:
        application.state.index = session_index
        application.state.index_size_bytes = 1024
        yield client
    application.dependency_overrides.clear()


def test_health_ok_and_503_when_unloaded(web_client: TestClient) -> None:
    resp = web_client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "index_loaded": True}
    assert "X-Request-ID" in resp.headers

    web_client.app.state.index = None  # type: ignore[attr-defined]
    resp_down = web_client.get("/health")
    assert resp_down.status_code == 503
    assert resp_down.json() == {"status": "unavailable", "index_loaded": False}


def test_stats_endpoint(web_client: TestClient, session_index: Index) -> None:
    resp = web_client.get("/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["num_docs"] == session_index.num_docs
    assert data["num_terms"] == len(session_index)
    assert data["index_size_bytes"] == 1024
    assert data["uptime_seconds"] >= 0.0


def test_search_success_and_pagination(web_client: TestClient) -> None:
    resp = web_client.get("/search", params={"q": "alice", "k": 1, "page": 1, "scorer": "bm25"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["query"] == "alice"
    assert data["scorer"] == "bm25"
    assert data["page"] == 1
    assert data["k"] == 1
    assert data["total"] >= 1
    assert len(data["results"]) == 1
    assert "snippet" in data["results"][0]

    resp_tfidf = web_client.get("/search", params={"q": "alice", "scorer": "tfidf"})
    assert resp_tfidf.status_code == 200
    assert resp_tfidf.json()["scorer"] == "tfidf"


def test_search_validation_422(web_client: TestClient) -> None:
    assert web_client.get("/search").status_code == 422
    assert web_client.get("/search", params={"q": ""}).status_code == 422
    assert web_client.get("/search", params={"q": "alice", "k": "abc"}).status_code == 422
    assert web_client.get("/search", params={"q": "alice", "k": 0}).status_code == 422
    assert web_client.get("/search", params={"q": "alice", "scorer": "unknown"}).status_code == 422
    assert web_client.get("/search", params={"q": "alice", "page": 0}).status_code == 422


def test_get_document_api_and_404(web_client: TestClient, session_index: Index) -> None:
    first_id = next(iter(session_index.doc_meta))
    resp = web_client.get(f"/docs/{first_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["doc_id"] == first_id
    assert data["title"] == session_index.doc_meta[first_id].title

    resp_missing = web_client.get("/docs/999999")
    assert resp_missing.status_code == 404
    assert "detail" in resp_missing.json()


def test_openapi_and_ui_routes(web_client: TestClient, session_index: Index) -> None:
    assert web_client.get("/docs").status_code == 200
    assert web_client.get("/static/style.css").status_code == 200

    home = web_client.get("/")
    assert home.status_code == 200
    assert "findex" in home.text

    home_query = web_client.get("/", params={"q": "alice"})
    assert home_query.status_code == 200
    assert "alice" in home_query.text.lower()

    frag = web_client.get("/search-fragment", params={"q": "alice"})
    assert frag.status_code == 200
    assert "result-card" in frag.text

    first_id = next(iter(session_index.doc_meta))
    doc_ui = web_client.get(f"/doc/{first_id}")
    assert doc_ui.status_code == 200
    assert session_index.doc_meta[first_id].title in doc_ui.text
