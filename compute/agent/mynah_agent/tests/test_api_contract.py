import pytest
from fastapi import HTTPException

from app import main


class _DummyCursor:
    def execute(self, *_args, **_kwargs):
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _DummyConn:
    def cursor(self):
        return _DummyCursor()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_api_route_contract_minimal_surface() -> None:
    routes = {
        (method, route.path)
        for route in main.app.routes
        if hasattr(route, "methods")
        for method in route.methods
    }
    expected = {
        ("GET", "/health"),
        ("GET", "/ready"),
        ("GET", "/ready/model"),
        ("POST", "/ingest/hr"),
        ("POST", "/ingest/health"),
        ("POST", "/ingest/audio"),
        ("GET", "/summary/hr/today"),
        ("GET", "/summary/audio/recent"),
        ("POST", "/pipeline/artifacts/ingest"),
        ("POST", "/pipeline/artifacts/process/{artifact_id}"),
        ("POST", "/pipeline/me_md/process"),
        ("POST", "/pipeline/audio/transcribe"),
        ("POST", "/pipeline/search/reindex/memory_notes"),
        ("POST", "/tools/retrieve"),
        ("POST", "/tools/report_generate"),
        ("GET", "/tools/report_recent"),
        ("GET", "/status"),
    }
    assert expected.issubset(routes)


def test_ready_is_core_runtime_not_model_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "_db_conn", lambda: _DummyConn())
    monkeypatch.setattr(main, "_assert_schema_ready", lambda _cur: None)
    monkeypatch.setattr(
        main,
        "_model_state",
        lambda: {
            "ollama_reachable": True,
            "generation_model_present": False,
            "embedding_model_present": False,
        },
    )

    payload = main.ready()
    assert payload["status"] == "ready"
    assert payload["model_state"]["generation_model_present"] is False


def test_ready_model_is_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        main,
        "_model_state",
        lambda: {
            "ollama_reachable": True,
            "generation_model_present": False,
            "embedding_model_present": True,
        },
    )
    with pytest.raises(HTTPException) as exc:
        main.ready_model()
    assert exc.value.status_code == 503
