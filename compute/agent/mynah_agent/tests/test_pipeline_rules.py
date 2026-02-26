import urllib.error
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from app import main


def _utc(ts: str) -> datetime:
    return datetime.fromisoformat(ts).astimezone(timezone.utc)


def test_resolve_default_anchor_ts_source_priority() -> None:
    resolved, mode = main._resolve_default_anchor_ts(
        source_ts=_utc("2026-02-25T18:00:00+00:00"),
        day_scope=False,
        upload_ts=_utc("2026-02-25T19:00:00+00:00"),
        explicit_candidates=[_utc("2026-02-24T07:30:00+00:00")],
        tz=ZoneInfo("UTC"),
    )
    assert resolved == _utc("2026-02-25T18:00:00+00:00")
    assert mode == "exact"


def test_resolve_group_hint_yesterday_evening() -> None:
    resolved, mode = main._resolve_group_hint_ts(
        hint="yesterday evening",
        anchor_ts=_utc("2026-02-25T14:03:05+00:00"),
        anchor_mode="upload",
        tz=ZoneInfo("UTC"),
    )
    assert resolved == _utc("2026-02-24T20:00:00+00:00")
    assert mode == "inferred"


def test_resolve_group_hint_today_keeps_anchor() -> None:
    resolved, mode = main._resolve_group_hint_ts(
        hint="today",
        anchor_ts=_utc("2026-02-25T14:03:05+00:00"),
        anchor_mode="upload",
        tz=ZoneInfo("UTC"),
    )
    assert resolved == _utc("2026-02-25T14:03:05+00:00")
    assert mode == "upload"


def test_resolve_group_hint_at_2pm_same_day() -> None:
    resolved, mode = main._resolve_group_hint_ts(
        hint="at 2pm",
        anchor_ts=_utc("2026-02-25T14:03:05+00:00"),
        anchor_mode="upload",
        tz=ZoneInfo("UTC"),
    )
    assert resolved == _utc("2026-02-25T14:00:00+00:00")
    assert mode == "inferred"


def test_compaction_retries_fail_closed_on_model_error(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts: list[tuple[int, str]] = []

    def fail_generate(_: str, response_format: object | None = None) -> str:
        assert response_format is not None
        raise urllib.error.URLError("model unavailable")

    def record_attempt(**kwargs: object) -> None:
        attempts.append((int(kwargs["attempt"]), str(kwargs["status"])))

    monkeypatch.setattr(main, "_ollama_generate", fail_generate)
    monkeypatch.setattr(main, "_audit_compaction", record_attempt)

    artifact = {
        "id": "art-1",
        "source_type": "manual_text",
        "content": "I felt tired this morning.",
        "upload_ts": _utc("2026-02-25T10:00:00+00:00"),
        "source_ts": None,
        "day_scope": False,
        "timezone": "UTC",
    }

    with pytest.raises(RuntimeError, match="compaction failed"):
        main._compact_with_retries(artifact, caller="test")

    assert len(attempts) == main.MAX_COMPACTION_RETRIES
    assert all(status == "rejected" for _, status in attempts)


def test_temporal_item_note_type_is_strict() -> None:
    with pytest.raises(ValueError, match="unsupported note_type"):
        main.TemporalItem(text="Some text", note_type="unknown")


def test_retrieval_request_normalizes_query() -> None:
    req = main.RetrievalRequest(query="   sleep trends   ", mode="hybrid")
    assert req.query == "sleep trends"


def test_query_has_recency_hint_detects_tokens() -> None:
    assert main._query_has_recency_hint("How was my sleep today?")
    assert not main._query_has_recency_hint("show long term baseline")


def test_extract_primary_model_text_prefers_response() -> None:
    text = main._extract_primary_model_text({"response": " ok ", "thinking": "{\"variants\": []}"})
    assert text == "ok"


def test_extract_primary_model_text_uses_thinking_when_response_empty() -> None:
    text = main._extract_primary_model_text({"response": "", "thinking": "{\"variants\": [\"a\"]}"})
    assert text == "{\"variants\": [\"a\"]}"


def test_expand_query_variants_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    outputs = iter(["not-json", "{\"variants\":[\"bad day causes\",\"recent rough day factors\"]}"])
    calls: list[object | None] = []

    def fake_generate(_prompt: str, response_format: object | None = None) -> str:
        calls.append(response_format)
        return next(outputs)

    monkeypatch.setattr(main, "_ollama_generate", fake_generate)
    variants = main._expand_query_variants("what patterns explain my bad days recently")

    assert variants == ["bad day causes", "recent rough day factors"]
    assert len(calls) == 2
