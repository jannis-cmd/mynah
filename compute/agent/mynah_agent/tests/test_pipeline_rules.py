import urllib.error
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from app import main


def _utc(ts: str) -> datetime:
    return datetime.fromisoformat(ts).astimezone(timezone.utc)


def test_resolve_note_ts_priority_exact_over_hint_and_upload() -> None:
    resolved, mode = main._resolve_note_ts(
        ts_hint="yesterday morning",
        source_ts=None,
        day_scope=False,
        upload_ts=_utc("2026-02-25T10:00:00+00:00"),
        explicit_candidates=[_utc("2026-02-24T07:30:00+00:00")],
        tz=ZoneInfo("UTC"),
    )
    assert resolved == _utc("2026-02-24T07:30:00+00:00")
    assert mode == "exact"


def test_resolve_note_ts_day_scope_forces_day_anchor() -> None:
    resolved, mode = main._resolve_note_ts(
        ts_hint="2026-02-20T08:15:00Z",
        source_ts=None,
        day_scope=True,
        upload_ts=_utc("2026-02-25T10:00:00+00:00"),
        explicit_candidates=[_utc("2026-02-20T08:15:00+00:00")],
        tz=ZoneInfo("UTC"),
    )
    assert resolved == _utc("2026-02-25T12:00:00+00:00")
    assert mode == "day"


def test_compaction_retries_fail_closed_on_model_error(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts: list[tuple[int, str]] = []

    def fail_generate(_: str) -> str:
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
