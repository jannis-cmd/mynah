import sqlite3

from fastapi.testclient import TestClient

from app import main


def _seed_audio_with_hint(db_path, artifacts_path):
    transcript_fixture_dir = artifacts_path / "transcript_fixtures"
    transcript_fixture_dir.mkdir(parents=True, exist_ok=True)
    hint_path = transcript_fixture_dir / "fixture_note_01.txt"
    hint_path.write_text("I felt better after the walk today.", encoding="utf-8")

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audio_note (
                id TEXT PRIMARY KEY,
                device_id TEXT NOT NULL,
                start_ts TEXT NOT NULL,
                end_ts TEXT NOT NULL,
                audio_path TEXT NOT NULL,
                audio_sha256 TEXT NOT NULL,
                audio_bytes INTEGER NOT NULL,
                transcript_hint_path TEXT,
                source TEXT NOT NULL,
                transcription_state TEXT NOT NULL DEFAULT 'pending',
                ingested_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO audio_note(
                id, device_id, start_ts, end_ts, audio_path, audio_sha256, audio_bytes,
                transcript_hint_path, source, transcription_state, ingested_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "fixture_note_01",
                "fixture_wearable_01",
                "2026-02-24T10:00:00+00:00",
                "2026-02-24T10:00:05+00:00",
                str(artifacts_path / "audio" / "fixture_note_01.wav"),
                "aaaabbbbcccc",
                4,
                str(hint_path),
                "test",
                "pending",
                "2026-02-24T10:00:10+00:00",
            ),
        )
        conn.commit()


def test_pipeline_audio_transcribe_creates_transcript_and_memory(tmp_path, monkeypatch):
    test_db = tmp_path / "mynah_test.db"
    artifacts = tmp_path / "artifacts"
    monkeypatch.setattr(main, "DB_PATH", test_db)
    monkeypatch.setattr(main, "ARTIFACTS_PATH", artifacts)
    _seed_audio_with_hint(test_db, artifacts)

    with TestClient(main.app) as client:
        resp = client.post(
            "/pipeline/audio/transcribe",
            json={"audio_id": "fixture_note_01", "caller": "pytest"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["transcript_created"] is True
        assert data["memory_created"] is True
        assert data["memory_id"]

        recent = client.get("/tools/transcript/recent", params={"limit": 5})
        assert recent.status_code == 200
        entries = recent.json()["entries"]
        assert len(entries) == 1
        assert entries[0]["audio_id"] == "fixture_note_01"

        search = client.post(
            "/tools/memory_search",
            json={"query": "walk", "limit": 10, "verified_only": True},
        )
        assert search.status_code == 200
        assert len(search.json()["entries"]) == 1


def test_pipeline_audio_transcribe_requires_hint(tmp_path, monkeypatch):
    test_db = tmp_path / "mynah_test.db"
    artifacts = tmp_path / "artifacts"
    monkeypatch.setattr(main, "DB_PATH", test_db)
    monkeypatch.setattr(main, "ARTIFACTS_PATH", artifacts)

    with sqlite3.connect(test_db) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audio_note (
                id TEXT PRIMARY KEY,
                device_id TEXT NOT NULL,
                start_ts TEXT NOT NULL,
                end_ts TEXT NOT NULL,
                audio_path TEXT NOT NULL,
                audio_sha256 TEXT NOT NULL,
                audio_bytes INTEGER NOT NULL,
                transcript_hint_path TEXT,
                source TEXT NOT NULL,
                transcription_state TEXT NOT NULL DEFAULT 'pending',
                ingested_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO audio_note(
                id, device_id, start_ts, end_ts, audio_path, audio_sha256, audio_bytes,
                transcript_hint_path, source, transcription_state, ingested_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "fixture_note_02",
                "fixture_wearable_01",
                "2026-02-24T10:00:00+00:00",
                "2026-02-24T10:00:05+00:00",
                str(artifacts / "audio" / "fixture_note_02.wav"),
                "ddddeeeeffff",
                4,
                None,
                "test",
                "pending",
                "2026-02-24T10:00:10+00:00",
            ),
        )
        conn.commit()

    with TestClient(main.app) as client:
        resp = client.post(
            "/pipeline/audio/transcribe",
            json={"audio_id": "fixture_note_02", "caller": "pytest"},
        )
        assert resp.status_code == 400
        assert resp.json()["detail"]["code"] == "TRANSCRIPT_HINT_MISSING"
