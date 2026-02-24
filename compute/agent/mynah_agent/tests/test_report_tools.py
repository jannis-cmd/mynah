import sqlite3

from fastapi.testclient import TestClient

from app import main


def _seed_report_inputs(db_path):
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS hr_sample (
                device_id TEXT NOT NULL,
                ts TEXT NOT NULL,
                bpm INTEGER NOT NULL,
                quality INTEGER NOT NULL,
                sensor_status TEXT NOT NULL,
                ingested_at TEXT NOT NULL,
                PRIMARY KEY(device_id, ts)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS transcript (
                audio_id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                model TEXT NOT NULL,
                path TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.executemany(
            """
            INSERT OR REPLACE INTO hr_sample(device_id, ts, bpm, quality, sensor_status, ingested_at)
            VALUES(?, ?, ?, ?, ?, ?)
            """,
            [
                ("fixture_wearable_01", "2026-02-24T08:00:00+00:00", 61, 95, "ok", "2026-02-24T08:00:02+00:00"),
                ("fixture_wearable_01", "2026-02-24T08:00:01+00:00", 69, 95, "ok", "2026-02-24T08:00:03+00:00"),
            ],
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO transcript(audio_id, text, model, path, created_at)
            VALUES(?, ?, ?, ?, ?)
            """,
            (
                "fixture_note_01",
                "I felt better after a walk.",
                "fixture_transcript_v1",
                "/tmp/fixture_note_01.txt",
                "2026-02-24T09:00:00+00:00",
            ),
        )
        conn.commit()


def test_report_generate_and_recent(tmp_path, monkeypatch):
    test_db = tmp_path / "mynah_test.db"
    artifacts = tmp_path / "artifacts"
    monkeypatch.setattr(main, "DB_PATH", test_db)
    monkeypatch.setattr(main, "ARTIFACTS_PATH", artifacts)
    _seed_report_inputs(test_db)

    with TestClient(main.app) as client:
        generate = client.post(
            "/tools/report_generate",
            json={"date": "2026-02-24", "caller": "pytest"},
        )
        assert generate.status_code == 200
        payload = generate.json()
        assert payload["report_date"] == "2026-02-24"
        assert payload["hr_samples"] == 2
        assert payload["transcripts"] == 1

        report_path = artifacts / "reports" / "2026-02-24" / "report.md"
        assert report_path.exists()
        report_text = report_path.read_text(encoding="utf-8")
        assert "MYNAH Daily Report - 2026-02-24" in report_text
        assert "Notes transcribed today: 1" in report_text

        recent = client.get("/tools/report_recent", params={"limit": 5})
        assert recent.status_code == 200
        entries = recent.json()["entries"]
        assert len(entries) == 1
        assert entries[0]["report_date"] == "2026-02-24"
