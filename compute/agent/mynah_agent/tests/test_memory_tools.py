import sqlite3

from fastapi.testclient import TestClient

from app import main


def _seed_hr_samples(db_path):
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
        conn.executemany(
            """
            INSERT OR REPLACE INTO hr_sample(device_id, ts, bpm, quality, sensor_status, ingested_at)
            VALUES(?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "fixture_wearable_01",
                    "2026-02-24T00:00:01+00:00",
                    62,
                    95,
                    "ok",
                    "2026-02-24T00:00:05+00:00",
                ),
                (
                    "fixture_wearable_01",
                    "2026-02-24T00:00:02+00:00",
                    67,
                    95,
                    "ok",
                    "2026-02-24T00:00:06+00:00",
                ),
            ],
        )
        conn.commit()


def test_memory_upsert_rejects_insight_with_single_citation(tmp_path, monkeypatch):
    test_db = tmp_path / "mynah_test.db"
    monkeypatch.setattr(main, "DB_PATH", test_db)
    _seed_hr_samples(test_db)

    payload = {
        "type": "insight",
        "title": "Morning trend",
        "summary": "HR trend rose after activity.",
        "tags": ["hr", "morning"],
        "sensitivity": "personal",
        "salience_score": 0.8,
        "confidence_score": 0.9,
        "caller": "pytest",
        "citations": [
            {
                "source_type": "hr_sample",
                "source_ref": "fixture_wearable_01|2026-02-24T00:00:01+00:00",
                "content_hash": "aaaaaaaaaaaaaaaa",
                "schema_version": 1,
                "snapshot_ref": "2026-02-24",
            }
        ],
    }

    with TestClient(main.app) as client:
        resp = client.post("/tools/memory_upsert", json=payload)
        assert resp.status_code == 400
        assert resp.json()["detail"]["code"] == "MEMORY_CITATION_MIN_NOT_MET"


def test_memory_upsert_supersession_and_verify(tmp_path, monkeypatch):
    test_db = tmp_path / "mynah_test.db"
    monkeypatch.setattr(main, "DB_PATH", test_db)
    _seed_hr_samples(test_db)

    with TestClient(main.app) as client:
        first = client.post(
            "/tools/memory_upsert",
            json={
                "type": "fact",
                "title": "Morning resting HR",
                "summary": "Typical morning resting HR is in the low 60s.",
                "tags": ["hr", "resting"],
                "sensitivity": "personal",
                "salience_score": 0.8,
                "confidence_score": 0.9,
                "caller": "pytest",
                "citations": [
                    {
                        "source_type": "hr_sample",
                        "source_ref": "fixture_wearable_01|2026-02-24T00:00:01+00:00",
                        "content_hash": "bbbbbbbbbbbbbbbb",
                        "schema_version": 1,
                        "snapshot_ref": "2026-02-24",
                    }
                ],
            },
        )
        assert first.status_code == 200
        first_id = first.json()["memory_id"]

        second = client.post(
            "/tools/memory_upsert",
            json={
                "type": "fact",
                "title": "Morning resting HR",
                "summary": "Updated baseline resting HR is mid 60s.",
                "tags": ["hr", "resting"],
                "sensitivity": "personal",
                "salience_score": 0.81,
                "confidence_score": 0.92,
                "caller": "pytest",
                "supersedes_memory_id": first_id,
                "citations": [
                    {
                        "source_type": "hr_sample",
                        "source_ref": "fixture_wearable_01|2026-02-24T00:00:02+00:00",
                        "content_hash": "cccccccccccccccc",
                        "schema_version": 1,
                        "snapshot_ref": "2026-02-24",
                    }
                ],
            },
        )
        assert second.status_code == 200
        second_id = second.json()["memory_id"]

        first_verify = client.get(f"/tools/memory_verify/{first_id}")
        assert first_verify.status_code == 200
        assert first_verify.json()["active"] is False
        assert first_verify.json()["superseded_by"] == second_id

        second_verify = client.get(f"/tools/memory_verify/{second_id}")
        assert second_verify.status_code == 200
        assert second_verify.json()["active"] is True
        assert second_verify.json()["verified"] is True

        search = client.post(
            "/tools/memory_search",
            json={"query": "resting", "limit": 10, "verified_only": True},
        )
        assert search.status_code == 200
        entries = search.json()["entries"]
        assert len(entries) == 1
        assert entries[0]["id"] == second_id
