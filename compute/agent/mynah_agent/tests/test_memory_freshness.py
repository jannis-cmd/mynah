import sqlite3

from fastapi.testclient import TestClient

from app import main


def _seed_hr_sample(db_path):
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
            INSERT OR REPLACE INTO hr_sample(device_id, ts, bpm, quality, sensor_status, ingested_at)
            VALUES(?, ?, ?, ?, ?, ?)
            """,
            ("fixture_wearable_01", "2026-02-24T00:00:01+00:00", 66, 95, "ok", "2026-02-24T00:00:05+00:00"),
        )
        conn.commit()


def test_stale_memory_excluded_until_reverified(tmp_path, monkeypatch):
    test_db = tmp_path / "mynah_test.db"
    monkeypatch.setattr(main, "DB_PATH", test_db)
    _seed_hr_sample(test_db)

    with TestClient(main.app) as client:
        upsert = client.post(
            "/tools/memory_upsert",
            json={
                "type": "fact",
                "title": "Morning baseline",
                "summary": "Resting HR baseline around mid 60s.",
                "tags": ["hr", "baseline"],
                "sensitivity": "personal",
                "salience_score": 0.8,
                "confidence_score": 0.9,
                "caller": "pytest",
                "citations": [
                    {
                        "source_type": "hr_sample",
                        "source_ref": "fixture_wearable_01|2026-02-24T00:00:01+00:00",
                        "content_hash": "abababababababab",
                        "schema_version": 1,
                        "snapshot_ref": "2026-02-24",
                    }
                ],
            },
        )
        assert upsert.status_code == 200
        memory_id = upsert.json()["memory_id"]

    with sqlite3.connect(test_db) as conn:
        conn.execute(
            """
            UPDATE memory_item
            SET updated_at = ?
            WHERE id = ?
            """,
            ("2020-01-01T00:00:00+00:00", memory_id),
        )
        conn.commit()

    with TestClient(main.app) as client:
        verify_stale = client.get(f"/tools/memory_verify/{memory_id}")
        assert verify_stale.status_code == 200
        stale_payload = verify_stale.json()
        assert stale_payload["stale"] is True
        assert stale_payload["verified"] is False

        search_stale = client.post(
            "/tools/memory_search",
            json={"query": "baseline", "limit": 10, "verified_only": True},
        )
        assert search_stale.status_code == 200
        assert all(entry["id"] != memory_id for entry in search_stale.json()["entries"])

        reverify = client.post(f"/tools/memory_reverify/{memory_id}")
        assert reverify.status_code == 200
        assert reverify.json()["verification"]["verified"] is True
        assert reverify.json()["verification"]["stale"] is False

        search_fresh = client.post(
            "/tools/memory_search",
            json={"query": "baseline", "limit": 10, "verified_only": True},
        )
        assert search_fresh.status_code == 200
        assert any(entry["id"] == memory_id for entry in search_fresh.json()["entries"])
