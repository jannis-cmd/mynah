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


def test_sql_query_readonly_accepts_select_and_audits(tmp_path, monkeypatch):
    test_db = tmp_path / "mynah_test.db"
    monkeypatch.setattr(main, "DB_PATH", test_db)
    _seed_hr_sample(test_db)

    with TestClient(main.app) as client:
        resp = client.post(
            "/tools/sql_query_readonly",
            json={
                "query": "SELECT COUNT(*) AS c FROM hr_sample WHERE device_id = ? LIMIT 1",
                "params": ["fixture_wearable_01"],
                "caller": "pytest",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["row_count"] == 1
        assert data["rows"][0]["c"] == 1

        audit = client.get("/tools/query_audit/recent", params={"limit": 5})
        assert audit.status_code == 200
        entries = audit.json()["entries"]
        assert entries[0]["status"] == "accepted"
        assert entries[0]["caller"] == "pytest"


def test_sql_query_readonly_rejects_without_limit(tmp_path, monkeypatch):
    test_db = tmp_path / "mynah_test.db"
    monkeypatch.setattr(main, "DB_PATH", test_db)
    _seed_hr_sample(test_db)

    with TestClient(main.app) as client:
        resp = client.post(
            "/tools/sql_query_readonly",
            json={"query": "SELECT * FROM hr_sample", "caller": "pytest"},
        )
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert detail["code"] == "SQL_LIMIT_REQUIRED"


def test_sql_query_readonly_rejects_mutating_statement(tmp_path, monkeypatch):
    test_db = tmp_path / "mynah_test.db"
    monkeypatch.setattr(main, "DB_PATH", test_db)
    _seed_hr_sample(test_db)

    with TestClient(main.app) as client:
        resp = client.post(
            "/tools/sql_query_readonly",
            json={"query": "DELETE FROM hr_sample WHERE 1 = 1 LIMIT 1", "caller": "pytest"},
        )
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert detail["code"] in {"SQL_STATEMENT_NOT_ALLOWED", "SQL_FORBIDDEN_KEYWORD"}

        audit = client.get("/tools/query_audit/recent", params={"limit": 5})
        assert audit.status_code == 200
        assert any(entry["status"] == "rejected" for entry in audit.json()["entries"])
