from fastapi.testclient import TestClient

from app import main


def _payload(device_id: str, bpms: list[int]) -> dict:
    base_ts = [
        "2026-02-24T00:00:01+00:00",
        "2026-02-24T00:00:02+00:00",
        "2026-02-24T00:00:03+00:00",
    ]
    return {
        "device_id": device_id,
        "source": "test",
        "samples": [
            {"ts": ts, "bpm": bpm, "quality": 95, "sensor_status": "ok"}
            for ts, bpm in zip(base_ts, bpms, strict=True)
        ],
    }


def test_ready_works_with_temp_db(tmp_path, monkeypatch):
    test_db = tmp_path / "mynah_test.db"
    monkeypatch.setattr(main, "DB_PATH", test_db)

    with TestClient(main.app) as client:
        resp = client.get("/ready")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ready"
    assert data["db_path"].endswith("mynah_test.db")


def test_ingest_hr_is_idempotent_upsert(tmp_path, monkeypatch):
    test_db = tmp_path / "mynah_test.db"
    monkeypatch.setattr(main, "DB_PATH", test_db)
    device_id = "fixture_wearable_01"

    with TestClient(main.app) as client:
        first = client.post("/ingest/hr", json=_payload(device_id, [62, 67, 71]))
        assert first.status_code == 200
        assert first.json()["accepted_samples"] == 3

        second = client.post("/ingest/hr", json=_payload(device_id, [70, 80, 90]))
        assert second.status_code == 200

        summary = client.get("/summary/hr/today", params={"date": "2026-02-24", "device_id": device_id})
        assert summary.status_code == 200
        data = summary.json()
        assert data["sample_count"] == 3
        assert data["min_bpm"] == 70
        assert data["max_bpm"] == 90
        assert data["avg_bpm"] == 80.0


def test_ingest_hr_requires_timezone_in_ts(tmp_path, monkeypatch):
    test_db = tmp_path / "mynah_test.db"
    monkeypatch.setattr(main, "DB_PATH", test_db)
    payload = {
        "device_id": "fixture_wearable_01",
        "samples": [
            {"ts": "2026-02-24T00:00:01", "bpm": 62, "quality": 95, "sensor_status": "ok"},
        ],
    }

    with TestClient(main.app) as client:
        resp = client.post("/ingest/hr", json=payload)

    assert resp.status_code == 422
