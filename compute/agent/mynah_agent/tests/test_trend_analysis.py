import sqlite3
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from app import main


def _seed_transcripts(db_path, run_id: str):
    with sqlite3.connect(db_path) as conn:
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
        start = datetime(2025, 7, 1, 20, 0, tzinfo=timezone.utc)
        for idx in range(20):
            created_at = (start + timedelta(days=idx)).isoformat()
            sleep_h = round(5.5 + idx * 0.08, 2)
            mood = round(4.2 + idx * 0.12, 2)
            stress = round(7.9 - idx * 0.10, 2)
            exercise_min = round(18 + idx * 1.4, 2)
            caffeine_late = 1 if idx % 3 == 0 else 0
            weekend = 1 if (start + timedelta(days=idx)).weekday() >= 5 else 0
            transcript_text = (
                f"Daily note. I slept {sleep_h} hours and mood was {mood}/10. "
                f"Stress {stress}/10. I did {exercise_min} minutes. "
                f"Late caffeine: {'yes' if caffeine_late else 'no'}. "
                f"[run_id={run_id};sleep_h={sleep_h};mood={mood};stress={stress};"
                f"exercise_min={exercise_min};caffeine_late={caffeine_late};weekend={weekend}]"
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO transcript(audio_id, text, model, path, created_at)
                VALUES(?, ?, 'fixture_transcript_v1', ?, ?)
                """,
                (f"{run_id}_{idx:03d}", transcript_text, f"/tmp/{run_id}_{idx:03d}.txt", created_at),
            )
        conn.commit()


def test_analysis_trends_prompt_driven_pipeline(tmp_path, monkeypatch):
    test_db = tmp_path / "mynah_test.db"
    monkeypatch.setattr(main, "DB_PATH", test_db)
    run_id = "pytest_trend_run_01"
    _seed_transcripts(test_db, run_id)

    with TestClient(main.app) as client:
        response = client.post(
            "/analysis/trends",
            json={
                "prompt": (
                    f"Analyze trends for run_id={run_id}. "
                    "Find correlation for sleep vs mood and exercise vs stress. "
                    "Find effect of late caffeine on mood and weekend on exercise."
                ),
                "caller": "pytest",
                "include_llm_summary": False,
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "ok"
        assert payload["record_count"] == 20
        assert payload["run_id"]

        metrics = {entry["name"]: entry for entry in payload["metrics"]}
        assert "corr_sleep_h_mood" in metrics
        assert metrics["corr_sleep_h_mood"]["value"] > 0
        corr_exercise_stress = metrics.get("corr_exercise_min_stress") or metrics.get("corr_stress_exercise_min")
        assert corr_exercise_stress is not None
        assert corr_exercise_stress["value"] < 0
        assert "delta_mood_by_caffeine_late" in metrics
        assert "delta_exercise_min_by_weekend" in metrics

        with sqlite3.connect(test_db) as conn:
            run = conn.execute(
                "SELECT status FROM analysis_run WHERE id = ?",
                (payload["run_id"],),
            ).fetchone()
            assert run is not None
            assert run[0] == "completed"
            step_count = conn.execute(
                "SELECT COUNT(*) FROM analysis_step WHERE run_id = ?",
                (payload["run_id"],),
            ).fetchone()[0]
            assert step_count >= 4


def test_analysis_trends_rejects_prompt_without_detectable_metrics(tmp_path, monkeypatch):
    test_db = tmp_path / "mynah_test.db"
    monkeypatch.setattr(main, "DB_PATH", test_db)
    run_id = "pytest_trend_run_02"
    _seed_transcripts(test_db, run_id)

    with TestClient(main.app) as client:
        response = client.post(
            "/analysis/trends",
            json={
                "prompt": f"Analyze run_id={run_id} and tell me how the day went.",
                "caller": "pytest",
                "include_llm_summary": False,
            },
        )
        assert response.status_code == 400
        assert "no trend metrics could be inferred" in response.json()["detail"]
