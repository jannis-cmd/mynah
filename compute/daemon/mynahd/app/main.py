import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Query
from pydantic import BaseModel, Field, field_validator

SERVICE = os.getenv("MYNAH_SERVICE_NAME", "mynahd")
DB_PATH = Path(os.getenv("MYNAH_DB_PATH", "/data/db/mynah.db"))

app = FastAPI(title="mynahd", version="0.1.0")


class HrSampleIn(BaseModel):
    ts: datetime
    bpm: int = Field(ge=20, le=260)
    quality: int = Field(default=100, ge=0, le=100)
    sensor_status: str = Field(default="ok", max_length=64)

    @field_validator("ts")
    @classmethod
    def normalize_ts(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("ts must include timezone information")
        return value.astimezone(timezone.utc)


class HrIngestRequest(BaseModel):
    device_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9._:-]+$")
    samples: list[HrSampleIn] = Field(min_length=1, max_length=3600)
    source: str = Field(default="simulated", max_length=32)


def _init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS service_heartbeat (
                service_name TEXT PRIMARY KEY,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS device (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                fw_version TEXT,
                added_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS hr_sample (
                device_id TEXT NOT NULL,
                ts TEXT NOT NULL,
                bpm INTEGER NOT NULL,
                quality INTEGER NOT NULL,
                sensor_status TEXT NOT NULL,
                ingested_at TEXT NOT NULL,
                PRIMARY KEY(device_id, ts),
                FOREIGN KEY(device_id) REFERENCES device(id)
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_hr_sample_ts ON hr_sample(ts)
            """
        )
        conn.commit()


def _heartbeat() -> str:
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO service_heartbeat(service_name, updated_at)
            VALUES(?, ?)
            ON CONFLICT(service_name) DO UPDATE SET updated_at=excluded.updated_at
            """,
            (SERVICE, now),
        )
        conn.commit()
    return now


def _ensure_device(conn: sqlite3.Connection, device_id: str) -> None:
    conn.execute(
        """
        INSERT INTO device(id, type, fw_version, added_at)
        VALUES(?, 'wearable', NULL, ?)
        ON CONFLICT(id) DO NOTHING
        """,
        (device_id, datetime.now(timezone.utc).isoformat()),
    )


@app.on_event("startup")
def startup() -> None:
    _init_db()
    _heartbeat()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": SERVICE}


@app.get("/ready")
def ready() -> dict:
    ts = _heartbeat()
    return {"status": "ready", "service": SERVICE, "db_path": str(DB_PATH), "heartbeat_at": ts}


@app.post("/ingest/hr")
def ingest_hr(payload: HrIngestRequest) -> dict:
    ingested_at = datetime.now(timezone.utc).isoformat()
    rows = [
        (
            payload.device_id,
            sample.ts.isoformat(),
            sample.bpm,
            sample.quality,
            sample.sensor_status,
            ingested_at,
        )
        for sample in payload.samples
    ]
    with sqlite3.connect(DB_PATH) as conn:
        _ensure_device(conn, payload.device_id)
        conn.executemany(
            """
            INSERT INTO hr_sample(device_id, ts, bpm, quality, sensor_status, ingested_at)
            VALUES(?, ?, ?, ?, ?, ?)
            ON CONFLICT(device_id, ts) DO UPDATE SET
                bpm=excluded.bpm,
                quality=excluded.quality,
                sensor_status=excluded.sensor_status,
                ingested_at=excluded.ingested_at
            """,
            rows,
        )
        conn.commit()
    return {
        "status": "ok",
        "device_id": payload.device_id,
        "accepted_samples": len(rows),
        "source": payload.source,
        "ingested_at": ingested_at,
    }


@app.get("/summary/hr/today")
def summary_hr_today(
    date: str | None = Query(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    device_id: str | None = Query(default=None, min_length=1, max_length=64),
) -> dict:
    day = date or datetime.now(timezone.utc).date().isoformat()
    start = f"{day}T00:00:00+00:00"
    end = f"{day}T23:59:59.999999+00:00"
    params: list[str] = [start, end]
    where = "ts >= ? AND ts <= ?"
    if device_id:
        where += " AND device_id = ?"
        params.append(device_id)

    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            f"""
            SELECT
                COUNT(*) AS sample_count,
                MIN(bpm) AS min_bpm,
                MAX(bpm) AS max_bpm,
                AVG(bpm) AS avg_bpm
            FROM hr_sample
            WHERE {where}
            """,
            params,
        ).fetchone()

    sample_count = int(row[0] or 0)
    avg_bpm = round(float(row[3]), 2) if row[3] is not None else None
    return {
        "status": "ok",
        "date": day,
        "device_id": device_id,
        "sample_count": sample_count,
        "min_bpm": int(row[1]) if row[1] is not None else None,
        "max_bpm": int(row[2]) if row[2] is not None else None,
        "avg_bpm": avg_bpm,
    }


@app.get("/status")
def status() -> dict:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT updated_at FROM service_heartbeat WHERE service_name = ?",
            (SERVICE,),
        ).fetchone()
        hr_count = conn.execute("SELECT COUNT(*) FROM hr_sample").fetchone()
    return {
        "service": SERVICE,
        "last_heartbeat": row[0] if row else None,
        "hr_sample_count_total": int(hr_count[0] if hr_count else 0),
    }
