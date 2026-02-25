import json
import os
from base64 import b64decode
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

import psycopg
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, field_validator, model_validator
from psycopg.rows import dict_row

SERVICE = os.getenv("MYNAH_SERVICE_NAME", "mynahd")
DATABASE_DSN = os.getenv("MYNAH_DATABASE_DSN", "postgresql://mynah:mynah@postgres:5432/mynah")
ARTIFACTS_PATH = Path(os.getenv("MYNAH_ARTIFACTS_PATH", "/home/appuser/data/artifacts"))

app = FastAPI(title="mynahd", version="0.4.0")


class HrSampleIn(BaseModel):
    ts: datetime
    bpm: int = Field(ge=20, le=260)
    quality: int = Field(default=100, ge=0, le=100)

    @field_validator("ts")
    @classmethod
    def normalize_ts(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("ts must include timezone information")
        return value.astimezone(timezone.utc)


class HrIngestRequest(BaseModel):
    device_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9._:-]+$")
    samples: list[HrSampleIn] = Field(min_length=1, max_length=3600)
    source: str = Field(default="simulated", max_length=64)


class HealthSampleIn(BaseModel):
    ts: datetime
    metric: str = Field(min_length=1, max_length=64)
    value_num: float | None = None
    value_json: dict[str, Any] | None = None
    unit: str | None = Field(default=None, max_length=32)
    quality: int | None = Field(default=None, ge=0, le=100)
    source: str | None = Field(default=None, max_length=64)

    @field_validator("ts")
    @classmethod
    def normalize_ts(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("ts must include timezone information")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def ensure_value(self) -> "HealthSampleIn":
        if self.value_num is None and self.value_json is None:
            raise ValueError("value_num or value_json is required")
        return self


class HealthIngestRequest(BaseModel):
    source: str = Field(min_length=1, max_length=64)
    samples: list[HealthSampleIn] = Field(min_length=1, max_length=3600)


class AudioIngestRequest(BaseModel):
    note_id: str | None = Field(default=None, min_length=4, max_length=80, pattern=r"^[A-Za-z0-9._:-]+$")
    device_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9._:-]+$")
    start_ts: datetime
    end_ts: datetime
    audio_b64: str = Field(min_length=8, max_length=1_500_000)
    transcript_hint: str | None = Field(default=None, max_length=100_000)
    source: str = Field(default="simulated", max_length=64)

    @field_validator("start_ts", "end_ts")
    @classmethod
    def normalize_ts(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("timestamp must include timezone information")
        return value.astimezone(timezone.utc)


def _db_conn() -> psycopg.Connection:
    return psycopg.connect(DATABASE_DSN, autocommit=False)


def _init_db() -> None:
    ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)
    with _db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute("CREATE SCHEMA IF NOT EXISTS core")
            cur.execute("CREATE SCHEMA IF NOT EXISTS health")
            cur.execute("CREATE SCHEMA IF NOT EXISTS memory")

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS core.service_heartbeat (
                    service_name TEXT PRIMARY KEY,
                    updated_at TIMESTAMPTZ NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS health.sample (
                    id BIGSERIAL PRIMARY KEY,
                    ts TIMESTAMPTZ NOT NULL,
                    metric TEXT NOT NULL,
                    value_num DOUBLE PRECISION,
                    value_json JSONB,
                    unit TEXT,
                    quality INTEGER,
                    source TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE(source, metric, ts),
                    CHECK (value_num IS NOT NULL OR value_json IS NOT NULL)
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_health_sample_ts ON health.sample(ts)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_health_sample_metric_ts ON health.sample(metric, ts)")

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS core.audio_note (
                    id TEXT PRIMARY KEY,
                    device_id TEXT NOT NULL,
                    start_ts TIMESTAMPTZ NOT NULL,
                    end_ts TIMESTAMPTZ NOT NULL,
                    audio_path TEXT NOT NULL,
                    audio_sha256 TEXT NOT NULL,
                    audio_bytes INTEGER NOT NULL,
                    transcript_hint_path TEXT,
                    source TEXT NOT NULL,
                    ingested_at TIMESTAMPTZ NOT NULL
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_audio_note_start_ts ON core.audio_note(start_ts)")

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS core.transcript (
                    audio_id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    model TEXT NOT NULL,
                    path TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL
                )
                """
            )
        conn.commit()


def _heartbeat() -> str:
    now = datetime.now(timezone.utc)
    with _db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO core.service_heartbeat(service_name, updated_at)
                VALUES(%s, %s)
                ON CONFLICT(service_name) DO UPDATE SET updated_at=excluded.updated_at
                """,
                (SERVICE, now),
            )
        conn.commit()
    return now.isoformat()


def _audio_artifact_paths(note_id: str, start_ts: datetime) -> tuple[Path, Path]:
    audio_dir = ARTIFACTS_PATH / "audio" / start_ts.strftime("%Y") / start_ts.strftime("%m")
    transcript_fixture_dir = ARTIFACTS_PATH / "transcript_fixtures"
    audio_dir.mkdir(parents=True, exist_ok=True)
    transcript_fixture_dir.mkdir(parents=True, exist_ok=True)
    return audio_dir / f"{note_id}.wav", transcript_fixture_dir / f"{note_id}.txt"


def _persist_audio_note(
    cur: psycopg.Cursor,
    *,
    note_id: str,
    device_id: str,
    start_ts: datetime,
    end_ts: datetime,
    audio_bytes: bytes,
    transcript_hint: str | None,
    source: str,
    ingested_at: datetime,
) -> dict[str, Any]:
    audio_sha = sha256(audio_bytes).hexdigest()
    audio_path, transcript_fixture_path = _audio_artifact_paths(note_id, start_ts)
    audio_path.write_bytes(audio_bytes)

    transcript_hint_path_str = None
    if transcript_hint:
        transcript_fixture_path.write_text(transcript_hint.strip(), encoding="utf-8")
        transcript_hint_path_str = str(transcript_fixture_path)

    cur.execute(
        """
        INSERT INTO core.audio_note(
            id, device_id, start_ts, end_ts, audio_path, audio_sha256, audio_bytes,
            transcript_hint_path, source, ingested_at
        )
        VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT(id) DO UPDATE SET
            device_id=excluded.device_id,
            start_ts=excluded.start_ts,
            end_ts=excluded.end_ts,
            audio_path=excluded.audio_path,
            audio_sha256=excluded.audio_sha256,
            audio_bytes=excluded.audio_bytes,
            transcript_hint_path=excluded.transcript_hint_path,
            source=excluded.source,
            ingested_at=excluded.ingested_at
        """,
        (
            note_id,
            device_id,
            start_ts,
            end_ts,
            str(audio_path),
            audio_sha,
            len(audio_bytes),
            transcript_hint_path_str,
            source,
            ingested_at,
        ),
    )
    return {
        "audio_id": note_id,
        "audio_bytes": len(audio_bytes),
        "audio_sha256": audio_sha,
        "transcript_hint_available": bool(transcript_hint_path_str),
    }


@app.on_event("startup")
def startup() -> None:
    _init_db()
    _heartbeat()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": SERVICE}


@app.get("/ready")
def ready() -> dict[str, str]:
    ts = _heartbeat()
    return {"status": "ready", "service": SERVICE, "database": "postgres", "heartbeat_at": ts}


@app.post("/ingest/hr")
def ingest_hr(payload: HrIngestRequest) -> dict[str, Any]:
    ingested_at = datetime.now(timezone.utc)
    inserted = 0
    with _db_conn() as conn:
        with conn.cursor() as cur:
            for sample in payload.samples:
                cur.execute(
                    """
                    INSERT INTO health.sample(ts, metric, value_num, unit, quality, source, created_at)
                    VALUES(%s, 'hr', %s, 'bpm', %s, %s, %s)
                    ON CONFLICT(source, metric, ts) DO UPDATE SET
                        value_num=excluded.value_num,
                        unit=excluded.unit,
                        quality=excluded.quality
                    """,
                    (sample.ts, float(sample.bpm), sample.quality, payload.device_id, ingested_at),
                )
                inserted += 1
        conn.commit()

    return {
        "status": "ok",
        "device_id": payload.device_id,
        "inserted": inserted,
        "source": payload.source,
        "ingested_at": ingested_at.isoformat(),
    }


@app.post("/ingest/health")
def ingest_health(payload: HealthIngestRequest) -> dict[str, Any]:
    ingested_at = datetime.now(timezone.utc)
    inserted = 0
    with _db_conn() as conn:
        with conn.cursor() as cur:
            for sample in payload.samples:
                source = sample.source or payload.source
                cur.execute(
                    """
                    INSERT INTO health.sample(ts, metric, value_num, value_json, unit, quality, source, created_at)
                    VALUES(%s, %s, %s, %s::jsonb, %s, %s, %s, %s)
                    ON CONFLICT(source, metric, ts) DO UPDATE SET
                        value_num=excluded.value_num,
                        value_json=excluded.value_json,
                        unit=excluded.unit,
                        quality=excluded.quality
                    """,
                    (
                        sample.ts,
                        sample.metric,
                        sample.value_num,
                        json.dumps(sample.value_json) if sample.value_json is not None else None,
                        sample.unit,
                        sample.quality,
                        source,
                        ingested_at,
                    ),
                )
                inserted += 1
        conn.commit()

    return {"status": "ok", "inserted": inserted, "ingested_at": ingested_at.isoformat()}


@app.post("/ingest/audio")
def ingest_audio(payload: AudioIngestRequest) -> dict[str, Any]:
    if payload.end_ts <= payload.start_ts:
        raise HTTPException(status_code=422, detail="end_ts must be later than start_ts")
    try:
        audio_bytes = b64decode(payload.audio_b64, validate=True)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail="audio_b64 must be valid base64") from exc
    if not audio_bytes:
        raise HTTPException(status_code=422, detail="decoded audio payload is empty")

    note_id = payload.note_id or str(uuid4())
    ingested_at = datetime.now(timezone.utc)
    with _db_conn() as conn:
        with conn.cursor() as cur:
            persisted = _persist_audio_note(
                cur,
                note_id=note_id,
                device_id=payload.device_id,
                start_ts=payload.start_ts,
                end_ts=payload.end_ts,
                audio_bytes=audio_bytes,
                transcript_hint=payload.transcript_hint,
                source=payload.source,
                ingested_at=ingested_at,
            )
        conn.commit()

    return {
        "status": "ok",
        "audio_id": note_id,
        "device_id": payload.device_id,
        "audio_bytes": persisted["audio_bytes"],
        "audio_sha256": persisted["audio_sha256"],
        "transcript_hint_available": persisted["transcript_hint_available"],
        "ingested_at": ingested_at.isoformat(),
    }


@app.get("/summary/hr/today")
def summary_hr_today(date: str | None = Query(default=None), device_id: str | None = Query(default=None)) -> dict[str, Any]:
    if date:
        try:
            day = datetime.fromisoformat(f"{date}T00:00:00+00:00")
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="date must be YYYY-MM-DD") from exc
    else:
        day = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

    day_end = day.replace(hour=23, minute=59, second=59, microsecond=999999)
    params: list[Any] = [day, day_end]
    where = "metric = 'hr' AND ts >= %s AND ts <= %s"
    if device_id:
        where += " AND source = %s"
        params.append(device_id)

    with _db_conn() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                f"""
                SELECT
                    COUNT(*) AS sample_count,
                    MIN(value_num) AS min_bpm,
                    MAX(value_num) AS max_bpm,
                    ROUND(AVG(value_num)::numeric, 2) AS avg_bpm
                FROM health.sample
                WHERE {where}
                """,
                params,
            )
            row = cur.fetchone()

    return {
        "status": "ok",
        "date": day.date().isoformat(),
        "device_id": device_id,
        "sample_count": int(row["sample_count"] or 0),
        "min_bpm": int(row["min_bpm"]) if row["min_bpm"] is not None else None,
        "max_bpm": int(row["max_bpm"]) if row["max_bpm"] is not None else None,
        "avg_bpm": float(row["avg_bpm"]) if row["avg_bpm"] is not None else None,
    }


@app.get("/summary/audio/recent")
def summary_audio_recent(limit: int = Query(default=10, ge=1, le=100)) -> dict[str, Any]:
    with _db_conn() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT
                    a.id,
                    a.device_id,
                    a.start_ts,
                    a.end_ts,
                    a.audio_bytes,
                    CASE WHEN t.audio_id IS NULL THEN 0 ELSE 1 END AS transcript_ready
                FROM core.audio_note a
                LEFT JOIN core.transcript t ON t.audio_id = a.id
                ORDER BY a.start_ts DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()

    return {"status": "ok", "entries": rows}


@app.get("/status")
def status() -> dict[str, Any]:
    with _db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM health.sample")
            health_count = int(cur.fetchone()[0])
            cur.execute("SELECT COUNT(*) FROM core.audio_note")
            audio_count = int(cur.fetchone()[0])

    return {
        "service": SERVICE,
        "database": "postgres",
        "health_sample_count_total": health_count,
        "audio_note_count_total": audio_count,
    }
