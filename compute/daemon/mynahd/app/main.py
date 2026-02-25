import json
import os
from base64 import b64decode
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

import psycopg
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, field_validator
from psycopg.rows import dict_row

SERVICE = os.getenv("MYNAH_SERVICE_NAME", "mynahd")
DATABASE_DSN = os.getenv("MYNAH_DATABASE_DSN", "postgresql://mynah:mynah@postgres:5432/mynah")
ARTIFACTS_PATH = Path(os.getenv("MYNAH_ARTIFACTS_PATH", "/home/appuser/data/artifacts"))

app = FastAPI(title="mynahd", version="0.2.0")


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
    source: str = Field(default="simulated", max_length=64)


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
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS service_heartbeat (
                    service_name TEXT PRIMARY KEY,
                    updated_at TIMESTAMPTZ NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS device (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    fw_version TEXT,
                    added_at TIMESTAMPTZ NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS hr_sample (
                    device_id TEXT NOT NULL,
                    ts TIMESTAMPTZ NOT NULL,
                    bpm INTEGER NOT NULL,
                    quality INTEGER NOT NULL,
                    sensor_status TEXT NOT NULL,
                    ingested_at TIMESTAMPTZ NOT NULL,
                    PRIMARY KEY(device_id, ts),
                    FOREIGN KEY(device_id) REFERENCES device(id)
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_hr_sample_ts ON hr_sample(ts)")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS audio_note (
                    id TEXT PRIMARY KEY,
                    device_id TEXT NOT NULL,
                    start_ts TIMESTAMPTZ NOT NULL,
                    end_ts TIMESTAMPTZ NOT NULL,
                    audio_path TEXT NOT NULL,
                    audio_sha256 TEXT NOT NULL,
                    audio_bytes INTEGER NOT NULL,
                    transcript_hint_path TEXT,
                    source TEXT NOT NULL,
                    transcription_state TEXT NOT NULL DEFAULT 'pending',
                    ingested_at TIMESTAMPTZ NOT NULL,
                    FOREIGN KEY(device_id) REFERENCES device(id)
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_audio_note_start_ts ON audio_note(start_ts)")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS transcript (
                    audio_id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    model TEXT NOT NULL,
                    path TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS artifacts (
                    id TEXT PRIMARY KEY,
                    artifact_type TEXT NOT NULL,
                    source TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL,
                    content_text TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    object_uri TEXT,
                    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                    processing_state TEXT NOT NULL DEFAULT 'pending',
                    processed_at TIMESTAMPTZ
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_artifacts_created_at ON artifacts(created_at DESC)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_artifacts_state ON artifacts(processing_state)")
        conn.commit()


def _heartbeat() -> str:
    now = datetime.now(timezone.utc)
    with _db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO service_heartbeat(service_name, updated_at)
                VALUES(%s, %s)
                ON CONFLICT(service_name) DO UPDATE SET updated_at=excluded.updated_at
                """,
                (SERVICE, now),
            )
        conn.commit()
    return now.isoformat()


def _ensure_device(cur: psycopg.Cursor, device_id: str) -> None:
    cur.execute(
        """
        INSERT INTO device(id, type, fw_version, added_at)
        VALUES(%s, 'wearable', NULL, %s)
        ON CONFLICT(id) DO NOTHING
        """,
        (device_id, datetime.now(timezone.utc)),
    )


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
) -> dict:
    _ensure_device(cur, device_id)
    audio_sha = sha256(audio_bytes).hexdigest()
    audio_path, transcript_fixture_path = _audio_artifact_paths(note_id, start_ts)
    audio_path.write_bytes(audio_bytes)

    transcript_hint_path_str = None
    artifact_id = None
    if transcript_hint:
        transcript_fixture_path.write_text(transcript_hint.strip(), encoding="utf-8")
        transcript_hint_path_str = str(transcript_fixture_path)
        artifact_id = str(uuid4())
        cur.execute(
            """
            INSERT INTO artifacts(
                id, artifact_type, source, created_at, content_text, content_hash, object_uri, metadata_json, processing_state
            )
            VALUES(%s, %s, %s, %s, %s, %s, %s, %s::jsonb, 'pending')
            """,
            (
                artifact_id,
                "voice_transcript",
                source,
                ingested_at,
                transcript_hint.strip(),
                sha256(transcript_hint.strip().encode("utf-8")).hexdigest(),
                str(transcript_fixture_path),
                json.dumps({"audio_id": note_id, "device_id": device_id}, sort_keys=True),
            ),
        )

    cur.execute(
        """
        INSERT INTO audio_note(
            id, device_id, start_ts, end_ts, audio_path, audio_sha256, audio_bytes,
            transcript_hint_path, source, transcription_state, ingested_at
        )
        VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, 'pending', %s)
        ON CONFLICT(id) DO UPDATE SET
            device_id=excluded.device_id,
            start_ts=excluded.start_ts,
            end_ts=excluded.end_ts,
            audio_path=excluded.audio_path,
            audio_sha256=excluded.audio_sha256,
            audio_bytes=excluded.audio_bytes,
            transcript_hint_path=excluded.transcript_hint_path,
            source=excluded.source,
            transcription_state='pending',
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
        "artifact_id": artifact_id,
    }


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
    return {"status": "ready", "service": SERVICE, "database": "postgres", "heartbeat_at": ts}


@app.post("/ingest/hr")
def ingest_hr(payload: HrIngestRequest) -> dict:
    ingested_at = datetime.now(timezone.utc)
    with _db_conn() as conn:
        with conn.cursor() as cur:
            _ensure_device(cur, payload.device_id)
            for sample in payload.samples:
                cur.execute(
                    """
                    INSERT INTO hr_sample(device_id, ts, bpm, quality, sensor_status, ingested_at)
                    VALUES(%s, %s, %s, %s, %s, %s)
                    ON CONFLICT(device_id, ts) DO UPDATE SET
                        bpm=excluded.bpm,
                        quality=excluded.quality,
                        sensor_status=excluded.sensor_status,
                        ingested_at=excluded.ingested_at
                    """,
                    (
                        payload.device_id,
                        sample.ts,
                        sample.bpm,
                        sample.quality,
                        sample.sensor_status,
                        ingested_at,
                    ),
                )
        conn.commit()
    return {
        "status": "ok",
        "device_id": payload.device_id,
        "accepted_samples": len(payload.samples),
        "source": payload.source,
        "ingested_at": ingested_at.isoformat(),
    }


@app.post("/ingest/audio")
def ingest_audio(payload: AudioIngestRequest) -> dict:
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
        "artifact_id": persisted["artifact_id"],
        "ingested_at": ingested_at.isoformat(),
    }


@app.get("/summary/hr/today")
def summary_hr_today(
    date: str | None = Query(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    device_id: str | None = Query(default=None, min_length=1, max_length=64),
) -> dict:
    day = date or datetime.now(timezone.utc).date().isoformat()
    start = datetime.fromisoformat(f"{day}T00:00:00+00:00")
    end = datetime.fromisoformat(f"{day}T23:59:59.999999+00:00")
    with _db_conn() as conn:
        with conn.cursor() as cur:
            if device_id:
                cur.execute(
                    """
                    SELECT COUNT(*), MIN(bpm), MAX(bpm), AVG(bpm)
                    FROM hr_sample
                    WHERE ts >= %s AND ts <= %s AND device_id = %s
                    """,
                    (start, end, device_id),
                )
            else:
                cur.execute(
                    """
                    SELECT COUNT(*), MIN(bpm), MAX(bpm), AVG(bpm)
                    FROM hr_sample
                    WHERE ts >= %s AND ts <= %s
                    """,
                    (start, end),
                )
            row = cur.fetchone()
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


@app.get("/summary/audio/recent")
def summary_audio_recent(limit: int = Query(default=10, ge=1, le=100)) -> dict:
    with _db_conn() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT
                    a.id,
                    a.device_id,
                    a.start_ts,
                    a.end_ts,
                    a.transcription_state,
                    a.audio_bytes,
                    CASE WHEN t.audio_id IS NULL THEN 0 ELSE 1 END AS transcript_ready
                FROM audio_note a
                LEFT JOIN transcript t ON t.audio_id = a.id
                ORDER BY a.start_ts DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()
    return {"status": "ok", "entries": rows}


@app.get("/status")
def status() -> dict:
    with _db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT updated_at FROM service_heartbeat WHERE service_name = %s", (SERVICE,))
            hb = cur.fetchone()
            cur.execute("SELECT COUNT(*) FROM hr_sample")
            hr_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM audio_note")
            audio_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM artifacts")
            artifact_count = cur.fetchone()[0]
    return {
        "service": SERVICE,
        "database": "postgres",
        "last_heartbeat": hb[0].isoformat() if hb else None,
        "hr_sample_count_total": int(hr_count),
        "audio_note_count_total": int(audio_count),
        "artifact_count_total": int(artifact_count),
    }
