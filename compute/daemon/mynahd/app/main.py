import os
import sqlite3
from base64 import b64decode
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

SERVICE = os.getenv("MYNAH_SERVICE_NAME", "mynahd")
DB_PATH = Path(os.getenv("MYNAH_DB_PATH", "/data/db/mynah.db"))
ARTIFACTS_PATH = Path(os.getenv("MYNAH_ARTIFACTS_PATH", "/home/appuser/data/artifacts"))

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


class AudioIngestRequest(BaseModel):
    note_id: str | None = Field(default=None, min_length=4, max_length=80, pattern=r"^[A-Za-z0-9._:-]+$")
    device_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9._:-]+$")
    start_ts: datetime
    end_ts: datetime
    audio_b64: str = Field(min_length=8, max_length=1_500_000)
    transcript_hint: str | None = Field(default=None, max_length=20_000)
    source: str = Field(default="simulated", max_length=32)

    @field_validator("start_ts", "end_ts")
    @classmethod
    def normalize_ts(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("timestamp must include timezone information")
        return value.astimezone(timezone.utc)


class AudioChunkIngestRequest(BaseModel):
    object_id: str = Field(min_length=4, max_length=80, pattern=r"^[A-Za-z0-9._:-]+$")
    device_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9._:-]+$")
    session_id: str = Field(default="default", min_length=1, max_length=80, pattern=r"^[A-Za-z0-9._:-]+$")
    start_ts: datetime
    end_ts: datetime
    chunk_index: int = Field(ge=0, le=4096)
    total_chunks: int = Field(ge=1, le=4096)
    chunk_b64: str = Field(min_length=4, max_length=1_500_000)
    transcript_hint: str | None = Field(default=None, max_length=20_000)
    source: str = Field(default="simulated", max_length=32)

    @field_validator("start_ts", "end_ts")
    @classmethod
    def normalize_ts(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("timestamp must include timezone information")
        return value.astimezone(timezone.utc)


def _init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)
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
                ingested_at TEXT NOT NULL,
                FOREIGN KEY(device_id) REFERENCES device(id)
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_audio_note_start_ts ON audio_note(start_ts)
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
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sync_object (
                object_id TEXT PRIMARY KEY,
                device_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                start_ts TEXT NOT NULL,
                end_ts TEXT NOT NULL,
                total_chunks INTEGER NOT NULL,
                received_chunks INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL,
                transcript_hint TEXT,
                source TEXT NOT NULL,
                audio_id TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sync_chunk (
                object_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                chunk_data BLOB NOT NULL,
                PRIMARY KEY(object_id, chunk_index),
                FOREIGN KEY(object_id) REFERENCES sync_object(object_id)
            )
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


def _audio_artifact_paths(note_id: str, start_ts: datetime) -> tuple[Path, Path]:
    audio_dir = ARTIFACTS_PATH / "audio" / start_ts.strftime("%Y") / start_ts.strftime("%m")
    transcript_fixture_dir = ARTIFACTS_PATH / "transcript_fixtures"
    audio_dir.mkdir(parents=True, exist_ok=True)
    transcript_fixture_dir.mkdir(parents=True, exist_ok=True)
    return audio_dir / f"{note_id}.wav", transcript_fixture_dir / f"{note_id}.txt"


def _persist_audio_note(
    conn: sqlite3.Connection,
    *,
    note_id: str,
    device_id: str,
    start_ts: datetime,
    end_ts: datetime,
    audio_bytes: bytes,
    transcript_hint: str | None,
    source: str,
    ingested_at: str,
) -> dict:
    _ensure_device(conn, device_id)
    audio_sha = sha256(audio_bytes).hexdigest()
    audio_path, transcript_fixture_path = _audio_artifact_paths(note_id, start_ts)
    audio_path.write_bytes(audio_bytes)

    transcript_hint_path_str = None
    if transcript_hint:
        transcript_fixture_path.write_text(transcript_hint.strip(), encoding="utf-8")
        transcript_hint_path_str = str(transcript_fixture_path)

    conn.execute(
        """
        INSERT INTO audio_note(
            id, device_id, start_ts, end_ts, audio_path, audio_sha256, audio_bytes,
            transcript_hint_path, source, transcription_state, ingested_at
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
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
            start_ts.isoformat(),
            end_ts.isoformat(),
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
    ingested_at = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        persisted = _persist_audio_note(
            conn,
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
        "ingested_at": ingested_at,
    }


@app.post("/ingest/audio_chunk")
def ingest_audio_chunk(payload: AudioChunkIngestRequest) -> dict:
    if payload.end_ts <= payload.start_ts:
        raise HTTPException(status_code=422, detail="end_ts must be later than start_ts")
    if payload.chunk_index >= payload.total_chunks:
        raise HTTPException(status_code=422, detail="chunk_index must be less than total_chunks")
    try:
        chunk_bytes = b64decode(payload.chunk_b64, validate=True)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail="chunk_b64 must be valid base64") from exc
    if not chunk_bytes:
        raise HTTPException(status_code=422, detail="decoded chunk payload is empty")

    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        existing = conn.execute(
            "SELECT * FROM sync_object WHERE object_id = ?",
            (payload.object_id,),
        ).fetchone()
        if existing:
            if (
                existing["device_id"] != payload.device_id
                or existing["total_chunks"] != payload.total_chunks
                or existing["start_ts"] != payload.start_ts.isoformat()
                or existing["end_ts"] != payload.end_ts.isoformat()
            ):
                raise HTTPException(status_code=409, detail="sync object metadata conflict")
            conn.execute(
                """
                UPDATE sync_object
                SET updated_at = ?, transcript_hint = COALESCE(?, transcript_hint)
                WHERE object_id = ?
                """,
                (now, payload.transcript_hint, payload.object_id),
            )
        else:
            _ensure_device(conn, payload.device_id)
            conn.execute(
                """
                INSERT INTO sync_object(
                    object_id, device_id, session_id, start_ts, end_ts, total_chunks,
                    received_chunks, status, transcript_hint, source, audio_id, updated_at
                )
                VALUES(?, ?, ?, ?, ?, ?, 0, 'pending', ?, ?, NULL, ?)
                """,
                (
                    payload.object_id,
                    payload.device_id,
                    payload.session_id,
                    payload.start_ts.isoformat(),
                    payload.end_ts.isoformat(),
                    payload.total_chunks,
                    payload.transcript_hint,
                    payload.source,
                    now,
                ),
            )

        conn.execute(
            """
            INSERT INTO sync_chunk(object_id, chunk_index, chunk_data)
            VALUES(?, ?, ?)
            ON CONFLICT(object_id, chunk_index) DO NOTHING
            """,
            (payload.object_id, payload.chunk_index, chunk_bytes),
        )

        received = conn.execute(
            "SELECT COUNT(*) FROM sync_chunk WHERE object_id = ?",
            (payload.object_id,),
        ).fetchone()
        received_chunks = int(received[0] if received else 0)
        conn.execute(
            """
            UPDATE sync_object
            SET received_chunks = ?, updated_at = ?
            WHERE object_id = ?
            """,
            (received_chunks, now, payload.object_id),
        )

        complete = received_chunks == payload.total_chunks
        persisted_audio = None
        if complete:
            state = conn.execute(
                "SELECT status, audio_id, transcript_hint FROM sync_object WHERE object_id = ?",
                (payload.object_id,),
            ).fetchone()
            if state and state["status"] != "complete":
                rows = conn.execute(
                    """
                    SELECT chunk_data
                    FROM sync_chunk
                    WHERE object_id = ?
                    ORDER BY chunk_index ASC
                    """,
                    (payload.object_id,),
                ).fetchall()
                audio_bytes = b"".join(bytes(row[0]) for row in rows)
                persisted_audio = _persist_audio_note(
                    conn,
                    note_id=payload.object_id,
                    device_id=payload.device_id,
                    start_ts=payload.start_ts,
                    end_ts=payload.end_ts,
                    audio_bytes=audio_bytes,
                    transcript_hint=state["transcript_hint"],
                    source=payload.source,
                    ingested_at=now,
                )
                conn.execute(
                    """
                    UPDATE sync_object
                    SET status = 'complete', audio_id = ?, updated_at = ?
                    WHERE object_id = ?
                    """,
                    (payload.object_id, now, payload.object_id),
                )
        conn.commit()

    return {
        "status": "ok",
        "object_id": payload.object_id,
        "received_chunks": received_chunks,
        "total_chunks": payload.total_chunks,
        "complete": complete,
        "audio_id": payload.object_id if complete else None,
        "audio_persisted": bool(persisted_audio),
    }


@app.get("/ingest/audio_chunk/status")
def ingest_audio_chunk_status(object_id: str = Query(min_length=4, max_length=80)) -> dict:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT object_id, device_id, session_id, total_chunks, received_chunks, status, audio_id, updated_at
            FROM sync_object
            WHERE object_id = ?
            """,
            (object_id,),
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="object_id not found")
    return {"status": "ok", "sync_object": dict(row)}


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


@app.get("/summary/audio/recent")
def summary_audio_recent(limit: int = Query(default=10, ge=1, le=100)) -> dict:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
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
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return {"status": "ok", "entries": [dict(row) for row in rows]}


@app.get("/status")
def status() -> dict:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT updated_at FROM service_heartbeat WHERE service_name = ?",
            (SERVICE,),
        ).fetchone()
        hr_count = conn.execute("SELECT COUNT(*) FROM hr_sample").fetchone()
        audio_count = conn.execute("SELECT COUNT(*) FROM audio_note").fetchone()
    return {
        "service": SERVICE,
        "last_heartbeat": row[0] if row else None,
        "hr_sample_count_total": int(hr_count[0] if hr_count else 0),
        "audio_note_count_total": int(audio_count[0] if audio_count else 0),
    }
