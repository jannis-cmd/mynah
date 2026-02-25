import json
import os
import re
import urllib.error
import urllib.request
from datetime import date, datetime, time, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

import psycopg
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, ValidationError, field_validator
from psycopg.rows import dict_row

SERVICE = os.getenv("MYNAH_SERVICE_NAME", "mynah_agent")
DATABASE_DSN = os.getenv("MYNAH_DATABASE_DSN", "postgresql://mynah:mynah@postgres:5432/mynah")
ARTIFACTS_PATH = Path(os.getenv("MYNAH_ARTIFACTS_PATH", "/home/appuser/data/artifacts"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:35b-a3b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding:0.6b")
OLLAMA_TIMEOUT_SEC = int(os.getenv("OLLAMA_TIMEOUT_SEC", "120"))
OLLAMA_EMBED_DIM = int(os.getenv("OLLAMA_EMBED_DIM", "1024"))
MAX_COMPACTION_RETRIES = int(os.getenv("MYNAH_COMPACTION_MAX_RETRIES", "3"))
EXACT_WINDOW_MIN = int(os.getenv("MYNAH_LINK_WINDOW_MIN", "90"))

ISO_DATETIME_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:\d{2})?\b")
ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
HOUR_MIN_RE = re.compile(r"\b(\d{1,2}):(\d{2})\b")

app = FastAPI(title="mynah_agent", version="0.4.0")


class ArtifactIngestRequest(BaseModel):
    source_type: str = Field(min_length=1, max_length=64)
    content: str = Field(min_length=1, max_length=200_000)
    upload_ts: datetime
    source_ts: datetime | None
    day_scope: bool
    timezone: str = Field(min_length=1, max_length=64)
    caller: str = Field(default="local_ui", min_length=1, max_length=64)

    @field_validator("upload_ts", "source_ts")
    @classmethod
    def normalize_ts(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("timestamps must include timezone information")
        return value.astimezone(timezone.utc)

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"invalid timezone: {value}") from exc
        return value


class ArtifactProcessRequest(BaseModel):
    caller: str = Field(default="local_ui", min_length=1, max_length=64)


class MeMarkdownProcessRequest(BaseModel):
    source_type: str = Field(default="manual_text", min_length=1, max_length=64)
    markdown: str = Field(min_length=1, max_length=200_000)
    upload_ts: datetime
    source_ts: datetime | None
    day_scope: bool
    timezone: str = Field(min_length=1, max_length=64)
    caller: str = Field(default="local_ui", min_length=1, max_length=64)

    @field_validator("upload_ts", "source_ts")
    @classmethod
    def normalize_ts(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("timestamps must include timezone information")
        return value.astimezone(timezone.utc)

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"invalid timezone: {value}") from exc
        return value


class AudioTranscribeRequest(BaseModel):
    audio_id: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9._:-]+$")
    caller: str = Field(default="local_ui", min_length=1, max_length=64)
    force: bool = Field(default=False)


class ReportGenerateRequest(BaseModel):
    date: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    caller: str = Field(default="local_ui", min_length=1, max_length=64)


class CompactedNote(BaseModel):
    text: str = Field(min_length=1, max_length=600)
    ts_hint: str | None = Field(default=None, max_length=120)


class CompactionOutput(BaseModel):
    notes: list[CompactedNote] = Field(min_length=1, max_length=256)


def _db_conn() -> psycopg.Connection:
    return psycopg.connect(DATABASE_DSN, autocommit=False)


def _init_db() -> None:
    ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)
    if OLLAMA_EMBED_DIM <= 0:
        raise RuntimeError("OLLAMA_EMBED_DIM must be positive")

    with _db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute("CREATE SCHEMA IF NOT EXISTS core")
            cur.execute("CREATE SCHEMA IF NOT EXISTS health")
            cur.execute("CREATE SCHEMA IF NOT EXISTS memory")

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS core.ingest_artifact (
                    id TEXT PRIMARY KEY,
                    source_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    upload_ts TIMESTAMPTZ NOT NULL,
                    source_ts TIMESTAMPTZ,
                    day_scope BOOLEAN NOT NULL,
                    timezone TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    processing_state TEXT NOT NULL DEFAULT 'pending',
                    created_at TIMESTAMPTZ NOT NULL,
                    processed_at TIMESTAMPTZ
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_ingest_artifact_state ON core.ingest_artifact(processing_state)")

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS core.compaction_attempt (
                    id BIGSERIAL PRIMARY KEY,
                    artifact_id TEXT NOT NULL,
                    attempt INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    caller TEXT NOT NULL,
                    model TEXT NOT NULL,
                    prompt_text TEXT NOT NULL,
                    output_text TEXT,
                    error_text TEXT,
                    created_at TIMESTAMPTZ NOT NULL
                )
                """
            )

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

            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS memory.note (
                    id BIGSERIAL PRIMARY KEY,
                    ts TIMESTAMPTZ NOT NULL,
                    ts_mode TEXT NOT NULL CHECK (ts_mode IN ('exact','day','inferred','upload')),
                    text TEXT NOT NULL,
                    embedding VECTOR({OLLAMA_EMBED_DIM}) NOT NULL,
                    source_artifact_id TEXT NOT NULL REFERENCES core.ingest_artifact(id) ON DELETE CASCADE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_memory_note_ts ON memory.note(ts)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_memory_note_source ON memory.note(source_artifact_id)")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS memory.health_link (
                    id BIGSERIAL PRIMARY KEY,
                    memory_id BIGINT NOT NULL REFERENCES memory.note(id) ON DELETE CASCADE,
                    health_sample_id BIGINT REFERENCES health.sample(id) ON DELETE CASCADE,
                    link_day DATE,
                    relation TEXT NOT NULL CHECK (relation IN ('mentions','during','correlates_with')),
                    confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
                    created_at TIMESTAMPTZ NOT NULL
                )
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS core.report_artifact (
                    report_date TEXT PRIMARY KEY,
                    path TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL,
                    generator TEXT NOT NULL,
                    note TEXT
                )
                """
            )
        conn.commit()

def _model_available(model_name: str) -> bool:
    req = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/tags", method="GET")
    with urllib.request.urlopen(req, timeout=5) as resp:
        tags = json.loads(resp.read().decode("utf-8"))
    models = [m.get("name") for m in tags.get("models", [])]
    return model_name in models


def _ollama_generate(prompt: str) -> str:
    payload = json.dumps({"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT_SEC) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("response", "").strip()


def _ollama_embed(text: str) -> list[float]:
    payloads = [
        {"model": OLLAMA_EMBED_MODEL, "input": text, "dimensions": OLLAMA_EMBED_DIM},
        {"model": OLLAMA_EMBED_MODEL, "prompt": text, "dimensions": OLLAMA_EMBED_DIM},
        {"model": OLLAMA_EMBED_MODEL, "input": text},
        {"model": OLLAMA_EMBED_MODEL, "prompt": text},
    ]
    for endpoint in ("/api/embed", "/api/embeddings"):
        for payload in payloads:
            try:
                req = urllib.request.Request(
                    f"{OLLAMA_BASE_URL}{endpoint}",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT_SEC) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                embedding = None
                if isinstance(data.get("embedding"), list):
                    embedding = [float(v) for v in data["embedding"]]
                elif isinstance(data.get("embeddings"), list) and data["embeddings"] and isinstance(data["embeddings"][0], list):
                    embedding = [float(v) for v in data["embeddings"][0]]
                if embedding is not None and len(embedding) == OLLAMA_EMBED_DIM:
                    return embedding
            except Exception:  # noqa: BLE001
                continue
    raise RuntimeError(f"failed to produce embedding length {OLLAMA_EMBED_DIM} using {OLLAMA_EMBED_MODEL}")


def _vector_literal(embedding: list[float]) -> str:
    return "[" + ",".join(f"{value:.10f}" for value in embedding) + "]"


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if not match:
        raise ValueError("no JSON object in model output")
    payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("JSON root must be object")
    return payload


def _insert_artifact(req: ArtifactIngestRequest) -> str:
    artifact_id = str(uuid4())
    now = datetime.now(timezone.utc)
    with _db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO core.ingest_artifact(
                    id, source_type, content, upload_ts, source_ts, day_scope, timezone,
                    content_hash, processing_state, created_at
                )
                VALUES(%s, %s, %s, %s, %s, %s, %s, %s, 'pending', %s)
                """,
                (
                    artifact_id,
                    req.source_type,
                    req.content,
                    req.upload_ts,
                    req.source_ts,
                    req.day_scope,
                    req.timezone,
                    sha256(req.content.encode("utf-8")).hexdigest(),
                    now,
                ),
            )
        conn.commit()
    return artifact_id


def _fetch_artifact(artifact_id: str) -> dict[str, Any] | None:
    with _db_conn() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT id, source_type, content, upload_ts, source_ts, day_scope, timezone, processing_state
                FROM core.ingest_artifact
                WHERE id = %s
                LIMIT 1
                """,
                (artifact_id,),
            )
            return cur.fetchone()


def _set_artifact_state(artifact_id: str, state: str) -> None:
    with _db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE core.ingest_artifact SET processing_state = %s, processed_at = %s WHERE id = %s",
                (state, datetime.now(timezone.utc), artifact_id),
            )
        conn.commit()


def _audit_compaction(
    *,
    artifact_id: str,
    attempt: int,
    status: str,
    caller: str,
    prompt_text: str,
    output_text: str | None,
    error_text: str | None,
) -> None:
    with _db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO core.compaction_attempt(
                    artifact_id, attempt, status, caller, model, prompt_text, output_text, error_text, created_at
                )
                VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    artifact_id,
                    attempt,
                    status,
                    caller,
                    OLLAMA_MODEL,
                    prompt_text,
                    output_text,
                    error_text,
                    datetime.now(timezone.utc),
                ),
            )
        conn.commit()

def _parse_iso_token(token: str, tz: ZoneInfo) -> datetime | None:
    value = token.strip()
    try:
        if "T" in value or " " in value:
            value = value.replace(" ", "T")
            if value.endswith("Z"):
                value = value[:-1] + "+00:00"
            dt = datetime.fromisoformat(value)
            if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
                dt = dt.replace(tzinfo=tz)
            return dt.astimezone(timezone.utc)
        parsed_date = date.fromisoformat(value)
        local = datetime.combine(parsed_date, time(hour=12, minute=0), tzinfo=tz)
        return local.astimezone(timezone.utc)
    except Exception:  # noqa: BLE001
        return None


def _extract_explicit_candidates(content: str, tz: ZoneInfo) -> list[datetime]:
    seen: set[str] = set()
    candidates: list[datetime] = []
    datetime_tokens = ISO_DATETIME_RE.findall(content)
    for token in datetime_tokens:
        parsed = _parse_iso_token(token, tz)
        if not parsed:
            continue
        key = parsed.isoformat()
        if key in seen:
            continue
        seen.add(key)
        candidates.append(parsed)
    for token in ISO_DATE_RE.findall(content):
        if any(token in dt for dt in datetime_tokens):
            continue
        parsed = _parse_iso_token(token, tz)
        if not parsed:
            continue
        key = parsed.isoformat()
        if key in seen:
            continue
        seen.add(key)
        candidates.append(parsed)
    return sorted(candidates)[:32]


def _resolve_relative_hint(hint: str, upload_ts: datetime, tz: ZoneInfo) -> datetime | None:
    lower = hint.lower().strip()
    local_upload = upload_ts.astimezone(tz)
    target_day = local_upload.date()
    if "yesterday" in lower:
        target_day -= timedelta(days=1)
    elif "tomorrow" in lower:
        target_day += timedelta(days=1)

    hour = None
    minute = 0
    hm = HOUR_MIN_RE.search(lower)
    if hm:
        hour = int(hm.group(1))
        minute = int(hm.group(2))
    elif "morning" in lower:
        hour = 9
    elif "afternoon" in lower:
        hour = 15
    elif "evening" in lower:
        hour = 19
    elif "night" in lower:
        hour = 22
    elif "noon" in lower or "midday" in lower:
        hour = 12

    if hour is None or hour < 0 or hour > 23:
        return None
    local = datetime(target_day.year, target_day.month, target_day.day, hour, minute, tzinfo=tz)
    return local.astimezone(timezone.utc)


def _day_anchor(upload_ts: datetime, tz: ZoneInfo) -> datetime:
    local_day = upload_ts.astimezone(tz).date()
    local_anchor = datetime(local_day.year, local_day.month, local_day.day, 12, 0, tzinfo=tz)
    return local_anchor.astimezone(timezone.utc)


def _resolve_note_ts(
    *,
    ts_hint: str | None,
    source_ts: datetime | None,
    day_scope: bool,
    upload_ts: datetime,
    explicit_candidates: list[datetime],
    tz: ZoneInfo,
) -> tuple[datetime, str]:
    if source_ts is not None:
        return source_ts, "exact"
    if day_scope:
        return _day_anchor(upload_ts, tz), "day"
    if explicit_candidates:
        return explicit_candidates[0], "exact"
    if ts_hint:
        explicit = _parse_iso_token(ts_hint, tz)
        if explicit is not None:
            return explicit, "exact"
        inferred = _resolve_relative_hint(ts_hint, upload_ts, tz)
        if inferred is not None:
            return inferred, "inferred"
    return upload_ts, "upload"


def _build_compaction_prompt(
    artifact: dict[str, Any],
    explicit_candidates: list[datetime],
    previous_error: str | None,
) -> str:
    explicit = [item.isoformat() for item in explicit_candidates[:10]]
    prompt = (
        "You are MYNAH memory compactor. Return ONLY one JSON object with key 'notes'.\n"
        "Format: {\"notes\":[{\"text\":\"...\",\"ts_hint\":null}]}\n"
        "Each note must be atomic, concise, and faithful to content.\n"
        "Do not invent facts and do not merge unrelated topics.\n"
        "ts_hint may be null, an explicit timestamp, or a relative phrase.\n\n"
        f"source_type: {artifact['source_type']}\n"
        f"day_scope: {artifact['day_scope']}\n"
        f"timezone: {artifact['timezone']}\n"
        f"source_ts: {artifact['source_ts']}\n"
        f"upload_ts: {artifact['upload_ts']}\n"
        f"explicit_timestamp_candidates: {json.dumps(explicit)}\n\n"
        f"content:\n{artifact['content'][:12000]}\n"
    )
    if previous_error:
        prompt += f"\nFix previous error exactly: {previous_error}\n"
    return prompt


def _link_note_to_health(cur: psycopg.Cursor, note_id: int, note_ts: datetime, ts_mode: str) -> int:
    now = datetime.now(timezone.utc)
    inserted = 0
    if ts_mode == "day":
        day_start = note_ts.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)
        cur.execute(
            """
            SELECT id FROM health.sample
            WHERE ts >= %s AND ts < %s
            ORDER BY ts ASC
            LIMIT 1000
            """,
            (day_start, day_end),
        )
        for row in cur.fetchall():
            cur.execute(
                """
                INSERT INTO memory.health_link(memory_id, health_sample_id, link_day, relation, confidence, created_at)
                VALUES(%s, %s, %s, 'during', %s, %s)
                """,
                (note_id, row[0], day_start.date(), 0.75, now),
            )
            inserted += 1
        return inserted

    window = timedelta(minutes=EXACT_WINDOW_MIN)
    start = note_ts - window
    end = note_ts + window
    confidence = {"exact": 0.9, "inferred": 0.6, "upload": 0.3}.get(ts_mode, 0.5)
    cur.execute(
        """
        SELECT id FROM health.sample
        WHERE ts >= %s AND ts <= %s
        ORDER BY ts ASC
        LIMIT 1000
        """,
        (start, end),
    )
    for row in cur.fetchall():
        cur.execute(
            """
            INSERT INTO memory.health_link(memory_id, health_sample_id, link_day, relation, confidence, created_at)
            VALUES(%s, %s, NULL, 'during', %s, %s)
            """,
            (note_id, row[0], confidence, now),
        )
        inserted += 1
    return inserted


def _compact_with_retries(artifact: dict[str, Any], caller: str) -> CompactionOutput:
    explicit_candidates = _extract_explicit_candidates(artifact["content"], ZoneInfo(artifact["timezone"]))
    previous_error = None
    for attempt in range(1, MAX_COMPACTION_RETRIES + 1):
        prompt = _build_compaction_prompt(artifact, explicit_candidates, previous_error)
        output_text = None
        try:
            output_text = _ollama_generate(prompt)
            parsed = _extract_json_object(output_text)
            compacted = CompactionOutput.model_validate(parsed)
            _audit_compaction(
                artifact_id=artifact["id"],
                attempt=attempt,
                status="accepted",
                caller=caller,
                prompt_text=prompt,
                output_text=output_text,
                error_text=None,
            )
            return compacted
        except (ValidationError, ValueError, urllib.error.URLError, urllib.error.HTTPError, RuntimeError) as exc:
            previous_error = str(exc)
            _audit_compaction(
                artifact_id=artifact["id"],
                attempt=attempt,
                status="rejected",
                caller=caller,
                prompt_text=prompt,
                output_text=output_text,
                error_text=previous_error,
            )
        except Exception as exc:  # noqa: BLE001
            previous_error = str(exc)
            _audit_compaction(
                artifact_id=artifact["id"],
                attempt=attempt,
                status="rejected",
                caller=caller,
                prompt_text=prompt,
                output_text=output_text,
                error_text=previous_error,
            )
    raise RuntimeError(f"compaction failed after {MAX_COMPACTION_RETRIES} retries")


def _process_artifact(artifact_id: str, caller: str) -> dict[str, Any]:
    artifact = _fetch_artifact(artifact_id)
    if not artifact:
        raise HTTPException(status_code=404, detail="artifact not found")
    if artifact["processing_state"] == "processed":
        return {"status": "ok", "artifact_id": artifact_id, "notes_created": 0, "links_created": 0, "cached": True}

    try:
        compacted = _compact_with_retries(artifact, caller)
    except RuntimeError as exc:
        _set_artifact_state(artifact_id, "failed")
        return {"status": "failed_closed", "artifact_id": artifact_id, "reason": str(exc)}

    tz = ZoneInfo(artifact["timezone"])
    explicit_candidates = _extract_explicit_candidates(artifact["content"], tz)
    notes_created = 0
    links_created = 0
    note_ids: list[int] = []

    with _db_conn() as conn:
        with conn.cursor() as cur:
            for item in compacted.notes:
                resolved_ts, ts_mode = _resolve_note_ts(
                    ts_hint=item.ts_hint,
                    source_ts=artifact["source_ts"],
                    day_scope=artifact["day_scope"],
                    upload_ts=artifact["upload_ts"],
                    explicit_candidates=explicit_candidates,
                    tz=tz,
                )
                embedding = _ollama_embed(item.text)
                cur.execute(
                    """
                    INSERT INTO memory.note(ts, ts_mode, text, embedding, source_artifact_id, created_at)
                    VALUES(%s, %s, %s, %s::vector, %s, %s)
                    RETURNING id
                    """,
                    (
                        resolved_ts,
                        ts_mode,
                        item.text.strip(),
                        _vector_literal(embedding),
                        artifact_id,
                        datetime.now(timezone.utc),
                    ),
                )
                note_id = int(cur.fetchone()[0])
                note_ids.append(note_id)
                notes_created += 1
                links_created += _link_note_to_health(cur, note_id, resolved_ts, ts_mode)

            cur.execute(
                "UPDATE core.ingest_artifact SET processing_state = 'processed', processed_at = %s WHERE id = %s",
                (datetime.now(timezone.utc), artifact_id),
            )
        conn.commit()

    return {
        "status": "ok",
        "artifact_id": artifact_id,
        "notes_created": notes_created,
        "links_created": links_created,
        "note_ids": note_ids,
    }

def _transcribe_audio_fixture(req: AudioTranscribeRequest) -> dict[str, Any]:
    with _db_conn() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT id, start_ts, transcript_hint_path
                FROM core.audio_note
                WHERE id = %s
                LIMIT 1
                """,
                (req.audio_id,),
            )
            audio = cur.fetchone()
            if not audio:
                raise HTTPException(status_code=404, detail="audio note not found")
            cur.execute("SELECT audio_id, path, text FROM core.transcript WHERE audio_id = %s LIMIT 1", (req.audio_id,))
            existing = cur.fetchone()

    if existing and not req.force:
        ingest = ArtifactIngestRequest(
            source_type="wearable_transcript",
            content=existing["text"],
            upload_ts=datetime.now(timezone.utc),
            source_ts=audio["start_ts"],
            day_scope=False,
            timezone="UTC",
            caller=req.caller,
        )
        artifact_id = _insert_artifact(ingest)
        return {"status": "ok", "audio_id": req.audio_id, "artifact_id": artifact_id, "transcript_created": False}

    hint_path = Path(audio["transcript_hint_path"]) if audio["transcript_hint_path"] else None
    if not hint_path or not hint_path.exists():
        raise HTTPException(status_code=400, detail="transcript_hint missing; ingest audio with transcript_hint first")
    transcript_text = hint_path.read_text(encoding="utf-8").strip()
    if not transcript_text:
        raise HTTPException(status_code=400, detail="transcript_hint file empty")

    transcripts_dir = ARTIFACTS_PATH / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = transcripts_dir / f"{req.audio_id}.txt"
    transcript_path.write_text(transcript_text, encoding="utf-8")
    now = datetime.now(timezone.utc)

    with _db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO core.transcript(audio_id, text, model, path, created_at)
                VALUES(%s, %s, %s, %s, %s)
                ON CONFLICT(audio_id) DO UPDATE SET
                    text=excluded.text,
                    model=excluded.model,
                    path=excluded.path,
                    created_at=excluded.created_at
                """,
                (req.audio_id, transcript_text, "artifact_transcript_v1", str(transcript_path), now),
            )
        conn.commit()

    ingest = ArtifactIngestRequest(
        source_type="wearable_transcript",
        content=transcript_text,
        upload_ts=now,
        source_ts=audio["start_ts"],
        day_scope=False,
        timezone="UTC",
        caller=req.caller,
    )
    artifact_id = _insert_artifact(ingest)
    process_result = _process_artifact(artifact_id, req.caller)
    return {
        "status": "ok",
        "audio_id": req.audio_id,
        "transcript_created": True,
        "artifact_id": artifact_id,
        "process_result": process_result,
    }


def _generate_report(req: ReportGenerateRequest) -> dict[str, Any]:
    report_date = req.date or datetime.now(timezone.utc).date().isoformat()
    start = datetime.fromisoformat(f"{report_date}T00:00:00+00:00")
    end = datetime.fromisoformat(f"{report_date}T23:59:59.999999+00:00")

    with _db_conn() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT COUNT(*) AS count FROM memory.note WHERE ts >= %s AND ts <= %s", (start, end))
            note_count = int(cur.fetchone()["count"])
            cur.execute("SELECT COUNT(*) AS count FROM health.sample WHERE ts >= %s AND ts <= %s", (start, end))
            health_count = int(cur.fetchone()["count"])
            cur.execute(
                """
                SELECT id, ts, ts_mode, SUBSTRING(text FROM 1 FOR 120) AS preview
                FROM memory.note
                WHERE ts >= %s AND ts <= %s
                ORDER BY ts DESC
                LIMIT 10
                """,
                (start, end),
            )
            notes = cur.fetchall()

    lines = [
        f"# MYNAH Daily Report - {report_date}",
        "",
        "## Counts",
        f"- Memory notes: {note_count}",
        f"- Health samples: {health_count}",
        "",
        "## Recent Memory Notes",
    ]
    for row in notes:
        lines.append(f"- [{row['ts_mode']}] {row['ts']}: {row['preview']}")

    report_dir = ARTIFACTS_PATH / "reports" / report_date
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    now = datetime.now(timezone.utc)
    with _db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO core.report_artifact(report_date, path, created_at, generator, note)
                VALUES(%s, %s, %s, %s, %s)
                ON CONFLICT(report_date) DO UPDATE SET
                    path=excluded.path,
                    created_at=excluded.created_at,
                    generator=excluded.generator,
                    note=excluded.note
                """,
                (report_date, str(report_path), now, SERVICE, f"notes={note_count};health={health_count}"),
            )
        conn.commit()

    return {"status": "ok", "report_date": report_date, "path": str(report_path), "notes": note_count, "health": health_count}


@app.on_event("startup")
def startup() -> None:
    _init_db()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": SERVICE}


@app.get("/ready")
def ready() -> dict[str, Any]:
    try:
        with _db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"postgres unavailable: {exc}") from exc

    try:
        gen_ok = _model_available(OLLAMA_MODEL)
        embed_ok = _model_available(OLLAMA_EMBED_MODEL)
    except urllib.error.URLError as exc:
        raise HTTPException(status_code=503, detail=f"ollama unavailable: {exc.reason}") from exc

    if not gen_ok:
        raise HTTPException(status_code=503, detail=f"required model missing: {OLLAMA_MODEL}")
    if not embed_ok:
        raise HTTPException(status_code=503, detail=f"required embed model missing: {OLLAMA_EMBED_MODEL}")

    return {
        "status": "ready",
        "service": SERVICE,
        "database": "postgres",
        "model": OLLAMA_MODEL,
        "embed_model": OLLAMA_EMBED_MODEL,
        "embed_dim": OLLAMA_EMBED_DIM,
    }


@app.post("/pipeline/artifacts/ingest")
def pipeline_artifacts_ingest(req: ArtifactIngestRequest) -> dict[str, str]:
    artifact_id = _insert_artifact(req)
    return {"status": "ok", "artifact_id": artifact_id}


@app.post("/pipeline/artifacts/process/{artifact_id}")
def pipeline_artifacts_process(artifact_id: str, req: ArtifactProcessRequest) -> dict[str, Any]:
    return _process_artifact(artifact_id, req.caller)


@app.post("/pipeline/me_md/process")
def pipeline_me_md_process(req: MeMarkdownProcessRequest) -> dict[str, Any]:
    ingest = ArtifactIngestRequest(
        source_type=req.source_type,
        content=req.markdown,
        upload_ts=req.upload_ts,
        source_ts=req.source_ts,
        day_scope=req.day_scope,
        timezone=req.timezone,
        caller=req.caller,
    )
    artifact_id = _insert_artifact(ingest)
    result = _process_artifact(artifact_id, req.caller)
    return {"status": "ok", "artifact_id": artifact_id, "result": result}


@app.post("/pipeline/audio/transcribe")
def pipeline_audio_transcribe(req: AudioTranscribeRequest) -> dict[str, Any]:
    return _transcribe_audio_fixture(req)


@app.get("/tools/transcript/recent")
def transcript_recent(limit: int = Query(default=20, ge=1, le=100)) -> dict[str, Any]:
    with _db_conn() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT audio_id, model, created_at, path, SUBSTRING(text FROM 1 FOR 160) AS preview
                FROM core.transcript
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()
    return {"status": "ok", "entries": rows}


@app.post("/tools/report_generate")
def tools_report_generate(req: ReportGenerateRequest) -> dict[str, Any]:
    return _generate_report(req)


@app.get("/tools/report_recent")
def tools_report_recent(limit: int = Query(default=20, ge=1, le=100)) -> dict[str, Any]:
    with _db_conn() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT report_date, path, created_at, generator, note
                FROM core.report_artifact
                ORDER BY report_date DESC
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
            cur.execute("SELECT COUNT(*) FROM core.ingest_artifact")
            artifact_count = int(cur.fetchone()[0])
            cur.execute("SELECT COUNT(*) FROM memory.note")
            note_count = int(cur.fetchone()[0])
            cur.execute("SELECT COUNT(*) FROM memory.health_link")
            link_count = int(cur.fetchone()[0])
            cur.execute("SELECT COUNT(*) FROM core.compaction_attempt")
            attempt_count = int(cur.fetchone()[0])
    return {
        "service": SERVICE,
        "database": "postgres",
        "artifact_count": artifact_count,
        "memory_note_count": note_count,
        "memory_health_link_count": link_count,
        "compaction_attempt_count": attempt_count,
        "embed_dim": OLLAMA_EMBED_DIM,
    }
