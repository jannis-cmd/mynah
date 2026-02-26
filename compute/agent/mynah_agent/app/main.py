import json
import os
import re
import urllib.error
import urllib.request
from base64 import b64decode
from datetime import date, datetime, time, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4
from zoneinfo import ZoneInfo

import psycopg
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from psycopg.rows import dict_row

SERVICE = os.getenv("MYNAH_SERVICE_NAME", "mynah_agent")
DATABASE_DSN = os.getenv("MYNAH_DATABASE_DSN", "postgresql://mynah:mynah@postgres:5432/mynah")
ARTIFACTS_PATH = Path(os.getenv("MYNAH_ARTIFACTS_PATH", "/home/appuser/data/artifacts"))
PROMPTS_PATH = Path(__file__).resolve().parent / "prompts"
TEMPORAL_COMPACTION_PROMPT_PATH = PROMPTS_PATH / "temporal_compaction_prompt.md"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:35b-a3b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding:0.6b")
OLLAMA_TIMEOUT_SEC = int(os.getenv("OLLAMA_TIMEOUT_SEC", "120"))
OLLAMA_EMBED_DIM = int(os.getenv("OLLAMA_EMBED_DIM", "1024"))
MAX_COMPACTION_RETRIES = int(os.getenv("MYNAH_COMPACTION_MAX_RETRIES", "3"))
EXACT_WINDOW_MIN = int(os.getenv("MYNAH_LINK_WINDOW_MIN", "90"))
REQUIRED_TABLES = (
    "core.ingest_artifact",
    "core.compaction_attempt",
    "search.embedding_model",
    "search.vector_index",
    "health.sample",
    "core.audio_note",
    "core.transcript",
    "memory.note",
    "memory.health_link",
    "core.report_artifact",
)

ISO_DATETIME_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:\d{2})?\b")
ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
HOUR_MIN_RE = re.compile(r"\b(\d{1,2}):(\d{2})\b")
AT_TIME_RE = re.compile(r"^at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$", re.IGNORECASE)

HINT_DAYPART_HOURS = {
    "morning": 9,
    "afternoon": 15,
    "evening": 20,
    "night": 22,
}

HINT_LITERAL_SET = {
    "default",
    "today",
    "now",
    "morning",
    "afternoon",
    "evening",
    "night",
    "yesterday",
    "yesterday morning",
    "yesterday afternoon",
    "yesterday evening",
    "yesterday night",
    "tomorrow",
    "tomorrow morning",
    "tomorrow afternoon",
    "tomorrow evening",
    "tomorrow night",
    "last night",
    "tonight",
    "this morning",
    "this afternoon",
    "this evening",
    "at time",
    "yesterday at time",
    "tomorrow at time",
}
NOTE_TYPE_SET = {"event", "fact", "observation", "feeling", "decision_context", "task"}
QUERY_MODE_SET = {"lexical", "semantic", "hybrid", "deep"}
FUSION_RRF_K = 60.0
MAX_CANDIDATE_LIMIT = 200
DEFAULT_CANDIDATE_MULTIPLIER = 4
MAX_QUERY_EXPANSIONS = 4
QUERY_EXPANSION_MAX_RETRIES = 3
DEFAULT_CONTEXT_PROFILE = "balanced"

RECENCY_HINT_TOKENS = {
    "today",
    "yesterday",
    "recent",
    "recently",
    "this week",
    "last week",
    "this month",
    "tonight",
    "this morning",
    "this evening",
}

app = FastAPI(title="mynah_agent", version="0.5.0")


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


class AudioTranscribeRequest(BaseModel):
    audio_id: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9._:-]+$")
    caller: str = Field(default="local_ui", min_length=1, max_length=64)
    force: bool = Field(default=False)


class ReportGenerateRequest(BaseModel):
    date: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    caller: str = Field(default="local_ui", min_length=1, max_length=64)


class TemporalItem(BaseModel):
    text: str = Field(min_length=1, max_length=600)
    note_type: str = Field(min_length=1, max_length=32)

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("item text empty")
        return text

    @field_validator("note_type")
    @classmethod
    def normalize_note_type(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in NOTE_TYPE_SET:
            raise ValueError(f"unsupported note_type: {value}")
        return normalized


class TemporalGroup(BaseModel):
    hint: str = Field(min_length=1, max_length=64)
    items: list[TemporalItem] = Field(min_length=1, max_length=128)

    @field_validator("hint")
    @classmethod
    def normalize_and_validate_hint(cls, value: str) -> str:
        hint = " ".join(value.lower().strip().split())
        if hint in HINT_LITERAL_SET:
            return hint
        if AT_TIME_RE.match(hint):
            return hint
        if hint.startswith("yesterday ") and AT_TIME_RE.match(hint[len("yesterday "):]):
            return hint
        if hint.startswith("tomorrow ") and AT_TIME_RE.match(hint[len("tomorrow "):]):
            return hint
        raise ValueError(f"unsupported hint: {value}")

    @field_validator("items")
    @classmethod
    def normalize_items(cls, values: list[TemporalItem]) -> list[TemporalItem]:
        out: list[TemporalItem] = []
        for value in values:
            out.append(value)
        if not out:
            raise ValueError("items must contain at least one non-empty entry")
        return out


class CompactionOutput(BaseModel):
    groups: list[TemporalGroup] = Field(min_length=1, max_length=64)


class QueryExpansionOutput(BaseModel):
    variants: list[str] = Field(default_factory=list, max_length=MAX_QUERY_EXPANSIONS)

    @field_validator("variants")
    @classmethod
    def normalize_variants(cls, values: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for value in values:
            normalized = " ".join(value.strip().split())
            if not normalized:
                continue
            key = normalized.casefold()
            if key in seen:
                continue
            seen.add(key)
            out.append(normalized)
            if len(out) >= MAX_QUERY_EXPANSIONS:
                break
        return out


class RetrievalRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    mode: Literal["lexical", "semantic", "hybrid", "deep"] = "hybrid"
    limit: int = Field(default=8, ge=1, le=50)
    include_health: bool = False
    query_expansion: bool = True
    rerank: bool | None = None
    context_profile: str = Field(default=DEFAULT_CONTEXT_PROFILE, min_length=1, max_length=64)

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        query = " ".join(value.strip().split())
        if not query:
            raise ValueError("query cannot be empty")
        return query


def _db_conn() -> psycopg.Connection:
    return psycopg.connect(DATABASE_DSN, autocommit=False)


def _table_exists(cur: psycopg.Cursor, full_name: str) -> bool:
    schema_name, table_name = full_name.split(".", 1)
    cur.execute(
        """
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = %s AND table_name = %s
        LIMIT 1
        """,
        (schema_name, table_name),
    )
    return cur.fetchone() is not None


def _assert_schema_ready(cur: psycopg.Cursor) -> None:
    missing = [name for name in REQUIRED_TABLES if not _table_exists(cur, name)]
    if missing:
        raise RuntimeError(
            "schema missing required tables: "
            + ", ".join(missing)
            + " (run storage/schema.sql migration)"
        )

    cur.execute(
        """
        SELECT a.atttypmod
        FROM pg_attribute a
        JOIN pg_class c ON c.oid = a.attrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'memory' AND c.relname = 'note' AND a.attname = 'embedding' AND NOT a.attisdropped
        LIMIT 1
        """
    )
    row = cur.fetchone()
    if not row:
        raise RuntimeError("memory.note.embedding column missing")
    raw_typmod = int(row[0])
    accepted_dims = {raw_typmod}
    if raw_typmod - 4 > 0:
        accepted_dims.add(raw_typmod - 4)
    if OLLAMA_EMBED_DIM not in accepted_dims:
        raise RuntimeError(
            f"embedding dimension mismatch: db_typmod={raw_typmod}, config={OLLAMA_EMBED_DIM}"
        )


def _init_db() -> None:
    ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)
    if OLLAMA_EMBED_DIM <= 0:
        raise RuntimeError("OLLAMA_EMBED_DIM must be positive")
    if not TEMPORAL_COMPACTION_PROMPT_PATH.exists():
        raise RuntimeError(f"missing prompt template: {TEMPORAL_COMPACTION_PROMPT_PATH}")

    with _db_conn() as conn:
        with conn.cursor() as cur:
            _assert_schema_ready(cur)

def _model_available(model_name: str) -> bool:
    req = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/tags", method="GET")
    with urllib.request.urlopen(req, timeout=5) as resp:
        tags = json.loads(resp.read().decode("utf-8"))
    models = [m.get("name") for m in tags.get("models", [])]
    return model_name in models


def _model_state() -> dict[str, Any]:
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            tags = json.loads(resp.read().decode("utf-8"))
        models = {m.get("name") for m in tags.get("models", [])}
    except urllib.error.URLError:
        return {
            "ollama_reachable": False,
            "generation_model_present": False,
            "embedding_model_present": False,
        }
    return {
        "ollama_reachable": True,
        "generation_model_present": OLLAMA_MODEL in models,
        "embedding_model_present": OLLAMA_EMBED_MODEL in models,
    }


def _extract_primary_model_text(data: dict[str, Any]) -> str:
    response = data.get("response")
    if isinstance(response, str) and response.strip():
        return response.strip()

    thinking = data.get("thinking")
    if isinstance(thinking, str) and thinking.strip():
        return thinking.strip()

    message = data.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()

    return ""


def _ollama_generate(prompt: str, response_format: str | dict[str, Any] | None = None) -> str:
    payload_data: dict[str, Any] = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    if response_format is not None:
        payload_data["format"] = response_format
    payload = json.dumps(payload_data).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT_SEC) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return _extract_primary_model_text(data)


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


def _embedding_model_id() -> str:
    return f"ollama:{OLLAMA_EMBED_MODEL}:{OLLAMA_EMBED_DIM}"


def _ensure_embedding_model_row(cur: psycopg.Cursor) -> str:
    model_id = _embedding_model_id()
    cur.execute(
        """
        INSERT INTO search.embedding_model(id, provider, dimension, status, created_at)
        VALUES(%s, 'ollama', %s, 'active', %s)
        ON CONFLICT(id) DO UPDATE SET
            provider = excluded.provider,
            dimension = excluded.dimension,
            status = 'active'
        """,
        (model_id, OLLAMA_EMBED_DIM, datetime.now(timezone.utc)),
    )
    return model_id


def _upsert_vector_index_row(
    cur: psycopg.Cursor,
    *,
    note_id: int,
    text: str,
    embedding: list[float],
    ts_mode: str,
    note_type: str,
    source_artifact_id: str,
) -> None:
    model_id = _ensure_embedding_model_row(cur)
    cur.execute(
        """
        INSERT INTO search.vector_index(
            target_table, target_id, chunk_idx, text_content, embedding, embedding_model_id,
            is_active, invalidated_at, extra_metadata, created_at
        )
        VALUES(
            'memory.note', %s, 0, %s, %s::vector, %s,
            TRUE, NULL, %s::jsonb, %s
        )
        ON CONFLICT(target_table, target_id, chunk_idx, embedding_model_id) DO UPDATE SET
            text_content = excluded.text_content,
            embedding = excluded.embedding,
            is_active = TRUE,
            invalidated_at = NULL,
            extra_metadata = excluded.extra_metadata
        """,
        (
            str(note_id),
            text,
            _vector_literal(embedding),
            model_id,
            json.dumps(
                {
                    "ts_mode": ts_mode,
                    "note_type": note_type,
                    "source_artifact_id": source_artifact_id,
                }
            ),
            datetime.now(timezone.utc),
        ),
    )


def _upsert_all_memory_note_vectors(cur: psycopg.Cursor) -> dict[str, Any]:
    model_id = _ensure_embedding_model_row(cur)
    cur.execute(
        """
        SELECT COUNT(*)
        FROM search.vector_index
        WHERE target_table = 'memory.note' AND embedding_model_id = %s
        """,
        (model_id,),
    )
    before = int(cur.fetchone()[0])
    now = datetime.now(timezone.utc)
    cur.execute(
        """
        INSERT INTO search.vector_index(
            target_table, target_id, chunk_idx, text_content, embedding, embedding_model_id,
            is_active, invalidated_at, extra_metadata, created_at
        )
        SELECT
            'memory.note',
            n.id::text,
            0,
            n.text,
            n.embedding,
            %s,
            TRUE,
            NULL,
            jsonb_build_object(
                'ts_mode', n.ts_mode,
                'note_type', n.note_type,
                'source_artifact_id', n.source_artifact_id
            ),
            %s
        FROM memory.note n
        ON CONFLICT(target_table, target_id, chunk_idx, embedding_model_id) DO UPDATE SET
            text_content = excluded.text_content,
            embedding = excluded.embedding,
            is_active = TRUE,
            invalidated_at = NULL,
            extra_metadata = excluded.extra_metadata
        """,
        (model_id, now),
    )
    cur.execute(
        """
        SELECT COUNT(*)
        FROM search.vector_index
        WHERE target_table = 'memory.note' AND embedding_model_id = %s
        """,
        (model_id,),
    )
    after = int(cur.fetchone()[0])
    return {"embedding_model_id": model_id, "before": before, "after": after, "delta": after - before}


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


def _normalize_hint(hint: str) -> str:
    return " ".join(hint.lower().strip().split())


def _parse_at_time(hint: str) -> tuple[int, int] | None:
    text = _normalize_hint(hint)
    match = AT_TIME_RE.match(text)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2) or "0")
    ampm = match.group(3)

    if ampm:
        if hour < 1 or hour > 12:
            return None
        if ampm.lower() == "am":
            hour = 0 if hour == 12 else hour
        else:
            hour = 12 if hour == 12 else hour + 12
    else:
        if hour < 0 or hour > 23:
            return None

    if minute < 0 or minute > 59:
        return None
    return hour, minute


def _day_anchor(upload_ts: datetime, tz: ZoneInfo) -> datetime:
    local_day = upload_ts.astimezone(tz).date()
    local_anchor = datetime(local_day.year, local_day.month, local_day.day, 12, 0, tzinfo=tz)
    return local_anchor.astimezone(timezone.utc)


def _with_local_time(base_ts: datetime, tz: ZoneInfo, day_delta: int, hour: int, minute: int) -> datetime:
    local = base_ts.astimezone(tz)
    target_day = local.date() + timedelta(days=day_delta)
    target = datetime(target_day.year, target_day.month, target_day.day, hour, minute, tzinfo=tz)
    return target.astimezone(timezone.utc)


def _resolve_default_anchor_ts(
    *,
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
    return upload_ts, "upload"


def _resolve_group_hint_ts(
    *,
    hint: str,
    anchor_ts: datetime,
    anchor_mode: str,
    tz: ZoneInfo,
) -> tuple[datetime, str]:
    normalized = _normalize_hint(hint)

    explicit = _parse_iso_token(normalized, tz)
    if explicit is not None:
        return explicit, "exact"

    if normalized in {"default", "today", "now"}:
        return anchor_ts, anchor_mode
    if normalized in {"morning", "this morning"}:
        return _with_local_time(anchor_ts, tz, 0, HINT_DAYPART_HOURS["morning"], 0), "inferred"
    if normalized in {"afternoon", "this afternoon"}:
        return _with_local_time(anchor_ts, tz, 0, HINT_DAYPART_HOURS["afternoon"], 0), "inferred"
    if normalized in {"evening", "this evening"}:
        return _with_local_time(anchor_ts, tz, 0, HINT_DAYPART_HOURS["evening"], 0), "inferred"
    if normalized in {"night", "tonight"}:
        return _with_local_time(anchor_ts, tz, 0, HINT_DAYPART_HOURS["night"], 0), "inferred"

    if normalized == "yesterday":
        return _with_local_time(anchor_ts, tz, -1, 12, 0), "inferred"
    if normalized == "tomorrow":
        return _with_local_time(anchor_ts, tz, 1, 12, 0), "inferred"
    if normalized == "yesterday morning":
        return _with_local_time(anchor_ts, tz, -1, HINT_DAYPART_HOURS["morning"], 0), "inferred"
    if normalized == "yesterday afternoon":
        return _with_local_time(anchor_ts, tz, -1, HINT_DAYPART_HOURS["afternoon"], 0), "inferred"
    if normalized == "yesterday evening":
        return _with_local_time(anchor_ts, tz, -1, HINT_DAYPART_HOURS["evening"], 0), "inferred"
    if normalized in {"last night", "yesterday night"}:
        return _with_local_time(anchor_ts, tz, -1, HINT_DAYPART_HOURS["night"], 0), "inferred"
    if normalized == "tomorrow morning":
        return _with_local_time(anchor_ts, tz, 1, HINT_DAYPART_HOURS["morning"], 0), "inferred"
    if normalized == "tomorrow afternoon":
        return _with_local_time(anchor_ts, tz, 1, HINT_DAYPART_HOURS["afternoon"], 0), "inferred"
    if normalized == "tomorrow evening":
        return _with_local_time(anchor_ts, tz, 1, HINT_DAYPART_HOURS["evening"], 0), "inferred"
    if normalized == "tomorrow night":
        return _with_local_time(anchor_ts, tz, 1, HINT_DAYPART_HOURS["night"], 0), "inferred"

    if normalized.startswith("yesterday "):
        parsed = _parse_at_time(normalized[len("yesterday "):])
        if parsed:
            return _with_local_time(anchor_ts, tz, -1, parsed[0], parsed[1]), "inferred"
    if normalized.startswith("tomorrow "):
        parsed = _parse_at_time(normalized[len("tomorrow "):])
        if parsed:
            return _with_local_time(anchor_ts, tz, 1, parsed[0], parsed[1]), "inferred"

    parsed = _parse_at_time(normalized)
    if parsed:
        return _with_local_time(anchor_ts, tz, 0, parsed[0], parsed[1]), "inferred"

    return anchor_ts, anchor_mode


def _build_compaction_prompt(
    artifact: dict[str, Any],
    explicit_candidates: list[datetime],
    previous_error: str | None,
) -> str:
    explicit = [item.isoformat() for item in explicit_candidates[:10]]
    template = TEMPORAL_COMPACTION_PROMPT_PATH.read_text(encoding="utf-8")
    previous_error_block = ""
    if previous_error:
        previous_error_block = f"\nFix previous error exactly: {previous_error}\n"
    return template.format(
        source_type=artifact["source_type"],
        day_scope=artifact["day_scope"],
        timezone=artifact["timezone"],
        source_ts=artifact["source_ts"],
        upload_ts=artifact["upload_ts"],
        explicit_timestamp_candidates=json.dumps(explicit),
        content=artifact["content"][:12000],
        previous_error_block=previous_error_block,
    )


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


def _query_has_recency_hint(query: str) -> bool:
    query_lc = query.casefold()
    return any(token in query_lc for token in RECENCY_HINT_TOKENS)


def _recency_boost(ts: datetime, *, enable: bool) -> float:
    if not enable:
        return 0.0
    age_days = max((datetime.now(timezone.utc) - ts).total_seconds() / 86400.0, 0.0)
    return 0.05 / (1.0 + age_days)


def _build_query_expansion_prompt(query: str, previous_error: str | None = None) -> str:
    previous_error_block = ""
    if previous_error:
        previous_error_block = (
            "Previous attempt failed validation.\n"
            f"Fix exactly this error: {previous_error}\n"
        )
    return (
        "Generate concise query variants for local retrieval.\n"
        "Return ONLY JSON.\n"
        "JSON schema:\n"
        "{\n"
        '  "variants": ["string", "string"]\n'
        "}\n"
        "Rules:\n"
        "- Keep meaning equivalent to original query.\n"
        "- Focus on synonyms and alternate phrasing.\n"
        "- Max 4 variants.\n"
        "- Keep each variant <= 12 words.\n"
        "- Do not include numbering.\n"
        "- If uncertain, return: {\"variants\": []}.\n"
        f"{previous_error_block}"
        f"Original query: {query}\n"
    )


def _expand_query_variants(query: str) -> list[str]:
    previous_error = None
    for _attempt in range(1, QUERY_EXPANSION_MAX_RETRIES + 1):
        prompt = _build_query_expansion_prompt(query, previous_error)
        try:
            output = _ollama_generate(prompt, response_format="json")
            parsed = _extract_json_object(output)
            expansion = QueryExpansionOutput.model_validate(parsed)
            out = [item for item in expansion.variants if item.casefold() != query.casefold()]
            return out[:MAX_QUERY_EXPANSIONS]
        except (ValidationError, ValueError, urllib.error.URLError, urllib.error.HTTPError, RuntimeError) as exc:
            previous_error = str(exc)
        except Exception as exc:  # noqa: BLE001
            previous_error = str(exc)
    raise RuntimeError(f"query expansion failed after {QUERY_EXPANSION_MAX_RETRIES} retries: {previous_error}")


def _retrieve_lexical_rows(cur: psycopg.Cursor, *, query: str, limit: int) -> list[dict[str, Any]]:
    cur.execute(
        """
        SELECT
            n.id AS note_id,
            n.ts,
            n.ts_mode,
            n.note_type,
            n.text,
            n.source_artifact_id,
            ts_rank_cd(to_tsvector('simple', n.text), plainto_tsquery('simple', %s))::double precision AS lexical_score
        FROM memory.note n
        WHERE to_tsvector('simple', n.text) @@ plainto_tsquery('simple', %s)
        ORDER BY lexical_score DESC, n.ts DESC, n.id DESC
        LIMIT %s
        """,
        (query, query, limit),
    )
    rows = [dict(row) for row in cur.fetchall()]
    if rows:
        return rows

    tokens = [token for token in re.findall(r"[a-z0-9]{2,}", query.casefold()) if token]
    if not tokens:
        return []
    cur.execute(
        """
        SELECT
            n.id AS note_id,
            n.ts,
            n.ts_mode,
            n.note_type,
            n.text,
            n.source_artifact_id,
            (
                SELECT COUNT(*)::double precision
                FROM unnest(%s::text[]) AS t(token)
                WHERE position(t.token in lower(n.text)) > 0
            ) AS lexical_hits
        FROM memory.note n
        WHERE EXISTS (
            SELECT 1
            FROM unnest(%s::text[]) AS t(token)
            WHERE position(t.token in lower(n.text)) > 0
        )
        ORDER BY lexical_hits DESC, n.ts DESC, n.id DESC
        LIMIT %s
        """,
        (tokens, tokens, limit),
    )
    broad_rows = []
    token_count = max(len(tokens), 1)
    for row in cur.fetchall():
        item = dict(row)
        lexical_hits = float(item.pop("lexical_hits") or 0.0)
        item["lexical_score"] = lexical_hits / float(token_count)
        broad_rows.append(item)
    return broad_rows


def _retrieve_semantic_rows(
    cur: psycopg.Cursor,
    *,
    query_vector_literal: str,
    embedding_model_id: str,
    limit: int,
) -> list[dict[str, Any]]:
    cur.execute(
        """
        SELECT
            v.id AS vector_row_id,
            v.chunk_idx,
            n.id AS note_id,
            n.ts,
            n.ts_mode,
            n.note_type,
            n.text,
            n.source_artifact_id,
            (1 - (v.embedding <=> %s::vector))::double precision AS semantic_score
        FROM search.vector_index v
        JOIN memory.note n
          ON v.target_table = 'memory.note'
         AND v.target_id = n.id::text
        WHERE v.is_active = TRUE
          AND v.embedding_model_id = %s
        ORDER BY v.embedding <=> %s::vector ASC, n.ts DESC, n.id DESC
        LIMIT %s
        """,
        (query_vector_literal, embedding_model_id, query_vector_literal, limit),
    )
    return list(cur.fetchall())


def _ensure_vector_index_ready(cur: psycopg.Cursor) -> dict[str, Any]:
    model_id = _ensure_embedding_model_row(cur)
    cur.execute("SELECT COUNT(*) AS count FROM memory.note")
    memory_notes = int(cur.fetchone()["count"])
    cur.execute(
        """
        SELECT COUNT(*) AS count
        FROM search.vector_index
        WHERE target_table = 'memory.note' AND embedding_model_id = %s AND is_active = TRUE
        """,
        (model_id,),
    )
    vector_rows = int(cur.fetchone()["count"])
    reindex_stats = None
    if vector_rows < memory_notes:
        reindex_stats = _upsert_all_memory_note_vectors(cur)
        vector_rows = int(reindex_stats["after"])
    return {
        "embedding_model_id": model_id,
        "memory_note_count": memory_notes,
        "vector_row_count": vector_rows,
        "reindexed": reindex_stats is not None,
        "reindex_stats": reindex_stats,
    }


def _new_candidate(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "note_id": int(row["note_id"]),
        "ts": row["ts"],
        "ts_mode": row["ts_mode"],
        "note_type": row["note_type"],
        "text": row["text"],
        "source_artifact_id": row["source_artifact_id"],
        "vector_row_id": row.get("vector_row_id"),
        "chunk_idx": int(row.get("chunk_idx") or 0),
        "lexical_score": 0.0,
        "semantic_score": 0.0,
        "lexical_rank": None,
        "semantic_rank": None,
        "rrf_score": 0.0,
        "time_boost": 0.0,
        "final_score": 0.0,
    }


def _accumulate_ranked_rows(
    candidates: dict[int, dict[str, Any]],
    rows: list[dict[str, Any]],
    *,
    channel: Literal["lexical", "semantic"],
    weight: float,
) -> None:
    for rank, row in enumerate(rows, start=1):
        note_id = int(row["note_id"])
        entry = candidates.get(note_id)
        if entry is None:
            entry = _new_candidate(row)
            candidates[note_id] = entry
        if row.get("vector_row_id") is not None:
            entry["vector_row_id"] = int(row["vector_row_id"])
            entry["chunk_idx"] = int(row.get("chunk_idx") or 0)
        if channel == "lexical":
            lexical_score = float(row.get("lexical_score") or 0.0) * weight
            entry["lexical_score"] = max(entry["lexical_score"], lexical_score)
            if entry["lexical_rank"] is None or rank < entry["lexical_rank"]:
                entry["lexical_rank"] = rank
        else:
            semantic_score = float(row.get("semantic_score") or 0.0) * weight
            entry["semantic_score"] = max(entry["semantic_score"], semantic_score)
            if entry["semantic_rank"] is None or rank < entry["semantic_rank"]:
                entry["semantic_rank"] = rank
        entry["rrf_score"] += weight * (1.0 / (FUSION_RRF_K + float(rank)))


def _candidate_sort_key(entry: dict[str, Any]) -> tuple[float, datetime, int]:
    return (float(entry["final_score"]), entry["ts"], int(entry["note_id"]))


def _attach_health_context(cur: psycopg.Cursor, note_ids: list[int]) -> dict[int, dict[str, Any]]:
    if not note_ids:
        return {}
    cur.execute(
        """
        SELECT
            l.memory_id,
            COUNT(*)::bigint AS sample_count,
            MIN(h.ts) AS min_ts,
            MAX(h.ts) AS max_ts
        FROM memory.health_link l
        LEFT JOIN health.sample h ON h.id = l.health_sample_id
        WHERE l.memory_id = ANY(%s)
        GROUP BY l.memory_id
        """,
        (note_ids,),
    )
    out: dict[int, dict[str, Any]] = {}
    for row in cur.fetchall():
        memory_id = int(row["memory_id"])
        out[memory_id] = {
            "sample_count": int(row["sample_count"] or 0),
            "min_ts": row["min_ts"].isoformat() if row["min_ts"] else None,
            "max_ts": row["max_ts"].isoformat() if row["max_ts"] else None,
        }
    return out


def _run_retrieval(req: RetrievalRequest) -> dict[str, Any]:
    mode = req.mode
    candidate_limit = min(MAX_CANDIDATE_LIMIT, max(req.limit * DEFAULT_CANDIDATE_MULTIPLIER, req.limit))
    recency_hint = _query_has_recency_hint(req.query)
    diagnostics: dict[str, Any] = {
        "mode": mode,
        "query_expansion_used": False,
        "rerank_used": False,
        "candidate_limit": candidate_limit,
        "context_profile": req.context_profile,
    }

    query_variants: list[tuple[str, float]] = [(req.query, 1.0)]
    if mode == "deep" and req.query_expansion:
        try:
            expansions = _expand_query_variants(req.query)
            diagnostics["query_expansion_used"] = bool(expansions)
            diagnostics["query_expansions"] = expansions
            for variant in expansions:
                query_variants.append((variant, 0.7))
        except Exception as exc:  # noqa: BLE001
            diagnostics["query_expansion_error"] = str(exc)

    with _db_conn() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            vector_state = _ensure_vector_index_ready(cur)
            diagnostics["vector_state"] = vector_state

            candidates: dict[int, dict[str, Any]] = {}

            if mode in {"lexical", "hybrid", "deep"}:
                for variant, weight in query_variants:
                    lexical_rows = _retrieve_lexical_rows(cur, query=variant, limit=candidate_limit)
                    _accumulate_ranked_rows(candidates, lexical_rows, channel="lexical", weight=weight)

            if mode in {"semantic", "hybrid", "deep"}:
                for variant, weight in query_variants:
                    query_embedding = _ollama_embed(variant)
                    query_vec = _vector_literal(query_embedding)
                    semantic_rows = _retrieve_semantic_rows(
                        cur,
                        query_vector_literal=query_vec,
                        embedding_model_id=vector_state["embedding_model_id"],
                        limit=candidate_limit,
                    )
                    _accumulate_ranked_rows(candidates, semantic_rows, channel="semantic", weight=weight)

            for entry in candidates.values():
                time_boost = _recency_boost(entry["ts"], enable=recency_hint)
                entry["time_boost"] = time_boost
                if mode == "lexical":
                    entry["final_score"] = float(entry["lexical_score"])
                elif mode == "semantic":
                    entry["final_score"] = float(entry["semantic_score"])
                else:
                    entry["final_score"] = float(entry["rrf_score"]) + float(time_boost)

            ordered = sorted(candidates.values(), key=_candidate_sort_key, reverse=True)
            top = ordered[: req.limit]
            health_map = _attach_health_context(cur, [int(item["note_id"]) for item in top]) if req.include_health else {}

        conn.commit()

    results = []
    for item in top:
        note_id = int(item["note_id"])
        vector_row_id = item.get("vector_row_id")
        citation = {
            "source_table": "memory.note",
            "source_id": note_id,
            "source_artifact_id": item["source_artifact_id"],
            "chunk_id": (
                f"search.vector_index:{vector_row_id}"
                if vector_row_id is not None
                else f"memory.note:{note_id}:{item['chunk_idx']}"
            ),
            "ts": item["ts"].isoformat(),
            "ts_mode": item["ts_mode"],
            "note_type": item["note_type"],
        }
        score = {
            "lexical": round(float(item["lexical_score"]), 6),
            "semantic": round(float(item["semantic_score"]), 6),
            "rrf": round(float(item["rrf_score"]), 6),
            "time_boost": round(float(item["time_boost"]), 6),
            "final": round(float(item["final_score"]), 6),
        }
        entry = {
            "note_id": note_id,
            "text": item["text"],
            "citation": citation,
            "score": score,
        }
        if req.include_health:
            entry["health_context"] = health_map.get(note_id, {"sample_count": 0, "min_ts": None, "max_ts": None})
        results.append(entry)

    return {
        "status": "ok",
        "query": req.query,
        "mode": mode,
        "result_count": len(results),
        "results": results,
        "diagnostics": diagnostics,
    }


def _compact_with_retries(artifact: dict[str, Any], caller: str) -> CompactionOutput:
    explicit_candidates = _extract_explicit_candidates(artifact["content"], ZoneInfo(artifact["timezone"]))
    json_schema = CompactionOutput.model_json_schema()
    previous_error = None
    for attempt in range(1, MAX_COMPACTION_RETRIES + 1):
        prompt = _build_compaction_prompt(artifact, explicit_candidates, previous_error)
        output_text = None
        try:
            output_text = _ollama_generate(prompt, response_format=json_schema)
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
    anchor_ts, anchor_mode = _resolve_default_anchor_ts(
        source_ts=artifact["source_ts"],
        day_scope=artifact["day_scope"],
        upload_ts=artifact["upload_ts"],
        explicit_candidates=explicit_candidates,
        tz=tz,
    )
    notes_created = 0
    links_created = 0
    note_ids: list[int] = []

    with _db_conn() as conn:
        with conn.cursor() as cur:
            for group in compacted.groups:
                group_ts, group_mode = _resolve_group_hint_ts(
                    hint=group.hint,
                    anchor_ts=anchor_ts,
                    anchor_mode=anchor_mode,
                    tz=tz,
                )
                for item_text in group.items:
                    embedding = _ollama_embed(item_text.text)
                    cur.execute(
                        """
                        INSERT INTO memory.note(ts, ts_mode, note_type, text, embedding, source_artifact_id, created_at)
                        VALUES(%s, %s, %s, %s, %s::vector, %s, %s)
                        RETURNING id
                        """,
                        (
                            group_ts,
                            group_mode,
                            item_text.note_type,
                            item_text.text,
                            _vector_literal(embedding),
                            artifact_id,
                            datetime.now(timezone.utc),
                        ),
                    )
                    note_id = int(cur.fetchone()[0])
                    note_ids.append(note_id)
                    notes_created += 1
                    _upsert_vector_index_row(
                        cur,
                        note_id=note_id,
                        text=item_text.text,
                        embedding=embedding,
                        ts_mode=group_mode,
                        note_type=item_text.note_type,
                        source_artifact_id=artifact_id,
                    )
                    links_created += _link_note_to_health(cur, note_id, group_ts, group_mode)

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
                _assert_schema_ready(cur)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"postgres unavailable: {exc}") from exc

    model_state = _model_state()

    return {
        "status": "ready",
        "service": SERVICE,
        "database": "postgres",
        "model": {"generation": OLLAMA_MODEL, "embedding": OLLAMA_EMBED_MODEL},
        "model_state": model_state,
        "embed_dim": OLLAMA_EMBED_DIM,
    }


@app.get("/ready/model")
def ready_model() -> dict[str, Any]:
    state = _model_state()
    if not state["ollama_reachable"]:
        raise HTTPException(status_code=503, detail="ollama unavailable")
    if not state["generation_model_present"]:
        raise HTTPException(status_code=503, detail=f"required model missing: {OLLAMA_MODEL}")
    if not state["embedding_model_present"]:
        raise HTTPException(status_code=503, detail=f"required embed model missing: {OLLAMA_EMBED_MODEL}")

    return {
        "status": "ready",
        "service": SERVICE,
        "model": {"generation": OLLAMA_MODEL, "embedding": OLLAMA_EMBED_MODEL},
        "model_state": state,
        "embed_dim": OLLAMA_EMBED_DIM,
    }


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


@app.post("/pipeline/search/reindex/memory_notes")
def pipeline_search_reindex_memory_notes() -> dict[str, Any]:
    with _db_conn() as conn:
        with conn.cursor() as cur:
            stats = _upsert_all_memory_note_vectors(cur)
        conn.commit()
    return {"status": "ok", **stats}


@app.post("/tools/retrieve")
def tools_retrieve(req: RetrievalRequest) -> dict[str, Any]:
    return _run_retrieval(req)


@app.get("/status")
def status() -> dict[str, Any]:
    with _db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM health.sample")
            health_count = int(cur.fetchone()[0])
            cur.execute("SELECT COUNT(*) FROM core.audio_note")
            audio_count = int(cur.fetchone()[0])
            cur.execute("SELECT COUNT(*) FROM core.ingest_artifact")
            artifact_count = int(cur.fetchone()[0])
            cur.execute("SELECT COUNT(*) FROM memory.note")
            note_count = int(cur.fetchone()[0])
            cur.execute("SELECT COUNT(*) FROM memory.health_link")
            link_count = int(cur.fetchone()[0])
            cur.execute("SELECT COUNT(*) FROM search.vector_index WHERE target_table = 'memory.note' AND is_active = TRUE")
            vector_index_count = int(cur.fetchone()[0])
            cur.execute("SELECT COUNT(*) FROM search.embedding_model WHERE status = 'active'")
            embedding_model_count = int(cur.fetchone()[0])
            cur.execute("SELECT COUNT(*) FROM core.compaction_attempt")
            attempt_count = int(cur.fetchone()[0])
    return {
        "service": SERVICE,
        "database": "postgres",
        "model": {"generation": OLLAMA_MODEL, "embedding": OLLAMA_EMBED_MODEL},
        "model_state": _model_state(),
        "health_sample_count_total": health_count,
        "audio_note_count_total": audio_count,
        "artifact_count": artifact_count,
        "memory_note_count": note_count,
        "memory_health_link_count": link_count,
        "vector_index_count": vector_index_count,
        "embedding_model_count": embedding_model_count,
        "compaction_attempt_count": attempt_count,
        "embed_dim": OLLAMA_EMBED_DIM,
    }
