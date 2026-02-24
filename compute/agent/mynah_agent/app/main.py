import json
import os
import re
import sqlite3
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

SERVICE = os.getenv("MYNAH_SERVICE_NAME", "mynah_agent")
DB_PATH = Path(os.getenv("MYNAH_DB_PATH", "/data/db/mynah.db"))
ARTIFACTS_PATH = Path(os.getenv("MYNAH_ARTIFACTS_PATH", "/home/appuser/data/artifacts"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
MAX_QUERY_RUNTIME_MS = 5000
MAX_QUERY_ROWS = 10_000
MAX_QUERY_PAYLOAD_BYTES = 5 * 1024 * 1024
MEMORY_TYPES = {"fact", "event", "note", "insight", "procedure"}
MEMORY_SENSITIVITY_LEVELS = {"low", "personal", "sensitive"}
MIN_MEMORY_SALIENCE = 0.50
MIN_MEMORY_CONFIDENCE = 0.70
MAX_MEMORY_WRITES_PER_HOUR = 120
MEMORY_TTL_DAYS = {
    "fact": 365,
    "event": 30,
    "note": 30,
    "insight": 14,
    "procedure": 90,
}
ALLOWED_QUERY_TABLES = {
    "agent_run_log",
    "analysis_run",
    "analysis_step",
    "audio_note",
    "device",
    "hr_sample",
    "memory_citation",
    "memory_item",
    "memory_revision",
    "memory_write_audit",
    "query_audit",
    "report_artifact",
    "service_heartbeat",
    "transcript",
}
TREND_OPERATION_TYPES = {"pearson", "mean_delta_by_flag", "slope_by_time"}
NUMERIC_FIELD_ALIASES = {
    "sleep_h": {"sleep", "sleep hours", "hours slept"},
    "mood": {"mood"},
    "stress": {"stress"},
    "pain": {"pain", "back pain", "lower-back pain"},
    "exercise_min": {"exercise", "movement", "activity"},
    "wakeups": {"wakeups", "wake-ups", "wake ups"},
}
BINARY_FIELD_ALIASES = {
    "caffeine_late": {"caffeine", "late caffeine"},
    "weekend": {"weekend", "weekends", "weekday weekend"},
}

FORBIDDEN_SQL_PATTERN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|ATTACH|DETACH|PRAGMA|VACUUM|REINDEX|TRIGGER|REPLACE)\b",
    re.IGNORECASE,
)
ALLOWED_SQL_START_PATTERN = re.compile(r"^(SELECT|WITH|EXPLAIN\s+QUERY\s+PLAN)\b", re.IGNORECASE)
LIMIT_PATTERN = re.compile(r"\bLIMIT\s+\d+\b", re.IGNORECASE)
TABLE_REF_PATTERN = re.compile(r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)
CTE_NAME_PATTERN = re.compile(r"(?:\bWITH\b|,)\s*([A-Za-z_][A-Za-z0-9_]*)\s+AS\b", re.IGNORECASE)

app = FastAPI(title="mynah_agent", version="0.1.0")


class AnalyzeRequest(BaseModel):
    prompt: str


class SqlQueryReadonlyRequest(BaseModel):
    query: str = Field(min_length=1, max_length=5000)
    params: list[str | int | float | None] = Field(default_factory=list)
    caller: str = Field(default="local_ui", min_length=1, max_length=64)


class MemoryCitationIn(BaseModel):
    source_type: str = Field(min_length=1, max_length=32, pattern=r"^[a-z_]+$")
    source_ref: str = Field(min_length=1, max_length=256)
    content_hash: str = Field(min_length=8, max_length=128)
    schema_version: int = Field(default=1, ge=1, le=10_000)
    snapshot_ref: str = Field(min_length=1, max_length=128)


class MemoryUpsertRequest(BaseModel):
    type: str = Field(min_length=1, max_length=32)
    title: str = Field(min_length=1, max_length=200)
    summary: str = Field(min_length=1, max_length=2000)
    tags: list[str] = Field(default_factory=list, max_length=32)
    sensitivity: str = Field(default="personal")
    salience_score: float = Field(ge=0.0, le=1.0)
    confidence_score: float = Field(ge=0.0, le=1.0)
    citations: list[MemoryCitationIn] = Field(min_length=1, max_length=20)
    caller: str = Field(default="local_ui", min_length=1, max_length=64)
    supersedes_memory_id: str | None = Field(default=None, min_length=8, max_length=64)

    @field_validator("type")
    @classmethod
    def validate_type(cls, value: str) -> str:
        value = value.lower().strip()
        if value not in MEMORY_TYPES:
            raise ValueError(f"invalid memory type: {value}")
        return value

    @field_validator("sensitivity")
    @classmethod
    def validate_sensitivity(cls, value: str) -> str:
        value = value.lower().strip()
        if value not in MEMORY_SENSITIVITY_LEVELS:
            raise ValueError(f"invalid sensitivity level: {value}")
        return value

    @field_validator("tags")
    @classmethod
    def normalize_tags(cls, value: list[str]) -> list[str]:
        normalized = []
        seen = set()
        for tag in value:
            t = tag.strip().lower()
            if not t:
                continue
            if t not in seen:
                normalized.append(t)
                seen.add(t)
        return normalized


class MemorySearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=200)
    limit: int = Field(default=20, ge=1, le=100)
    verified_only: bool = Field(default=True)


class AudioTranscribeRequest(BaseModel):
    audio_id: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9._:-]+$")
    caller: str = Field(default="local_ui", min_length=1, max_length=64)
    force: bool = Field(default=False)


class ReportGenerateRequest(BaseModel):
    date: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    caller: str = Field(default="local_ui", min_length=1, max_length=64)


class TrendMetricSpec(BaseModel):
    name: str = Field(min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_.:-]+$")
    operation: Literal["pearson", "mean_delta_by_flag", "slope_by_time"]
    x_field: str | None = Field(default=None, min_length=1, max_length=64)
    y_field: str | None = Field(default=None, min_length=1, max_length=64)
    value_field: str | None = Field(default=None, min_length=1, max_length=64)
    flag_field: str | None = Field(default=None, min_length=1, max_length=64)
    target_field: str | None = Field(default=None, min_length=1, max_length=64)


class TrendDirectionCheck(BaseModel):
    name: str = Field(min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_.:-]+$")
    metric: str = Field(min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_.:-]+$")
    expected_sign: Literal["positive", "negative", "nonzero"]


class TrendAnalysisRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    caller: str = Field(default="local_ui", min_length=1, max_length=64)
    text_filter: str | None = Field(default=None, max_length=128)
    created_from: str | None = Field(default=None, max_length=64)
    created_to: str | None = Field(default=None, max_length=64)
    limit: int = Field(default=500, ge=10, le=5000)
    metrics: list[TrendMetricSpec] = Field(default_factory=list, max_length=32)
    direction_checks: list[TrendDirectionCheck] = Field(default_factory=list, max_length=32)
    include_llm_summary: bool = Field(default=False)


class SQLValidationError(Exception):
    def __init__(self, code: str, reason: str, suggestion: str, retryable: bool = False):
        super().__init__(reason)
        self.code = code
        self.reason = reason
        self.suggestion = suggestion
        self.retryable = retryable


class MemoryGovernanceError(Exception):
    def __init__(self, code: str, reason: str, suggestion: str, retryable: bool = False):
        super().__init__(reason)
        self.code = code
        self.reason = reason
        self.suggestion = suggestion
        self.retryable = retryable


def _init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_run_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                prompt TEXT NOT NULL,
                model TEXT NOT NULL,
                response TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS query_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                caller TEXT NOT NULL,
                query_text TEXT NOT NULL,
                params_hash TEXT NOT NULL,
                latency_ms INTEGER NOT NULL,
                row_count INTEGER NOT NULL,
                status TEXT NOT NULL,
                denial_reason TEXT,
                result_hash TEXT
            )
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
                ingested_at TEXT NOT NULL
            )
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
            CREATE TABLE IF NOT EXISTS report_artifact (
                report_date TEXT PRIMARY KEY,
                path TEXT NOT NULL,
                created_at TEXT NOT NULL,
                generator TEXT NOT NULL,
                note TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_item (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                title TEXT NOT NULL,
                summary TEXT NOT NULL,
                tags_json TEXT NOT NULL,
                sensitivity TEXT NOT NULL,
                dedupe_hash TEXT NOT NULL,
                superseded_by TEXT,
                is_deleted INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.execute("DROP INDEX IF EXISTS idx_memory_item_dedupe_hash")
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_memory_item_dedupe_hash ON memory_item(dedupe_hash)
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_citation (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_ref TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                schema_version INTEGER NOT NULL,
                snapshot_ref TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(memory_id) REFERENCES memory_item(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_revision (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id TEXT NOT NULL,
                revision INTEGER NOT NULL,
                reason TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(memory_id) REFERENCES memory_item(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_write_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                caller TEXT NOT NULL,
                status TEXT NOT NULL,
                memory_id TEXT,
                reason TEXT,
                salience_score REAL NOT NULL,
                confidence_score REAL NOT NULL,
                dedupe_hash TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_run (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                caller TEXT NOT NULL,
                status TEXT NOT NULL,
                prompt TEXT NOT NULL,
                request_json TEXT NOT NULL,
                response_json TEXT,
                error TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_step (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                step_name TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                input_json TEXT,
                output_json TEXT,
                error TEXT,
                FOREIGN KEY(run_id) REFERENCES analysis_run(id)
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_analysis_step_run_id ON analysis_step(run_id)
            """
        )
        conn.commit()


def _ollama_tags() -> dict:
    req = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/tags", method="GET")
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _model_available(model_name: str) -> bool:
    tags = _ollama_tags()
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
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("response", "").strip()


def _hash_payload(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return sha256(encoded).hexdigest()


def _normalize_query(query: str) -> str:
    normalized = query.strip()
    if normalized.endswith(";"):
        normalized = normalized[:-1].strip()
    return normalized


def _validate_readonly_query(query: str) -> str:
    normalized = _normalize_query(query)
    if ";" in normalized:
        raise SQLValidationError(
            code="SQL_MULTI_STATEMENT_NOT_ALLOWED",
            reason="multiple SQL statements are not allowed",
            suggestion="submit exactly one read-only query statement",
        )
    if not ALLOWED_SQL_START_PATTERN.match(normalized):
        raise SQLValidationError(
            code="SQL_STATEMENT_NOT_ALLOWED",
            reason="only SELECT, WITH, or EXPLAIN QUERY PLAN statements are allowed",
            suggestion="rewrite the query as a read-only SELECT statement",
        )
    if FORBIDDEN_SQL_PATTERN.search(normalized):
        raise SQLValidationError(
            code="SQL_FORBIDDEN_KEYWORD",
            reason="query contains forbidden mutating or privilege-expanding keyword",
            suggestion="remove mutating keywords and keep query read-only",
        )
    upper = normalized.upper()
    if (upper.startswith("SELECT") or upper.startswith("WITH")) and not LIMIT_PATTERN.search(normalized):
        raise SQLValidationError(
            code="SQL_LIMIT_REQUIRED",
            reason="query must include an explicit LIMIT clause",
            suggestion="add LIMIT <n> to keep result sets bounded",
        )
    cte_names = {name.lower() for name in CTE_NAME_PATTERN.findall(normalized)}
    table_refs = {name.lower() for name in TABLE_REF_PATTERN.findall(normalized)}
    disallowed_tables = sorted(table_refs - cte_names - ALLOWED_QUERY_TABLES)
    if disallowed_tables:
        raise SQLValidationError(
            code="SQL_TABLE_NOT_ALLOWED",
            reason=f"query references non-allowlisted table(s): {', '.join(disallowed_tables)}",
            suggestion="query only analytics-safe allowlisted tables",
        )
    return normalized


def _sql_error(code: str, reason: str, suggestion: str, retryable: bool = False) -> dict:
    return {"code": code, "reason": reason, "retryable": retryable, "suggestion": suggestion}


def _audit_query(
    *,
    caller: str,
    query_text: str,
    params_hash: str,
    latency_ms: int,
    row_count: int,
    status: str,
    denial_reason: str | None,
    result_hash: str | None,
) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO query_audit(
                created_at, caller, query_text, params_hash, latency_ms, row_count, status, denial_reason, result_hash
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                caller,
                query_text,
                params_hash,
                latency_ms,
                row_count,
                status,
                denial_reason,
                result_hash,
            ),
        )
        conn.commit()


def _run_readonly_sql(query: str, params: list[str | int | float | None]) -> tuple[list[dict], int]:
    db_uri = f"file:{DB_PATH.as_posix()}?mode=ro"
    with sqlite3.connect(db_uri, uri=True, timeout=MAX_QUERY_RUNTIME_MS / 1000) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(query, tuple(params))
        rows = cursor.fetchmany(MAX_QUERY_ROWS + 1)
        if len(rows) > MAX_QUERY_ROWS:
            raise SQLValidationError(
                code="SQL_ROW_LIMIT_EXCEEDED",
                reason=f"query returned more than {MAX_QUERY_ROWS} rows",
                suggestion=f"decrease result size with a smaller LIMIT (max rows: {MAX_QUERY_ROWS})",
            )
    result_rows = [dict(row) for row in rows]
    payload_size = len(json.dumps(result_rows, default=str).encode("utf-8"))
    if payload_size > MAX_QUERY_PAYLOAD_BYTES:
        raise SQLValidationError(
            code="SQL_PAYLOAD_TOO_LARGE",
            reason=f"query result exceeds {MAX_QUERY_PAYLOAD_BYTES} bytes",
            suggestion="select fewer columns or reduce LIMIT",
        )
    return result_rows, payload_size


def _analysis_run_start(caller: str, prompt: str, request_payload: dict) -> str:
    run_id = str(uuid4())
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO analysis_run(id, created_at, caller, status, prompt, request_json)
            VALUES(?, ?, ?, 'running', ?, ?)
            """,
            (run_id, now, caller, prompt, json.dumps(request_payload, sort_keys=True, default=str)),
        )
        conn.commit()
    return run_id


def _analysis_run_finish(run_id: str, status: str, response_payload: dict | None, error: str | None) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            UPDATE analysis_run
            SET status = ?, response_json = ?, error = ?
            WHERE id = ?
            """,
            (
                status,
                json.dumps(response_payload, sort_keys=True, default=str) if response_payload is not None else None,
                error,
                run_id,
            ),
        )
        conn.commit()


def _analysis_step_start(run_id: str, step_name: str, step_input: dict | None) -> int:
    started_at = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            """
            INSERT INTO analysis_step(run_id, step_name, status, started_at, input_json)
            VALUES(?, ?, 'running', ?, ?)
            """,
            (
                run_id,
                step_name,
                started_at,
                json.dumps(step_input, sort_keys=True, default=str) if step_input is not None else None,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def _analysis_step_finish(step_id: int, status: str, step_output: dict | None, error: str | None) -> None:
    ended_at = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            UPDATE analysis_step
            SET status = ?, ended_at = ?, output_json = ?, error = ?
            WHERE id = ?
            """,
            (
                status,
                ended_at,
                json.dumps(step_output, sort_keys=True, default=str) if step_output is not None else None,
                error,
                step_id,
            ),
        )
        conn.commit()


def _analysis_step_execute(run_id: str, step_name: str, step_input: dict, worker) -> dict:
    step_id = _analysis_step_start(run_id, step_name, step_input)
    try:
        output = worker()
    except Exception as exc:  # noqa: BLE001
        _analysis_step_finish(step_id, "failed", None, str(exc))
        raise
    _analysis_step_finish(step_id, "completed", output, None)
    return output


def _extract_prompt_run_id(prompt: str) -> str | None:
    match = re.search(r"run_id\s*[:=]\s*([A-Za-z0-9._:-]+)", prompt, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip().rstrip(".,;:)]}!?")


def _fetch_transcript_rows(req: TrendAnalysisRequest) -> list[sqlite3.Row]:
    run_id = _extract_prompt_run_id(req.prompt)
    where_parts: list[str] = []
    params: list[str | int] = []

    if req.text_filter:
        where_parts.append("text LIKE ?")
        params.append(f"%{req.text_filter.strip()}%")
    if run_id:
        where_parts.append("text LIKE ?")
        params.append(f"%run_id={run_id}%")
    if req.created_from:
        where_parts.append("created_at >= ?")
        params.append(req.created_from)
    if req.created_to:
        where_parts.append("created_at <= ?")
        params.append(req.created_to)

    where_clause = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""
    query = (
        "SELECT audio_id, text, created_at FROM transcript "
        f"{where_clause} "
        "ORDER BY created_at ASC LIMIT ?"
    )
    params.append(req.limit)
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query, tuple(params)).fetchall()
    return rows


def _parse_boolish(value: str) -> int | None:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return 1
    if normalized in {"0", "false", "no", "n"}:
        return 0
    return None


def _safe_float(value: str) -> float | None:
    try:
        return float(value.strip())
    except ValueError:
        return None


def _extract_feature_record(text: str, created_at: str) -> dict[str, float]:
    record: dict[str, float] = {}

    metadata_matches = re.findall(r"\[([^\[\]]+)\]", text)
    if metadata_matches:
        key_map = {
            "sleep_h": "sleep_h",
            "sleep": "sleep_h",
            "mood": "mood",
            "stress": "stress",
            "pain": "pain",
            "exercise_min": "exercise_min",
            "exercise": "exercise_min",
            "caffeine_late": "caffeine_late",
            "late_caffeine": "caffeine_late",
            "weekend": "weekend",
            "wakeups": "wakeups",
        }
        for token in metadata_matches[-1].split(";"):
            if "=" not in token:
                continue
            raw_key, raw_value = token.split("=", 1)
            key = key_map.get(raw_key.strip().lower())
            if not key:
                continue
            boolish = _parse_boolish(raw_value)
            if boolish is not None:
                record[key] = float(boolish)
                continue
            as_float = _safe_float(raw_value)
            if as_float is not None:
                record[key] = as_float

    sleep_match = re.search(r"slept\s+([0-9]+(?:\.[0-9]+)?)\s+hours", text, flags=re.IGNORECASE)
    if sleep_match and "sleep_h" not in record:
        record["sleep_h"] = float(sleep_match.group(1))

    mood_match = re.search(r"mood\s+was\s+([0-9]+(?:\.[0-9]+)?)\s*/\s*10", text, flags=re.IGNORECASE)
    if mood_match and "mood" not in record:
        record["mood"] = float(mood_match.group(1))

    stress_match = re.search(r"stress\s+([0-9]+(?:\.[0-9]+)?)\s*/\s*10", text, flags=re.IGNORECASE)
    if stress_match and "stress" not in record:
        record["stress"] = float(stress_match.group(1))

    pain_match = re.search(r"pain\s+(?:reached|was|is)?\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*10", text, flags=re.IGNORECASE)
    if pain_match and "pain" not in record:
        record["pain"] = float(pain_match.group(1))

    exercise_match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s+minutes", text, flags=re.IGNORECASE)
    if exercise_match and "exercise_min" not in record:
        record["exercise_min"] = float(exercise_match.group(1))

    caffeine_match = re.search(r"late caffeine\s*:\s*(yes|no|true|false|1|0)", text, flags=re.IGNORECASE)
    if caffeine_match and "caffeine_late" not in record:
        parsed = _parse_boolish(caffeine_match.group(1))
        if parsed is not None:
            record["caffeine_late"] = float(parsed)

    wakeups_match = re.search(r"([0-9]+)\s+wake[- ]?ups", text, flags=re.IGNORECASE)
    if wakeups_match and "wakeups" not in record:
        record["wakeups"] = float(wakeups_match.group(1))

    if "weekend" not in record:
        try:
            day = datetime.fromisoformat(created_at).astimezone(timezone.utc).weekday()
            record["weekend"] = 1.0 if day >= 5 else 0.0
        except ValueError:
            pass

    try:
        ts = datetime.fromisoformat(created_at).astimezone(timezone.utc).timestamp()
    except ValueError:
        ts = 0.0
    record["_ts_epoch"] = float(ts)
    return record


def _extract_records(rows: list[sqlite3.Row]) -> list[dict[str, float]]:
    records: list[dict[str, float]] = []
    for row in rows:
        extracted = _extract_feature_record(str(row["text"]), str(row["created_at"]))
        extracted["_audio_id"] = str(row["audio_id"])
        records.append(extracted)
    return records


def _field_available(records: list[dict[str, float]], field: str) -> bool:
    return sum(1 for record in records if field in record) >= 3


def _alias_present(prompt_lower: str, aliases: set[str]) -> bool:
    return any(alias in prompt_lower for alias in aliases)


def _infer_prompt_metrics(prompt: str, records: list[dict[str, float]]) -> list[TrendMetricSpec]:
    prompt_lower = prompt.lower()
    metrics: list[TrendMetricSpec] = []
    seen: set[str] = set()

    def add(metric: TrendMetricSpec) -> None:
        if metric.name in seen:
            return
        seen.add(metric.name)
        metrics.append(metric)

    numeric_mentions = {
        field: _alias_present(prompt_lower, aliases)
        for field, aliases in NUMERIC_FIELD_ALIASES.items()
        if _field_available(records, field)
    }
    binary_mentions = {
        field: _alias_present(prompt_lower, aliases)
        for field, aliases in BINARY_FIELD_ALIASES.items()
        if _field_available(records, field)
    }

    numeric_fields = [field for field, present in numeric_mentions.items() if present]
    binary_fields = [field for field, present in binary_mentions.items() if present]

    connector_tokens = (" vs ", " versus ", " correlation", " relationship", " related", " linked", " trend")
    for idx, left in enumerate(numeric_fields):
        for right in numeric_fields[idx + 1 :]:
            left_aliases = NUMERIC_FIELD_ALIASES[left]
            right_aliases = NUMERIC_FIELD_ALIASES[right]
            has_explicit_vs = any(
                (f"{left_alias} vs {right_alias}" in prompt_lower) or (f"{right_alias} vs {left_alias}" in prompt_lower)
                for left_alias in left_aliases
                for right_alias in right_aliases
            )
            if has_explicit_vs:
                add(
                    TrendMetricSpec(
                        name=f"corr_{left}_{right}",
                        operation="pearson",
                        x_field=left,
                        y_field=right,
                    )
                )
                continue
            if any(token in prompt_lower for token in connector_tokens) and numeric_mentions.get(left, False) and numeric_mentions.get(right, False):
                add(
                    TrendMetricSpec(
                        name=f"corr_{left}_{right}",
                        operation="pearson",
                        x_field=left,
                        y_field=right,
                    )
                )

    for flag_field in binary_fields:
        for value_field in numeric_fields:
            if flag_field == value_field:
                continue
            if f"effect of {flag_field.replace('_', ' ')} on {value_field.replace('_', ' ')}" in prompt_lower:
                add(
                    TrendMetricSpec(
                        name=f"delta_{value_field}_by_{flag_field}",
                        operation="mean_delta_by_flag",
                        value_field=value_field,
                        flag_field=flag_field,
                    )
                )
                continue
            if (
                "effect" in prompt_lower
                or "impact" in prompt_lower
                or "difference" in prompt_lower
                or "delta" in prompt_lower
                or "when" in prompt_lower
            ) and numeric_mentions.get(value_field, False) and binary_mentions.get(flag_field, False):
                add(
                    TrendMetricSpec(
                        name=f"delta_{value_field}_by_{flag_field}",
                        operation="mean_delta_by_flag",
                        value_field=value_field,
                        flag_field=flag_field,
                    )
                )

    if "over time" in prompt_lower or "monthly trend" in prompt_lower or "slope" in prompt_lower:
        for field in numeric_fields:
            add(
                TrendMetricSpec(
                    name=f"slope_{field}",
                    operation="slope_by_time",
                    target_field=field,
                )
            )

    return metrics


def _validate_metric_spec(metric: TrendMetricSpec, records: list[dict[str, float]]) -> None:
    if metric.operation not in TREND_OPERATION_TYPES:
        raise ValueError(f"unsupported metric operation: {metric.operation}")

    def require_field(field_name: str | None, usage: str) -> str:
        if not field_name:
            raise ValueError(f"{metric.name} requires field `{usage}`")
        if not _field_available(records, field_name):
            raise ValueError(f"{metric.name} references unavailable field `{field_name}`")
        return field_name

    if metric.operation == "pearson":
        require_field(metric.x_field, "x_field")
        require_field(metric.y_field, "y_field")
    elif metric.operation == "mean_delta_by_flag":
        value_field = require_field(metric.value_field, "value_field")
        flag_field = require_field(metric.flag_field, "flag_field")
        values = {int(record[flag_field]) for record in records if flag_field in record}
        if not values.issubset({0, 1}):
            raise ValueError(f"{metric.name} flag field `{flag_field}` must be binary (0/1)")
        if value_field == flag_field:
            raise ValueError(f"{metric.name} cannot compare field `{value_field}` against itself")
    elif metric.operation == "slope_by_time":
        require_field(metric.target_field, "target_field")


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denom_x = sum((x - mean_x) ** 2 for x in xs) ** 0.5
    denom_y = sum((y - mean_y) ** 2 for y in ys) ** 0.5
    if denom_x == 0.0 or denom_y == 0.0:
        return 0.0
    return numerator / denom_x / denom_y


def _linear_slope(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def _compute_metric(metric: TrendMetricSpec, records: list[dict[str, float]]) -> dict:
    if metric.operation == "pearson":
        points = [(record[metric.x_field], record[metric.y_field]) for record in records if metric.x_field in record and metric.y_field in record]
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        return {
            "name": metric.name,
            "operation": metric.operation,
            "fields": {"x_field": metric.x_field, "y_field": metric.y_field},
            "sample_count": len(points),
            "value": round(_pearson(xs, ys), 6),
        }

    if metric.operation == "mean_delta_by_flag":
        ones = [float(record[metric.value_field]) for record in records if metric.value_field in record and metric.flag_field in record and int(record[metric.flag_field]) == 1]
        zeros = [float(record[metric.value_field]) for record in records if metric.value_field in record and metric.flag_field in record and int(record[metric.flag_field]) == 0]
        mean_ones = (sum(ones) / len(ones)) if ones else 0.0
        mean_zeros = (sum(zeros) / len(zeros)) if zeros else 0.0
        delta = mean_ones - mean_zeros
        return {
            "name": metric.name,
            "operation": metric.operation,
            "fields": {"value_field": metric.value_field, "flag_field": metric.flag_field},
            "sample_count": len(ones) + len(zeros),
            "group_one_count": len(ones),
            "group_zero_count": len(zeros),
            "value": round(delta, 6),
            "group_one_mean": round(mean_ones, 6),
            "group_zero_mean": round(mean_zeros, 6),
        }

    points = [(record["_ts_epoch"], record[metric.target_field]) for record in records if "_ts_epoch" in record and metric.target_field in record]
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return {
        "name": metric.name,
        "operation": metric.operation,
        "fields": {"target_field": metric.target_field},
        "sample_count": len(points),
        "value": round(_linear_slope(xs, ys), 10),
    }


def _compute_metrics_parallel(metric_specs: list[TrendMetricSpec], records: list[dict[str, float]]) -> dict:
    if not metric_specs:
        return {"metrics": []}
    max_workers = min(8, len(metric_specs))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(lambda spec: _compute_metric(spec, records), metric_specs))
    return {"metrics": results}


def _metric_sign(value: float) -> str:
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "zero"


def _evaluate_direction_checks(direction_checks: list[TrendDirectionCheck], metric_results: list[dict]) -> list[dict]:
    metric_map = {metric["name"]: metric for metric in metric_results}
    results: list[dict] = []
    for check in direction_checks:
        metric = metric_map.get(check.metric)
        if not metric:
            results.append(
                {
                    "name": check.name,
                    "metric": check.metric,
                    "expected_sign": check.expected_sign,
                    "actual_sign": None,
                    "pass": False,
                    "reason": "metric_not_found",
                }
            )
            continue
        value = float(metric["value"])
        actual_sign = _metric_sign(value)
        if check.expected_sign == "nonzero":
            passed = actual_sign != "zero"
        else:
            passed = actual_sign == check.expected_sign
        results.append(
            {
                "name": check.name,
                "metric": check.metric,
                "expected_sign": check.expected_sign,
                "actual_sign": actual_sign,
                "pass": passed,
            }
        )
    return results


def _summarize_trend_metrics(prompt: str, metric_results: list[dict], direction_results: list[dict]) -> str:
    summary_prompt = (
        "You are summarizing deterministic metric outputs from a local trend-analysis pipeline.\n"
        "Respond with 4-8 concise bullet points, plain text only.\n\n"
        f"User request: {prompt}\n\n"
        f"Metrics: {json.dumps(metric_results, sort_keys=True)}\n"
        f"Direction checks: {json.dumps(direction_results, sort_keys=True)}"
    )
    return _ollama_generate(summary_prompt)


def _execute_trend_analysis(req: TrendAnalysisRequest) -> dict:
    request_payload = req.model_dump()
    run_id = _analysis_run_start(req.caller, req.prompt, request_payload)
    try:
        select_output = _analysis_step_execute(
            run_id,
            "select_transcripts",
            {"limit": req.limit, "text_filter": req.text_filter, "created_from": req.created_from, "created_to": req.created_to},
            lambda: {"rows": [dict(row) for row in _fetch_transcript_rows(req)]},
        )
        row_dicts = select_output["rows"]
        if not row_dicts:
            raise ValueError("no transcripts matched analysis scope")

        extract_output = _analysis_step_execute(
            run_id,
            "extract_features",
            {"row_count": len(row_dicts)},
            lambda: {"records": _extract_records(row_dicts)},
        )
        records = extract_output["records"]
        if not records:
            raise ValueError("no structured records extracted from transcripts")

        plan_output = _analysis_step_execute(
            run_id,
            "plan_metrics",
            {"provided_metrics": len(req.metrics), "prompt": req.prompt},
            lambda: {
                "metrics": [metric.model_dump() for metric in (req.metrics if req.metrics else _infer_prompt_metrics(req.prompt, records))]
            },
        )
        metric_specs = [TrendMetricSpec(**metric) for metric in plan_output["metrics"]]
        if not metric_specs:
            raise ValueError("no trend metrics could be inferred from the prompt; provide explicit metric specs")
        for metric in metric_specs:
            _validate_metric_spec(metric, records)

        compute_output = _analysis_step_execute(
            run_id,
            "compute_metrics",
            {"metric_count": len(metric_specs)},
            lambda: _compute_metrics_parallel(metric_specs, records),
        )
        metric_results = compute_output["metrics"]

        direction_checks = req.direction_checks
        if not direction_checks:
            auto_checks: list[TrendDirectionCheck] = []
            for metric in metric_specs:
                if metric.operation == "pearson":
                    auto_checks.append(
                        TrendDirectionCheck(
                            name=f"sign_{metric.name}",
                            metric=metric.name,
                            expected_sign="nonzero",
                        )
                    )
            direction_checks = auto_checks

        direction_output = _analysis_step_execute(
            run_id,
            "evaluate_directions",
            {"check_count": len(direction_checks)},
            lambda: {"direction_results": _evaluate_direction_checks(direction_checks, metric_results)},
        )

        llm_summary = None
        if req.include_llm_summary:
            llm_output = _analysis_step_execute(
                run_id,
                "llm_summary",
                {"metric_count": len(metric_results)},
                lambda: {"summary": _summarize_trend_metrics(req.prompt, metric_results, direction_output["direction_results"])},
            )
            llm_summary = llm_output["summary"]

        response = {
            "status": "ok",
            "run_id": run_id,
            "record_count": len(records),
            "metrics": metric_results,
            "direction_results": direction_output["direction_results"],
            "llm_summary": llm_summary,
        }
        _analysis_run_finish(run_id, "completed", response, None)
        return response
    except Exception as exc:  # noqa: BLE001
        _analysis_run_finish(run_id, "failed", None, str(exc))
        raise


def _required_citation_count(memory_type: str) -> int:
    return 2 if memory_type == "insight" else 1


def _memory_stale_state(memory_type: str, updated_at: str) -> tuple[bool, int]:
    ttl_days = MEMORY_TTL_DAYS.get(memory_type, 30)
    try:
        updated_ts = datetime.fromisoformat(updated_at)
    except ValueError:
        return True, ttl_days
    age_days = (datetime.now(timezone.utc) - updated_ts.astimezone(timezone.utc)).days
    return age_days > ttl_days, ttl_days


def _memory_error(code: str, reason: str, suggestion: str, retryable: bool = False) -> dict:
    return {"code": code, "reason": reason, "retryable": retryable, "suggestion": suggestion}


def _record_memory_write_audit(
    *,
    caller: str,
    status: str,
    memory_id: str | None,
    reason: str | None,
    salience_score: float,
    confidence_score: float,
    dedupe_hash: str,
) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO memory_write_audit(
                created_at, caller, status, memory_id, reason, salience_score, confidence_score, dedupe_hash
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                caller,
                status,
                memory_id,
                reason,
                salience_score,
                confidence_score,
                dedupe_hash,
            ),
        )
        conn.commit()


def _check_memory_rate_limit(conn: sqlite3.Connection, caller: str) -> None:
    window_start = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    row = conn.execute(
        """
        SELECT COUNT(*)
        FROM memory_write_audit
        WHERE caller = ? AND status = 'accepted' AND created_at >= ?
        """,
        (caller, window_start),
    ).fetchone()
    if int(row[0] if row else 0) >= MAX_MEMORY_WRITES_PER_HOUR:
        raise MemoryGovernanceError(
            code="MEMORY_WRITE_RATE_LIMITED",
            reason=f"memory write rate exceeded ({MAX_MEMORY_WRITES_PER_HOUR}/hour)",
            suggestion="retry later or lower write frequency",
            retryable=True,
        )


def _citation_exists(source_type: str, source_ref: str) -> bool:
    if source_type == "hr_sample":
        if "|" not in source_ref:
            return False
        device_id, ts = source_ref.split("|", 1)
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute(
                "SELECT 1 FROM hr_sample WHERE device_id = ? AND ts = ? LIMIT 1",
                (device_id, ts),
            ).fetchone()
        return row is not None
    if source_type == "transcript":
        path = Path(source_ref)
        if not path.is_absolute():
            path = ARTIFACTS_PATH / "transcripts" / path
        return path.exists()
    if source_type == "audio_note":
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute("SELECT 1 FROM audio_note WHERE id = ? LIMIT 1", (source_ref,)).fetchone()
        return row is not None
    return False


def _verify_memory(memory_id: str) -> dict:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        memory = conn.execute(
            """
            SELECT id, type, title, superseded_by, is_deleted, updated_at
            FROM memory_item
            WHERE id = ?
            """,
            (memory_id,),
        ).fetchone()
        if not memory:
            return {
                "status": "not_found",
                "memory_id": memory_id,
                "verified": False,
                "required_citations": 0,
                "valid_citations": 0,
                "total_citations": 0,
            }
        citations = conn.execute(
            """
            SELECT source_type, source_ref
            FROM memory_citation
            WHERE memory_id = ?
            """,
            (memory_id,),
        ).fetchall()

    required = _required_citation_count(memory["type"])
    valid = 0
    for citation in citations:
        if _citation_exists(citation["source_type"], citation["source_ref"]):
            valid += 1

    active = memory["superseded_by"] is None and int(memory["is_deleted"]) == 0
    stale, ttl_days = _memory_stale_state(memory["type"], memory["updated_at"])
    verified = active and not stale and valid >= required and len(citations) >= required
    return {
        "status": "ok",
        "memory_id": memory["id"],
        "type": memory["type"],
        "title": memory["title"],
        "active": active,
        "verified": verified,
        "stale": stale,
        "ttl_days": ttl_days,
        "required_citations": required,
        "valid_citations": valid,
        "total_citations": len(citations),
        "superseded_by": memory["superseded_by"],
    }


def _memory_dedupe_hash(memory_type: str, title: str, summary: str, tags: list[str]) -> str:
    payload = {
        "type": memory_type,
        "title": title.strip().lower(),
        "summary": summary.strip().lower(),
        "tags": sorted(tags),
    }
    return _hash_payload(payload)


def _next_memory_revision(conn: sqlite3.Connection, memory_id: str) -> int:
    row = conn.execute(
        "SELECT COALESCE(MAX(revision), 0) FROM memory_revision WHERE memory_id = ?",
        (memory_id,),
    ).fetchone()
    return int(row[0] if row else 0) + 1


def _create_memory(req: MemoryUpsertRequest) -> dict:
    if req.salience_score < MIN_MEMORY_SALIENCE:
        raise MemoryGovernanceError(
            code="MEMORY_SALIENCE_TOO_LOW",
            reason=f"salience_score must be >= {MIN_MEMORY_SALIENCE}",
            suggestion="raise salience threshold before writing memory",
        )
    if req.confidence_score < MIN_MEMORY_CONFIDENCE:
        raise MemoryGovernanceError(
            code="MEMORY_CONFIDENCE_TOO_LOW",
            reason=f"confidence_score must be >= {MIN_MEMORY_CONFIDENCE}",
            suggestion="raise confidence or avoid writing low-trust memory",
        )

    required_citations = _required_citation_count(req.type)
    if len(req.citations) < required_citations:
        raise MemoryGovernanceError(
            code="MEMORY_CITATION_MIN_NOT_MET",
            reason=f"{req.type} requires at least {required_citations} citation(s)",
            suggestion="add sufficient citations before upsert",
        )

    for citation in req.citations:
        if not _citation_exists(citation.source_type, citation.source_ref):
            raise MemoryGovernanceError(
                code="MEMORY_CITATION_INVALID_REF",
                reason=f"citation source reference not found: {citation.source_type}:{citation.source_ref}",
                suggestion="use source references that exist in local storage",
            )

    dedupe_hash = _memory_dedupe_hash(req.type, req.title, req.summary, req.tags)
    now = datetime.now(timezone.utc).isoformat()
    memory_id = str(uuid4())

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        _check_memory_rate_limit(conn, req.caller)

        dup = conn.execute(
            """
            SELECT id
            FROM memory_item
            WHERE dedupe_hash = ? AND is_deleted = 0 AND superseded_by IS NULL
            LIMIT 1
            """,
            (dedupe_hash,),
        ).fetchone()
        if dup:
            raise MemoryGovernanceError(
                code="MEMORY_DEDUPE_REJECTED",
                reason=f"duplicate active memory exists: {dup['id']}",
                suggestion="update/supersede existing memory instead of duplicating",
            )

        superseded_id = None
        if req.supersedes_memory_id:
            current = conn.execute(
                """
                SELECT id, superseded_by
                FROM memory_item
                WHERE id = ? AND is_deleted = 0
                LIMIT 1
                """,
                (req.supersedes_memory_id,),
            ).fetchone()
            if not current:
                raise MemoryGovernanceError(
                    code="MEMORY_SUPERSEDES_TARGET_NOT_FOUND",
                    reason=f"supersedes target not found: {req.supersedes_memory_id}",
                    suggestion="supply a valid active memory id",
                )
            if current["superseded_by"] is not None:
                raise MemoryGovernanceError(
                    code="MEMORY_SUPERSEDES_TARGET_INACTIVE",
                    reason=f"supersedes target already inactive: {req.supersedes_memory_id}",
                    suggestion="use an active memory as supersession target",
                )
            superseded_id = req.supersedes_memory_id

        conn.execute(
            """
            INSERT INTO memory_item(
                id, type, created_at, updated_at, title, summary, tags_json, sensitivity, dedupe_hash, superseded_by
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
            """,
            (
                memory_id,
                req.type,
                now,
                now,
                req.title.strip(),
                req.summary.strip(),
                json.dumps(req.tags),
                req.sensitivity,
                dedupe_hash,
            ),
        )

        for citation in req.citations:
            conn.execute(
                """
                INSERT INTO memory_citation(
                    memory_id, source_type, source_ref, content_hash, schema_version, snapshot_ref, created_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    citation.source_type,
                    citation.source_ref,
                    citation.content_hash,
                    citation.schema_version,
                    citation.snapshot_ref,
                    now,
                ),
            )

        created_payload = {
            "type": req.type,
            "title": req.title,
            "summary": req.summary,
            "tags": req.tags,
            "sensitivity": req.sensitivity,
            "citations": [citation.model_dump() for citation in req.citations],
            "supersedes_memory_id": superseded_id,
        }
        conn.execute(
            """
            INSERT INTO memory_revision(memory_id, revision, reason, payload_json, created_at)
            VALUES(?, 1, 'created', ?, ?)
            """,
            (memory_id, json.dumps(created_payload, sort_keys=True), now),
        )

        if superseded_id:
            conn.execute(
                """
                UPDATE memory_item
                SET superseded_by = ?, updated_at = ?
                WHERE id = ? AND superseded_by IS NULL
                """,
                (memory_id, now, superseded_id),
            )
            conn.execute(
                """
                INSERT INTO memory_revision(memory_id, revision, reason, payload_json, created_at)
                VALUES(?, ?, 'superseded', ?, ?)
                """,
                (
                    superseded_id,
                    _next_memory_revision(conn, superseded_id),
                    json.dumps({"superseded_by": memory_id}, sort_keys=True),
                    now,
                ),
            )
        conn.commit()

    _record_memory_write_audit(
        caller=req.caller,
        status="accepted",
        memory_id=memory_id,
        reason=None,
        salience_score=req.salience_score,
        confidence_score=req.confidence_score,
        dedupe_hash=dedupe_hash,
    )
    return {
        "status": "ok",
        "memory_id": memory_id,
        "supersedes_memory_id": superseded_id,
        "required_citations": required_citations,
        "provided_citations": len(req.citations),
    }


def _find_memory_by_transcript_ref(transcript_ref: str) -> str | None:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            """
            SELECT m.id
            FROM memory_item m
            JOIN memory_citation c ON c.memory_id = m.id
            WHERE m.is_deleted = 0
              AND m.superseded_by IS NULL
              AND c.source_type = 'transcript'
              AND c.source_ref = ?
            ORDER BY m.updated_at DESC
            LIMIT 1
            """,
            (transcript_ref,),
        ).fetchone()
    return row[0] if row else None


def _reverify_memory(memory_id: str) -> dict:
    verification = _verify_memory(memory_id)
    if verification["status"] != "ok":
        raise MemoryGovernanceError(
            code="MEMORY_NOT_FOUND",
            reason=f"memory not found: {memory_id}",
            suggestion="use an existing memory id",
        )
    if not verification["active"]:
        raise MemoryGovernanceError(
            code="MEMORY_NOT_ACTIVE",
            reason=f"memory is not active: {memory_id}",
            suggestion="reverify active non-superseded memory items only",
        )
    if verification["valid_citations"] < verification["required_citations"]:
        raise MemoryGovernanceError(
            code="MEMORY_REVERIFY_CITATION_FAILED",
            reason="memory citations did not pass reverification",
            suggestion="repair citation links before reverification",
        )

    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            UPDATE memory_item
            SET updated_at = ?
            WHERE id = ?
            """,
            (now, memory_id),
        )
        conn.execute(
            """
            INSERT INTO memory_revision(memory_id, revision, reason, payload_json, created_at)
            VALUES(?, ?, 'reverified', ?, ?)
            """,
            (
                memory_id,
                _next_memory_revision(conn, memory_id),
                json.dumps({"reverified_at": now}, sort_keys=True),
                now,
            ),
        )
        conn.commit()
    return _verify_memory(memory_id)


def _transcribe_audio_fixture(req: AudioTranscribeRequest) -> dict:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        audio = conn.execute(
            """
            SELECT id, transcript_hint_path, transcription_state
            FROM audio_note
            WHERE id = ?
            LIMIT 1
            """,
            (req.audio_id,),
        ).fetchone()
        if not audio:
            raise MemoryGovernanceError(
                code="AUDIO_NOTE_NOT_FOUND",
                reason=f"audio note not found: {req.audio_id}",
                suggestion="ingest audio note before transcription",
            )

        existing_transcript = conn.execute(
            """
            SELECT audio_id, text, path
            FROM transcript
            WHERE audio_id = ?
            LIMIT 1
            """,
            (req.audio_id,),
        ).fetchone()
        if existing_transcript and not req.force:
            transcript_ref = Path(existing_transcript["path"]).name
            existing_memory_id = _find_memory_by_transcript_ref(transcript_ref)
            return {
                "status": "ok",
                "audio_id": req.audio_id,
                "transcript_created": False,
                "memory_created": False,
                "memory_id": existing_memory_id,
                "transcript_ref": transcript_ref,
            }

    hint_path = Path(audio["transcript_hint_path"]) if audio["transcript_hint_path"] else None
    if not hint_path or not hint_path.exists():
        raise MemoryGovernanceError(
            code="TRANSCRIPT_HINT_MISSING",
            reason=f"no transcript fixture available for audio note {req.audio_id}",
            suggestion="ingest audio with transcript_hint for deterministic no-wearable E2E runs",
        )

    transcript_text = hint_path.read_text(encoding="utf-8").strip()
    if not transcript_text:
        raise MemoryGovernanceError(
            code="TRANSCRIPT_HINT_EMPTY",
            reason=f"transcript fixture is empty for audio note {req.audio_id}",
            suggestion="provide non-empty transcript_hint during audio ingest",
        )

    transcripts_dir = ARTIFACTS_PATH / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = transcripts_dir / f"{req.audio_id}.txt"
    transcript_path.write_text(transcript_text, encoding="utf-8")
    now = datetime.now(timezone.utc).isoformat()

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO transcript(audio_id, text, model, path, created_at)
            VALUES(?, ?, 'fixture_transcript_v1', ?, ?)
            ON CONFLICT(audio_id) DO UPDATE SET
                text=excluded.text,
                model=excluded.model,
                path=excluded.path,
                created_at=excluded.created_at
            """,
            (req.audio_id, transcript_text, str(transcript_path), now),
        )
        conn.execute(
            """
            UPDATE audio_note
            SET transcription_state = 'completed'
            WHERE id = ?
            """,
            (req.audio_id,),
        )
        conn.commit()

    transcript_ref = transcript_path.name
    existing_memory_id = _find_memory_by_transcript_ref(transcript_ref)
    if existing_memory_id:
        return {
            "status": "ok",
            "audio_id": req.audio_id,
            "transcript_created": True,
            "memory_created": False,
            "memory_id": existing_memory_id,
            "transcript_ref": transcript_ref,
        }

    memory_req = MemoryUpsertRequest(
        type="note",
        title=f"Voice note {req.audio_id[:8]}",
        summary=transcript_text[:1200],
        tags=["voice_note", "transcript"],
        sensitivity="personal",
        salience_score=0.80,
        confidence_score=0.90,
        caller=req.caller,
        citations=[
            MemoryCitationIn(
                source_type="transcript",
                source_ref=transcript_ref,
                content_hash=sha256(transcript_text.encode("utf-8")).hexdigest(),
                schema_version=1,
                snapshot_ref=now,
            )
        ],
    )
    memory_result = _create_memory(memory_req)
    return {
        "status": "ok",
        "audio_id": req.audio_id,
        "transcript_created": True,
        "memory_created": True,
        "memory_id": memory_result["memory_id"],
        "transcript_ref": transcript_ref,
    }


def _day_bounds(day: str) -> tuple[str, str]:
    return f"{day}T00:00:00+00:00", f"{day}T23:59:59.999999+00:00"


def _generate_report(req: ReportGenerateRequest) -> dict:
    report_date = req.date or datetime.now(timezone.utc).date().isoformat()
    start, end = _day_bounds(report_date)

    with sqlite3.connect(DB_PATH) as conn:
        hr_row = conn.execute(
            """
            SELECT COUNT(*), MIN(bpm), MAX(bpm), AVG(bpm)
            FROM hr_sample
            WHERE ts >= ? AND ts <= ?
            """,
            (start, end),
        ).fetchone()
        transcript_rows = conn.execute(
            """
            SELECT audio_id, SUBSTR(text, 1, 200) AS preview
            FROM transcript
            WHERE created_at >= ? AND created_at <= ?
            ORDER BY created_at DESC
            LIMIT 10
            """,
            (start, end),
        ).fetchall()

    sample_count = int(hr_row[0] or 0)
    min_bpm = int(hr_row[1]) if hr_row[1] is not None else None
    max_bpm = int(hr_row[2]) if hr_row[2] is not None else None
    avg_bpm = round(float(hr_row[3]), 2) if hr_row[3] is not None else None
    transcript_count = len(transcript_rows)

    lines = [
        f"# MYNAH Daily Report - {report_date}",
        "",
        "## HR Summary",
        f"- Samples: {sample_count}",
        f"- Min BPM: {min_bpm if min_bpm is not None else 'n/a'}",
        f"- Max BPM: {max_bpm if max_bpm is not None else 'n/a'}",
        f"- Avg BPM: {avg_bpm if avg_bpm is not None else 'n/a'}",
        "",
        "## Voice Notes",
        f"- Notes transcribed today: {transcript_count}",
    ]
    if transcript_rows:
        lines.append("")
        lines.append("### Transcript Previews")
        for row in transcript_rows:
            lines.append(f"- `{row[0]}`: {row[1]}")
    lines.append("")
    lines.append("## Generator")
    lines.append(f"- Created by: {SERVICE}")
    lines.append(f"- Requested by: {req.caller}")
    lines.append(f"- Created at (UTC): {datetime.now(timezone.utc).isoformat()}")
    markdown = "\n".join(lines) + "\n"

    report_dir = ARTIFACTS_PATH / "reports" / report_date
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "report.md"
    report_path.write_text(markdown, encoding="utf-8")
    now = datetime.now(timezone.utc).isoformat()

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO report_artifact(report_date, path, created_at, generator, note)
            VALUES(?, ?, ?, ?, ?)
            ON CONFLICT(report_date) DO UPDATE SET
                path=excluded.path,
                created_at=excluded.created_at,
                generator=excluded.generator,
                note=excluded.note
            """,
            (
                report_date,
                str(report_path),
                now,
                SERVICE,
                f"hr_samples={sample_count};transcripts={transcript_count}",
            ),
        )
        conn.commit()

    return {
        "status": "ok",
        "report_date": report_date,
        "path": str(report_path),
        "hr_samples": sample_count,
        "transcripts": transcript_count,
    }


@app.on_event("startup")
def startup() -> None:
    _init_db()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": SERVICE}


@app.get("/ready")
def ready() -> dict:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("SELECT 1").fetchone()
    try:
        tags = _ollama_tags()
    except urllib.error.URLError as exc:
        raise HTTPException(status_code=503, detail=f"ollama unavailable: {exc.reason}") from exc
    models = [m.get("name") for m in tags.get("models", [])]
    if OLLAMA_MODEL not in models:
        raise HTTPException(status_code=503, detail=f"required model missing: {OLLAMA_MODEL}")
    return {"status": "ready", "service": SERVICE, "model": OLLAMA_MODEL, "available_models": models}


@app.post("/tools/sql_query_readonly")
def sql_query_readonly(req: SqlQueryReadonlyRequest) -> dict:
    started = time.perf_counter()
    params_hash = _hash_payload(req.params)
    normalized_query = _normalize_query(req.query)
    try:
        validated_query = _validate_readonly_query(normalized_query)
        rows, payload_size = _run_readonly_sql(validated_query, req.params)
        latency_ms = int((time.perf_counter() - started) * 1000)
        _audit_query(
            caller=req.caller,
            query_text=validated_query,
            params_hash=params_hash,
            latency_ms=latency_ms,
            row_count=len(rows),
            status="accepted",
            denial_reason=None,
            result_hash=_hash_payload(rows),
        )
        return {
            "status": "ok",
            "row_count": len(rows),
            "rows": rows,
            "latency_ms": latency_ms,
            "payload_bytes": payload_size,
            "max_rows": MAX_QUERY_ROWS,
        }
    except SQLValidationError as exc:
        latency_ms = int((time.perf_counter() - started) * 1000)
        _audit_query(
            caller=req.caller,
            query_text=normalized_query,
            params_hash=params_hash,
            latency_ms=latency_ms,
            row_count=0,
            status="rejected",
            denial_reason=f"{exc.code}: {exc.reason}",
            result_hash=None,
        )
        raise HTTPException(
            status_code=400,
            detail=_sql_error(exc.code, exc.reason, exc.suggestion, exc.retryable),
        ) from exc
    except sqlite3.Error as exc:
        latency_ms = int((time.perf_counter() - started) * 1000)
        _audit_query(
            caller=req.caller,
            query_text=normalized_query,
            params_hash=params_hash,
            latency_ms=latency_ms,
            row_count=0,
            status="rejected",
            denial_reason=f"SQL_EXECUTION_FAILED: {exc}",
            result_hash=None,
        )
        raise HTTPException(
            status_code=400,
            detail=_sql_error(
                "SQL_EXECUTION_FAILED",
                f"sql execution failed: {exc}",
                "inspect query syntax and allowed tables",
                False,
            ),
        ) from exc


@app.get("/tools/query_audit/recent")
def query_audit_recent(limit: int = Query(default=20, ge=1, le=100)) -> dict:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, created_at, caller, query_text, latency_ms, row_count, status, denial_reason
            FROM query_audit
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return {"status": "ok", "entries": [dict(row) for row in rows]}


@app.post("/tools/memory_upsert")
def memory_upsert(req: MemoryUpsertRequest) -> dict:
    dedupe_hash = _memory_dedupe_hash(req.type, req.title, req.summary, req.tags)
    try:
        return _create_memory(req)
    except MemoryGovernanceError as exc:
        _record_memory_write_audit(
            caller=req.caller,
            status="rejected",
            memory_id=None,
            reason=f"{exc.code}: {exc.reason}",
            salience_score=req.salience_score,
            confidence_score=req.confidence_score,
            dedupe_hash=dedupe_hash,
        )
        raise HTTPException(
            status_code=400,
            detail=_memory_error(exc.code, exc.reason, exc.suggestion, exc.retryable),
        ) from exc
    except sqlite3.Error as exc:
        _record_memory_write_audit(
            caller=req.caller,
            status="rejected",
            memory_id=None,
            reason=f"MEMORY_DB_WRITE_FAILED: {exc}",
            salience_score=req.salience_score,
            confidence_score=req.confidence_score,
            dedupe_hash=dedupe_hash,
        )
        raise HTTPException(
            status_code=500,
            detail=_memory_error(
                "MEMORY_DB_WRITE_FAILED",
                f"memory write failed: {exc}",
                "inspect database integrity and retry",
                True,
            ),
        ) from exc


@app.get("/tools/memory_verify/{memory_id}")
def memory_verify(memory_id: str) -> dict:
    return _verify_memory(memory_id)


@app.post("/tools/memory_reverify/{memory_id}")
def memory_reverify(memory_id: str) -> dict:
    try:
        verification = _reverify_memory(memory_id)
    except MemoryGovernanceError as exc:
        raise HTTPException(
            status_code=400,
            detail=_memory_error(exc.code, exc.reason, exc.suggestion, exc.retryable),
        ) from exc
    return {"status": "ok", "memory_id": memory_id, "verification": verification}


@app.post("/tools/memory_search")
def memory_search(req: MemorySearchRequest) -> dict:
    pattern = f"%{req.query.strip()}%"
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, type, title, summary, tags_json, sensitivity, created_at, updated_at
            FROM memory_item
            WHERE is_deleted = 0
              AND superseded_by IS NULL
              AND (title LIKE ? OR summary LIKE ? OR tags_json LIKE ?)
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (pattern, pattern, pattern, req.limit),
        ).fetchall()

    result = []
    for row in rows:
        verification = _verify_memory(row["id"])
        if req.verified_only and not verification["verified"]:
            continue
        result.append(
            {
                "id": row["id"],
                "type": row["type"],
                "title": row["title"],
                "summary": row["summary"],
                "tags": json.loads(row["tags_json"]),
                "sensitivity": row["sensitivity"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "verified": verification["verified"],
                "stale": verification["stale"],
                "ttl_days": verification["ttl_days"],
                "valid_citations": verification["valid_citations"],
                "required_citations": verification["required_citations"],
            }
        )
    return {"status": "ok", "entries": result}


@app.post("/pipeline/audio/transcribe")
def pipeline_audio_transcribe(req: AudioTranscribeRequest) -> dict:
    try:
        return _transcribe_audio_fixture(req)
    except MemoryGovernanceError as exc:
        raise HTTPException(
            status_code=400,
            detail=_memory_error(exc.code, exc.reason, exc.suggestion, exc.retryable),
        ) from exc


@app.get("/tools/transcript/recent")
def transcript_recent(limit: int = Query(default=20, ge=1, le=100)) -> dict:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT audio_id, model, created_at, path, SUBSTR(text, 1, 160) AS preview
            FROM transcript
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return {"status": "ok", "entries": [dict(row) for row in rows]}


@app.post("/tools/report_generate")
def report_generate(req: ReportGenerateRequest) -> dict:
    return _generate_report(req)


@app.get("/tools/report_recent")
def report_recent(limit: int = Query(default=20, ge=1, le=100)) -> dict:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT report_date, path, created_at, generator, note
            FROM report_artifact
            ORDER BY report_date DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return {"status": "ok", "entries": [dict(row) for row in rows]}


@app.post("/analysis/trends")
def analysis_trends(req: TrendAnalysisRequest) -> dict:
    try:
        return _execute_trend_analysis(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"trend analysis failed: {exc}") from exc


@app.post("/analyze")
def analyze(req: AnalyzeRequest) -> dict:
    if not _model_available(OLLAMA_MODEL):
        raise HTTPException(status_code=503, detail=f"required model missing: {OLLAMA_MODEL}")
    response_text = _ollama_generate(req.prompt)
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO agent_run_log(created_at, prompt, model, response) VALUES(?, ?, ?, ?)",
            (now, req.prompt, OLLAMA_MODEL, response_text),
        )
        conn.commit()
    return {"model": OLLAMA_MODEL, "response": response_text}
