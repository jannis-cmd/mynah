import json
import os
import re
import sqlite3
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
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
