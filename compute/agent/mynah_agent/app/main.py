import json
import os
import re
import sqlite3
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

SERVICE = os.getenv("MYNAH_SERVICE_NAME", "mynah_agent")
DB_PATH = Path(os.getenv("MYNAH_DB_PATH", "/data/db/mynah.db"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
MAX_QUERY_RUNTIME_MS = 5000
MAX_QUERY_ROWS = 10_000
MAX_QUERY_PAYLOAD_BYTES = 5 * 1024 * 1024
ALLOWED_QUERY_TABLES = {"hr_sample", "device", "query_audit", "service_heartbeat", "agent_run_log"}

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


class SQLValidationError(Exception):
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


def _sql_error(code: str, reason: str, suggestion: str, retryable: bool = False) -> dict:
    return {"code": code, "reason": reason, "retryable": retryable, "suggestion": suggestion}


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
