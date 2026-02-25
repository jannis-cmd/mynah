
import json
import os
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import psycopg
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, ValidationError, field_validator
from psycopg.rows import dict_row

SERVICE = os.getenv("MYNAH_SERVICE_NAME", "mynah_agent")
DATABASE_DSN = os.getenv("MYNAH_DATABASE_DSN", "postgresql://mynah:mynah@postgres:5432/mynah")
ARTIFACTS_PATH = Path(os.getenv("MYNAH_ARTIFACTS_PATH", "/home/appuser/data/artifacts"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_TIMEOUT_SEC = int(os.getenv("OLLAMA_TIMEOUT_SEC", "120"))
MAX_WRITEPLAN_RETRIES = int(os.getenv("MYNAH_WRITEPLAN_MAX_RETRIES", "3"))
SIMILARITY_TOP_K = int(os.getenv("MYNAH_SIMILARITY_TOP_K", "12"))

SENSITIVE_TYPES = {"health", "decision"}

app = FastAPI(title="mynah_agent", version="0.2.0")


class ArtifactIngestRequest(BaseModel):
    artifact_type: Literal["transcript", "me_md", "doc", "agent_note", "voice_transcript"]
    source: str = Field(default="manual", min_length=1, max_length=64)
    content_text: str = Field(min_length=1, max_length=200_000)
    object_uri: str | None = Field(default=None, max_length=512)
    metadata: dict[str, Any] = Field(default_factory=dict)
    caller: str = Field(default="local_ui", min_length=1, max_length=64)


class ArtifactProcessRequest(BaseModel):
    caller: str = Field(default="local_ui", min_length=1, max_length=64)


class MeMarkdownProcessRequest(BaseModel):
    markdown: str = Field(min_length=1, max_length=200_000)
    source: str = Field(default="me_md", min_length=1, max_length=64)
    caller: str = Field(default="local_ui", min_length=1, max_length=64)


class AudioTranscribeRequest(BaseModel):
    audio_id: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9._:-]+$")
    caller: str = Field(default="local_ui", min_length=1, max_length=64)
    force: bool = Field(default=False)


class ReportGenerateRequest(BaseModel):
    date: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    caller: str = Field(default="local_ui", min_length=1, max_length=64)


class WriteEntryPlan(BaseModel):
    action: Literal["create", "update", "link", "duplicate", "question"]
    entry_type: Literal["memory", "health", "preference", "idea", "decision", "relationship", "event", "task"] | None = None
    title: str | None = Field(default=None, max_length=200)
    body: str | None = Field(default=None, max_length=4000)
    tags: list[str] = Field(default_factory=list, max_length=32)
    time_start: str | None = Field(default=None, max_length=64)
    time_end: str | None = Field(default=None, max_length=64)
    target_entry_id: str | None = Field(default=None, max_length=128)
    relation: str | None = Field(default=None, max_length=64)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("tags")
    @classmethod
    def normalize_tags(cls, value: list[str]) -> list[str]:
        seen = set()
        out = []
        for tag in value:
            t = tag.strip().lower()
            if not t or t in seen:
                continue
            seen.add(t)
            out.append(t)
        return out


class WriteFactPlan(BaseModel):
    action: Literal["create", "update", "duplicate", "question"]
    fact_type: str = Field(min_length=1, max_length=64)
    payload: dict[str, Any] = Field(default_factory=dict)
    effective_time: str | None = Field(default=None, max_length=64)
    source_entry_ref: str | None = Field(default=None, max_length=128)
    target_fact_id: str | None = Field(default=None, max_length=128)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class WriteLinkPlan(BaseModel):
    from_ref: str = Field(min_length=1, max_length=128)
    to_ref: str = Field(min_length=1, max_length=128)
    relation: str = Field(min_length=1, max_length=64)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class DedupeCandidate(BaseModel):
    candidate_existing_id: str = Field(min_length=1, max_length=128)
    reason: str = Field(min_length=1, max_length=300)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class WritePlan(BaseModel):
    artifact_summary: str = Field(default="", max_length=2000)
    new_entries: list[WriteEntryPlan] = Field(default_factory=list, max_length=64)
    new_facts: list[WriteFactPlan] = Field(default_factory=list, max_length=64)
    links: list[WriteLinkPlan] = Field(default_factory=list, max_length=128)
    dedupe_candidates: list[DedupeCandidate] = Field(default_factory=list, max_length=64)
    questions: list[str] = Field(default_factory=list, max_length=64)


class ValidationIssue(BaseModel):
    code: str
    field: str
    reason: str
    suggestion: str


def _db_conn() -> psycopg.Connection:
    return psycopg.connect(DATABASE_DSN, autocommit=False)


def _init_db() -> None:
    ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)
    with _db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
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
                    ingested_at TIMESTAMPTZ NOT NULL
                )
                """
            )
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
                CREATE TABLE IF NOT EXISTS entries (
                    id TEXT PRIMARY KEY,
                    entry_type TEXT NOT NULL,
                    title TEXT,
                    body TEXT NOT NULL,
                    time_start TIMESTAMPTZ,
                    time_end TIMESTAMPTZ,
                    confidence REAL NOT NULL,
                    artifact_id TEXT,
                    created_by TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                    embedding VECTOR,
                    created_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS entry_version (
                    id BIGSERIAL PRIMARY KEY,
                    entry_id TEXT NOT NULL,
                    version_index INTEGER NOT NULL,
                    snapshot_json JSONB NOT NULL,
                    reason TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS facts (
                    id TEXT PRIMARY KEY,
                    fact_type TEXT NOT NULL,
                    payload_json JSONB NOT NULL,
                    effective_time TIMESTAMPTZ,
                    source_entry_id TEXT,
                    artifact_id TEXT,
                    confidence REAL NOT NULL,
                    status TEXT NOT NULL DEFAULT 'candidate',
                    created_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS links (
                    id BIGSERIAL PRIMARY KEY,
                    from_id TEXT NOT NULL,
                    to_id TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    artifact_id TEXT,
                    created_at TIMESTAMPTZ NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS write_plan_audit (
                    id BIGSERIAL PRIMARY KEY,
                    artifact_id TEXT NOT NULL,
                    attempt INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    caller TEXT NOT NULL,
                    model TEXT NOT NULL,
                    prompt_text TEXT NOT NULL,
                    llm_output_text TEXT,
                    plan_json JSONB,
                    validation_errors_json JSONB,
                    created_at TIMESTAMPTZ NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS report_artifact (
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


def _ollama_embed(text: str) -> list[float] | None:
    payloads = [
        {"model": OLLAMA_MODEL, "input": text},
        {"model": OLLAMA_MODEL, "prompt": text},
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
                if "embedding" in data and isinstance(data["embedding"], list):
                    return [float(v) for v in data["embedding"]]
                if "embeddings" in data and isinstance(data["embeddings"], list) and data["embeddings"]:
                    first = data["embeddings"][0]
                    if isinstance(first, list):
                        return [float(v) for v in first]
            except Exception:  # noqa: BLE001
                continue
    return None


def _vector_literal(embedding: list[float]) -> str:
    return "[" + ",".join(f"{value:.10f}" for value in embedding) + "]"


def _extract_json_object(text: str) -> dict[str, Any]:
    if not text:
        raise ValueError("empty model output")
    stripped = text.strip()
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if not match:
        raise ValueError("no json object in model output")
    obj = json.loads(match.group(0))
    if not isinstance(obj, dict):
        raise ValueError("model json root must be object")
    return obj


def _validate_write_plan(plan: WritePlan) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for idx, entry in enumerate(plan.new_entries):
        field = f"new_entries[{idx}]"
        if entry.action in {"create", "update"} and not entry.entry_type:
            issues.append(
                ValidationIssue(
                    code="ENTRY_TYPE_REQUIRED",
                    field=field,
                    reason="entry_type is required for create/update",
                    suggestion="set entry_type to allowed taxonomy value",
                )
            )
        if entry.action in {"create", "update"} and not (entry.body and entry.body.strip()):
            issues.append(
                ValidationIssue(
                    code="ENTRY_BODY_REQUIRED",
                    field=field,
                    reason="body is required for create/update",
                    suggestion="provide concise body text",
                )
            )
        if entry.action in {"update", "link", "duplicate"} and not entry.target_entry_id:
            issues.append(
                ValidationIssue(
                    code="ENTRY_TARGET_REQUIRED",
                    field=field,
                    reason="target_entry_id is required for update/link/duplicate",
                    suggestion="set target_entry_id to an existing entry id",
                )
            )
        if entry.action == "link" and not entry.relation:
            issues.append(
                ValidationIssue(
                    code="ENTRY_RELATION_REQUIRED",
                    field=field,
                    reason="relation is required for link action",
                    suggestion="set relation such as about/supports/refers_to",
                )
            )
    for idx, fact in enumerate(plan.new_facts):
        field = f"new_facts[{idx}]"
        if fact.action in {"create", "update"} and not fact.payload:
            issues.append(
                ValidationIssue(
                    code="FACT_PAYLOAD_REQUIRED",
                    field=field,
                    reason="payload must be non-empty for create/update",
                    suggestion="provide structured payload fields",
                )
            )
        if fact.action in {"update", "duplicate"} and not fact.target_fact_id:
            issues.append(
                ValidationIssue(
                    code="FACT_TARGET_REQUIRED",
                    field=field,
                    reason="target_fact_id required for update/duplicate",
                    suggestion="set target_fact_id to an existing fact id",
                )
            )
    for idx, link in enumerate(plan.links):
        field = f"links[{idx}]"
        if link.from_ref == link.to_ref:
            issues.append(
                ValidationIssue(
                    code="LINK_SELF_REFERENCE",
                    field=field,
                    reason="from_ref and to_ref cannot be identical",
                    suggestion="link two different entries",
                )
            )
    return issues


def _insert_artifact(
    *,
    artifact_type: str,
    source: str,
    content_text: str,
    object_uri: str | None,
    metadata: dict[str, Any],
) -> str:
    artifact_id = str(uuid4())
    now = datetime.now(timezone.utc)
    with _db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO artifacts(
                    id, artifact_type, source, created_at, content_text, content_hash,
                    object_uri, metadata_json, processing_state
                )
                VALUES(%s, %s, %s, %s, %s, %s, %s, %s::jsonb, 'pending')
                """,
                (
                    artifact_id,
                    artifact_type,
                    source,
                    now,
                    content_text,
                    sha256(content_text.encode("utf-8")).hexdigest(),
                    object_uri,
                    json.dumps(metadata, sort_keys=True),
                ),
            )
        conn.commit()
    return artifact_id


def _fetch_artifact(artifact_id: str) -> dict[str, Any] | None:
    with _db_conn() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT id, artifact_type, source, created_at, content_text, object_uri, metadata_json
                FROM artifacts
                WHERE id = %s
                LIMIT 1
                """,
                (artifact_id,),
            )
            return cur.fetchone()


def _find_similar_entries(content_text: str, limit: int) -> tuple[list[dict[str, Any]], list[float] | None]:
    embedding = _ollama_embed(content_text)
    if not embedding:
        return [], None
    vector = _vector_literal(embedding)
    with _db_conn() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT id, entry_type, title, body, confidence, created_at,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM entries
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (vector, vector, limit),
            )
            rows = cur.fetchall()
    return rows, embedding

def _build_prompt(artifact: dict[str, Any], similar_entries: list[dict[str, Any]], issues: list[ValidationIssue]) -> str:
    similar_payload = []
    for row in similar_entries:
        similar_payload.append(
            {
                "id": row["id"],
                "entry_type": row["entry_type"],
                "title": row["title"],
                "body": (row["body"] or "")[:300],
                "similarity": row.get("similarity"),
            }
        )
    prompt = (
        "You are MYNAH's structured writer. Return ONLY one JSON object.\n"
        "Allowed entry_type taxonomy: memory, health, preference, idea, decision, relationship, event, task.\n"
        "Actions: create, update, link, duplicate, question.\n"
        "Use existing ids only from similar entries; temporary refs only as new_entries[i].\n"
        "Sensitive types health/decision should use conservative confidence if uncertain.\n"
        "Required root keys: artifact_summary, new_entries, new_facts, links, dedupe_candidates, questions.\n\n"
        f"Artifact metadata: {json.dumps({'id': artifact['id'], 'artifact_type': artifact['artifact_type'], 'source': artifact['source']}, sort_keys=True)}\n"
        f"Artifact text:\n{artifact['content_text'][:12000]}\n\n"
        f"Similar entries: {json.dumps(similar_payload, sort_keys=True)}\n"
    )
    if issues:
        issues_json = json.dumps([i.model_dump() for i in issues], sort_keys=True)
        prompt += f"\nPrevious validation errors to fix exactly: {issues_json}\n"
    prompt += (
        "\nJSON example:\n"
        "{\n"
        "  \"artifact_summary\": \"...\",\n"
        "  \"new_entries\": [{\"action\":\"create\",\"entry_type\":\"memory\",\"title\":\"...\",\"body\":\"...\",\"confidence\":0.7}],\n"
        "  \"new_facts\": [],\n"
        "  \"links\": [],\n"
        "  \"dedupe_candidates\": [],\n"
        "  \"questions\": []\n"
        "}\n"
    )
    return prompt


def _audit_attempt(
    *,
    artifact_id: str,
    attempt: int,
    status: str,
    caller: str,
    prompt_text: str,
    llm_output_text: str | None,
    plan_json: dict[str, Any] | None,
    validation_errors: list[dict[str, Any]] | None,
) -> None:
    with _db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO write_plan_audit(
                    artifact_id, attempt, status, caller, model, prompt_text, llm_output_text,
                    plan_json, validation_errors_json, created_at
                )
                VALUES(%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s)
                """,
                (
                    artifact_id,
                    attempt,
                    status,
                    caller,
                    OLLAMA_MODEL,
                    prompt_text,
                    llm_output_text,
                    json.dumps(plan_json, sort_keys=True) if plan_json is not None else None,
                    json.dumps(validation_errors, sort_keys=True) if validation_errors is not None else None,
                    datetime.now(timezone.utc),
                ),
            )
        conn.commit()


def _resolve_entry_ref(ref: str, created_entries: list[str]) -> str | None:
    match = re.match(r"^new_entries\[(\d+)\]$", ref.strip())
    if not match:
        return ref.strip()
    idx = int(match.group(1))
    if 0 <= idx < len(created_entries):
        return created_entries[idx]
    return None


def _persist_unprocessed_candidate(artifact: dict[str, Any], caller: str, reason: str) -> str:
    entry_id = str(uuid4())
    now = datetime.now(timezone.utc)
    with _db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO entries(
                    id, entry_type, title, body, confidence, artifact_id,
                    created_by, status, metadata_json, embedding, created_at, updated_at
                )
                VALUES(%s, 'unprocessed_candidate', %s, %s, %s, %s, %s, 'candidate', %s::jsonb, NULL, %s, %s)
                """,
                (
                    entry_id,
                    "Unprocessed Artifact",
                    artifact["content_text"][:4000],
                    0.0,
                    artifact["id"],
                    caller,
                    json.dumps({"reason": reason}, sort_keys=True),
                    now,
                    now,
                ),
            )
            cur.execute(
                "UPDATE artifacts SET processing_state = 'failed', processed_at = %s WHERE id = %s",
                (now, artifact["id"]),
            )
        conn.commit()
    return entry_id


def _apply_plan(artifact: dict[str, Any], plan: WritePlan, caller: str, artifact_embedding: list[float] | None) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    created_entries: list[str] = []
    updated_entries: list[str] = []
    created_facts: list[str] = []
    created_links = 0

    with _db_conn() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            for entry in plan.new_entries:
                if entry.action == "create":
                    entry_id = str(uuid4())
                    body = (entry.body or "").strip()
                    embedding = _ollama_embed(body) if body else None
                    status = "candidate" if (entry.entry_type in SENSITIVE_TYPES and entry.confidence < 0.80) else "active"
                    cur.execute(
                        """
                        INSERT INTO entries(
                            id, entry_type, title, body, time_start, time_end, confidence, artifact_id,
                            created_by, status, metadata_json, embedding, created_at, updated_at
                        )
                        VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::vector, %s, %s)
                        """,
                        (
                            entry_id,
                            entry.entry_type,
                            (entry.title or "").strip() or None,
                            body,
                            entry.time_start,
                            entry.time_end,
                            entry.confidence,
                            artifact["id"],
                            caller,
                            status,
                            json.dumps({"tags": entry.tags}, sort_keys=True),
                            _vector_literal(embedding) if embedding else None,
                            now,
                            now,
                        ),
                    )
                    created_entries.append(entry_id)
                elif entry.action == "update" and entry.target_entry_id:
                    cur.execute("SELECT * FROM entries WHERE id = %s LIMIT 1", (entry.target_entry_id,))
                    existing = cur.fetchone()
                    if not existing:
                        continue
                    cur.execute("SELECT COALESCE(MAX(version_index), 0) + 1 FROM entry_version WHERE entry_id = %s", (entry.target_entry_id,))
                    next_version = int(cur.fetchone()[0])
                    cur.execute(
                        """
                        INSERT INTO entry_version(entry_id, version_index, snapshot_json, reason, created_at)
                        VALUES(%s, %s, %s::jsonb, %s, %s)
                        """,
                        (
                            entry.target_entry_id,
                            next_version,
                            json.dumps(dict(existing), sort_keys=True, default=str),
                            "write_plan_update",
                            now,
                        ),
                    )
                    new_body = (entry.body or existing["body"]).strip()
                    embedding = _ollama_embed(new_body) if new_body else None
                    new_type = entry.entry_type or existing["entry_type"]
                    new_status = "candidate" if (new_type in SENSITIVE_TYPES and entry.confidence < 0.80) else existing["status"]
                    cur.execute(
                        """
                        UPDATE entries
                        SET entry_type = %s,
                            title = %s,
                            body = %s,
                            time_start = COALESCE(%s, time_start),
                            time_end = COALESCE(%s, time_end),
                            confidence = %s,
                            status = %s,
                            metadata_json = metadata_json || %s::jsonb,
                            embedding = COALESCE(%s::vector, embedding),
                            updated_at = %s
                        WHERE id = %s
                        """,
                        (
                            new_type,
                            (entry.title or existing["title"] or "").strip() or None,
                            new_body,
                            entry.time_start,
                            entry.time_end,
                            entry.confidence,
                            new_status,
                            json.dumps({"tags": entry.tags}, sort_keys=True),
                            _vector_literal(embedding) if embedding else None,
                            now,
                            entry.target_entry_id,
                        ),
                    )
                    updated_entries.append(entry.target_entry_id)
                elif entry.action == "link" and entry.target_entry_id and created_entries:
                    cur.execute(
                        """
                        INSERT INTO links(from_id, to_id, relation, confidence, artifact_id, created_at)
                        VALUES(%s, %s, %s, %s, %s, %s)
                        """,
                        (created_entries[-1], entry.target_entry_id, entry.relation or "related", entry.confidence, artifact["id"], now),
                    )
                    created_links += 1

            for fact in plan.new_facts:
                if fact.action not in {"create", "update"}:
                    continue
                fact_id = fact.target_fact_id if (fact.action == "update" and fact.target_fact_id) else str(uuid4())
                source_entry_id = _resolve_entry_ref(fact.source_entry_ref, created_entries) if fact.source_entry_ref else None
                status = "confirmed" if fact.confidence >= 0.85 else "candidate"
                cur.execute(
                    """
                    INSERT INTO facts(
                        id, fact_type, payload_json, effective_time, source_entry_id, artifact_id,
                        confidence, status, created_at, updated_at
                    )
                    VALUES(%s, %s, %s::jsonb, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT(id) DO UPDATE SET
                        fact_type = excluded.fact_type,
                        payload_json = excluded.payload_json,
                        effective_time = excluded.effective_time,
                        source_entry_id = excluded.source_entry_id,
                        artifact_id = excluded.artifact_id,
                        confidence = excluded.confidence,
                        status = excluded.status,
                        updated_at = excluded.updated_at
                    """,
                    (
                        fact_id,
                        fact.fact_type,
                        json.dumps(fact.payload, sort_keys=True),
                        fact.effective_time,
                        source_entry_id,
                        artifact["id"],
                        fact.confidence,
                        status,
                        now,
                        now,
                    ),
                )
                created_facts.append(fact_id)

            for link in plan.links:
                from_id = _resolve_entry_ref(link.from_ref, created_entries)
                to_id = _resolve_entry_ref(link.to_ref, created_entries)
                if not from_id or not to_id:
                    continue
                cur.execute(
                    """
                    INSERT INTO links(from_id, to_id, relation, confidence, artifact_id, created_at)
                    VALUES(%s, %s, %s, %s, %s, %s)
                    """,
                    (from_id, to_id, link.relation, link.confidence, artifact["id"], now),
                )
                created_links += 1

            cur.execute("UPDATE artifacts SET processing_state = 'processed', processed_at = %s WHERE id = %s", (now, artifact["id"]))
        conn.commit()

    return {
        "created_entries": len(created_entries),
        "updated_entries": len(updated_entries),
        "created_facts": len(created_facts),
        "created_links": created_links,
        "entry_ids": created_entries,
        "updated_entry_ids": updated_entries,
        "fact_ids": created_facts,
        "artifact_embedding_created": bool(artifact_embedding),
    }


def _process_artifact(artifact_id: str, caller: str) -> dict[str, Any]:
    artifact = _fetch_artifact(artifact_id)
    if not artifact:
        raise HTTPException(status_code=404, detail="artifact not found")

    similar_entries, artifact_embedding = _find_similar_entries(artifact["content_text"], SIMILARITY_TOP_K)
    issues: list[ValidationIssue] = []
    for attempt in range(1, MAX_WRITEPLAN_RETRIES + 1):
        prompt = _build_prompt(artifact, similar_entries, issues)
        llm_output = _ollama_generate(prompt)
        plan_dict: dict[str, Any] | None = None
        validation_issues: list[ValidationIssue] = []
        try:
            plan_dict = _extract_json_object(llm_output)
            plan = WritePlan(**plan_dict)
            validation_issues = _validate_write_plan(plan)
        except (ValueError, ValidationError) as exc:
            validation_issues = [
                ValidationIssue(
                    code="WRITE_PLAN_PARSE_FAILED",
                    field="root",
                    reason=str(exc),
                    suggestion="return strict JSON object matching required root keys",
                )
            ]

        _audit_attempt(
            artifact_id=artifact_id,
            attempt=attempt,
            status="valid" if not validation_issues else "invalid",
            caller=caller,
            prompt_text=prompt,
            llm_output_text=llm_output,
            plan_json=plan_dict,
            validation_errors=[item.model_dump() for item in validation_issues],
        )

        if not validation_issues:
            result = _apply_plan(artifact, plan, caller, artifact_embedding)
            return {
                "status": "ok",
                "artifact_id": artifact_id,
                "attempts": attempt,
                "write_result": result,
                "artifact_summary": plan.artifact_summary,
                "questions": plan.questions,
            }
        issues = validation_issues

    fallback_id = _persist_unprocessed_candidate(artifact, caller, "validator_retries_exhausted")
    return {
        "status": "failed_closed",
        "artifact_id": artifact_id,
        "attempts": MAX_WRITEPLAN_RETRIES,
        "fallback_entry_id": fallback_id,
        "validation_errors": [item.model_dump() for item in issues],
    }


def _transcribe_audio_fixture(req: AudioTranscribeRequest) -> dict:
    with _db_conn() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT id, transcript_hint_path
                FROM audio_note
                WHERE id = %s
                LIMIT 1
                """,
                (req.audio_id,),
            )
            audio = cur.fetchone()
            if not audio:
                raise HTTPException(status_code=404, detail="audio note not found")

            cur.execute("SELECT audio_id, path FROM transcript WHERE audio_id = %s LIMIT 1", (req.audio_id,))
            existing = cur.fetchone()

    if existing and not req.force:
        artifact_id = _insert_artifact(
            artifact_type="voice_transcript",
            source="audio_transcribe_cached",
            content_text="cached transcript reused",
            object_uri=existing["path"],
            metadata={"audio_id": req.audio_id, "cached": True},
        )
        return {"status": "ok", "audio_id": req.audio_id, "transcript_created": False, "artifact_id": artifact_id, "processed": False}

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
                INSERT INTO transcript(audio_id, text, model, path, created_at)
                VALUES(%s, %s, %s, %s, %s)
                ON CONFLICT(audio_id) DO UPDATE SET
                    text=excluded.text,
                    model=excluded.model,
                    path=excluded.path,
                    created_at=excluded.created_at
                """,
                (req.audio_id, transcript_text, "artifact_transcript_v1", str(transcript_path), now),
            )
            cur.execute("UPDATE audio_note SET transcription_state = 'completed' WHERE id = %s", (req.audio_id,))
        conn.commit()

    artifact_id = _insert_artifact(
        artifact_type="voice_transcript",
        source="audio_transcribe",
        content_text=transcript_text,
        object_uri=str(transcript_path),
        metadata={"audio_id": req.audio_id},
    )
    process_result = _process_artifact(artifact_id, req.caller)
    return {
        "status": "ok",
        "audio_id": req.audio_id,
        "transcript_created": True,
        "artifact_id": artifact_id,
        "processed": process_result["status"],
        "write_result": process_result.get("write_result"),
    }


def _generate_report(req: ReportGenerateRequest) -> dict:
    report_date = req.date or datetime.now(timezone.utc).date().isoformat()
    start = datetime.fromisoformat(f"{report_date}T00:00:00+00:00")
    end = datetime.fromisoformat(f"{report_date}T23:59:59.999999+00:00")
    with _db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM entries WHERE created_at >= %s AND created_at <= %s", (start, end))
            entry_count = int(cur.fetchone()[0])
            cur.execute("SELECT COUNT(*) FROM facts WHERE created_at >= %s AND created_at <= %s", (start, end))
            fact_count = int(cur.fetchone()[0])
            cur.execute(
                """
                SELECT id, entry_type, COALESCE(title, '(untitled)')
                FROM entries
                WHERE created_at >= %s AND created_at <= %s
                ORDER BY created_at DESC
                LIMIT 10
                """,
                (start, end),
            )
            rows = cur.fetchall()

    lines = [
        f"# MYNAH Daily Report - {report_date}",
        "",
        "## Structured Writes",
        f"- Entries written: {entry_count}",
        f"- Facts written: {fact_count}",
        "",
        "## Latest Entries",
    ]
    for row in rows:
        lines.append(f"- `{row[0]}` [{row[1]}] {row[2]}")

    report_dir = ARTIFACTS_PATH / "reports" / report_date
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    now = datetime.now(timezone.utc)
    with _db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO report_artifact(report_date, path, created_at, generator, note)
                VALUES(%s, %s, %s, %s, %s)
                ON CONFLICT(report_date) DO UPDATE SET
                    path=excluded.path,
                    created_at=excluded.created_at,
                    generator=excluded.generator,
                    note=excluded.note
                """,
                (report_date, str(report_path), now, SERVICE, f"entries={entry_count};facts={fact_count}"),
            )
        conn.commit()
    return {"status": "ok", "report_date": report_date, "path": str(report_path), "entries": entry_count, "facts": fact_count}


@app.on_event("startup")
def startup() -> None:
    _init_db()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": SERVICE}


@app.get("/ready")
def ready() -> dict:
    try:
        with _db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"postgres unavailable: {exc}") from exc
    try:
        model_ok = _model_available(OLLAMA_MODEL)
    except urllib.error.URLError as exc:
        raise HTTPException(status_code=503, detail=f"ollama unavailable: {exc.reason}") from exc
    if not model_ok:
        raise HTTPException(status_code=503, detail=f"required model missing: {OLLAMA_MODEL}")
    return {"status": "ready", "service": SERVICE, "database": "postgres", "model": OLLAMA_MODEL}


@app.post("/pipeline/artifacts/ingest")
def pipeline_artifacts_ingest(req: ArtifactIngestRequest) -> dict:
    artifact_id = _insert_artifact(
        artifact_type=req.artifact_type,
        source=req.source,
        content_text=req.content_text,
        object_uri=req.object_uri,
        metadata={"caller": req.caller, **req.metadata},
    )
    return {"status": "ok", "artifact_id": artifact_id}


@app.post("/pipeline/artifacts/process/{artifact_id}")
def pipeline_artifacts_process(artifact_id: str, req: ArtifactProcessRequest) -> dict:
    return _process_artifact(artifact_id, req.caller)


@app.post("/pipeline/me_md/process")
def pipeline_me_md_process(req: MeMarkdownProcessRequest) -> dict:
    artifact_id = _insert_artifact(
        artifact_type="me_md",
        source=req.source,
        content_text=req.markdown,
        object_uri=None,
        metadata={"caller": req.caller},
    )
    result = _process_artifact(artifact_id, req.caller)
    return {"status": "ok", "artifact_id": artifact_id, "result": result}


@app.post("/pipeline/audio/transcribe")
def pipeline_audio_transcribe(req: AudioTranscribeRequest) -> dict:
    return _transcribe_audio_fixture(req)


@app.get("/tools/transcript/recent")
def transcript_recent(limit: int = Query(default=20, ge=1, le=100)) -> dict:
    with _db_conn() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT audio_id, model, created_at, path, SUBSTRING(text FROM 1 FOR 160) AS preview
                FROM transcript
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()
    return {"status": "ok", "entries": rows}


@app.post("/tools/report_generate")
def tools_report_generate(req: ReportGenerateRequest) -> dict:
    return _generate_report(req)


@app.get("/tools/report_recent")
def tools_report_recent(limit: int = Query(default=20, ge=1, le=100)) -> dict:
    with _db_conn() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT report_date, path, created_at, generator, note
                FROM report_artifact
                ORDER BY report_date DESC
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
            cur.execute("SELECT COUNT(*) FROM artifacts")
            artifact_count = int(cur.fetchone()[0])
            cur.execute("SELECT COUNT(*) FROM entries")
            entry_count = int(cur.fetchone()[0])
            cur.execute("SELECT COUNT(*) FROM facts")
            fact_count = int(cur.fetchone()[0])
            cur.execute("SELECT COUNT(*) FROM write_plan_audit")
            audit_count = int(cur.fetchone()[0])
    return {
        "service": SERVICE,
        "database": "postgres",
        "artifact_count": artifact_count,
        "entry_count": entry_count,
        "fact_count": fact_count,
        "write_plan_audit_count": audit_count,
    }
