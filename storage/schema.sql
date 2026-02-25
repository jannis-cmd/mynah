CREATE EXTENSION IF NOT EXISTS vector;

CREATE SCHEMA IF NOT EXISTS core;
CREATE SCHEMA IF NOT EXISTS health;
CREATE SCHEMA IF NOT EXISTS memory;

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
);
CREATE INDEX IF NOT EXISTS idx_ingest_artifact_state ON core.ingest_artifact(processing_state);

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
);

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
);
CREATE INDEX IF NOT EXISTS idx_health_sample_ts ON health.sample(ts);
CREATE INDEX IF NOT EXISTS idx_health_sample_metric_ts ON health.sample(metric, ts);

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
);
CREATE INDEX IF NOT EXISTS idx_audio_note_start_ts ON core.audio_note(start_ts);

CREATE TABLE IF NOT EXISTS core.transcript (
    audio_id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    model TEXT NOT NULL,
    path TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS memory.note (
    id BIGSERIAL PRIMARY KEY,
    ts TIMESTAMPTZ NOT NULL,
    ts_mode TEXT NOT NULL CHECK (ts_mode IN ('exact','day','inferred','upload')),
    text TEXT NOT NULL,
    embedding VECTOR(1024) NOT NULL,
    source_artifact_id TEXT NOT NULL REFERENCES core.ingest_artifact(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_memory_note_ts ON memory.note(ts);
CREATE INDEX IF NOT EXISTS idx_memory_note_source ON memory.note(source_artifact_id);

CREATE TABLE IF NOT EXISTS memory.health_link (
    id BIGSERIAL PRIMARY KEY,
    memory_id BIGINT NOT NULL REFERENCES memory.note(id) ON DELETE CASCADE,
    health_sample_id BIGINT REFERENCES health.sample(id) ON DELETE CASCADE,
    link_day DATE,
    relation TEXT NOT NULL CHECK (relation IN ('mentions','during','correlates_with')),
    confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    created_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS core.report_artifact (
    report_date TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    generator TEXT NOT NULL,
    note TEXT
);
