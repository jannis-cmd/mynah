CREATE EXTENSION IF NOT EXISTS vector;

CREATE SCHEMA IF NOT EXISTS core;
CREATE SCHEMA IF NOT EXISTS health;
CREATE SCHEMA IF NOT EXISTS memory;
CREATE SCHEMA IF NOT EXISTS decision;
CREATE SCHEMA IF NOT EXISTS preference;
CREATE SCHEMA IF NOT EXISTS search;

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
    extractor_version TEXT NOT NULL DEFAULT 'v1',
    extraction_schema_version TEXT NOT NULL DEFAULT 'v1',
    reprocess_of_artifact_id TEXT,
    created_at TIMESTAMPTZ NOT NULL,
    processed_at TIMESTAMPTZ
);

ALTER TABLE core.ingest_artifact ADD COLUMN IF NOT EXISTS extractor_version TEXT NOT NULL DEFAULT 'v1';
ALTER TABLE core.ingest_artifact ADD COLUMN IF NOT EXISTS extraction_schema_version TEXT NOT NULL DEFAULT 'v1';
ALTER TABLE core.ingest_artifact ADD COLUMN IF NOT EXISTS reprocess_of_artifact_id TEXT;

CREATE INDEX IF NOT EXISTS idx_ingest_artifact_state ON core.ingest_artifact(processing_state);
CREATE INDEX IF NOT EXISTS idx_ingest_artifact_hash ON core.ingest_artifact(content_hash);
CREATE INDEX IF NOT EXISTS idx_ingest_artifact_source_ts ON core.ingest_artifact(source_ts);

CREATE TABLE IF NOT EXISTS core.artifact_meta (
    artifact_id TEXT NOT NULL REFERENCES core.ingest_artifact(id) ON DELETE CASCADE,
    meta_key TEXT NOT NULL,
    value_json JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (artifact_id, meta_key)
);

CREATE TABLE IF NOT EXISTS core.compaction_attempt (
    id BIGSERIAL PRIMARY KEY,
    artifact_id TEXT NOT NULL REFERENCES core.ingest_artifact(id) ON DELETE CASCADE,
    attempt INTEGER NOT NULL,
    status TEXT NOT NULL,
    caller TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_text TEXT NOT NULL,
    output_text TEXT,
    error_text TEXT,
    created_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS core.extraction_failure (
    id BIGSERIAL PRIMARY KEY,
    artifact_id TEXT NOT NULL REFERENCES core.ingest_artifact(id) ON DELETE CASCADE,
    stage TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    model TEXT,
    output_text TEXT,
    error_code TEXT,
    error_text TEXT NOT NULL,
    retriable BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_extraction_failure_artifact ON core.extraction_failure(artifact_id, created_at DESC);

CREATE TABLE IF NOT EXISTS core.open_question (
    id BIGSERIAL PRIMARY KEY,
    artifact_id TEXT NOT NULL REFERENCES core.ingest_artifact(id) ON DELETE CASCADE,
    ts TIMESTAMPTZ NOT NULL,
    question_text TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'candidate' CHECK (status IN ('candidate','open','resolved','dropped')),
    resolved_at TIMESTAMPTZ,
    resolution_note TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
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

CREATE TABLE IF NOT EXISTS health.metric_def (
    metric TEXT PRIMARY KEY,
    unit TEXT,
    kind TEXT NOT NULL CHECK (kind IN ('numeric','json','event')),
    expected_min DOUBLE PRECISION,
    expected_max DOUBLE PRECISION,
    default_aggregation TEXT NOT NULL DEFAULT 'avg' CHECK (default_aggregation IN ('avg','sum','min','max','count','latest')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

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
    note_type TEXT NOT NULL DEFAULT 'observation' CHECK (note_type IN ('event','fact','observation','feeling','decision_context','task')),
    text TEXT NOT NULL,
    embedding VECTOR(1024) NOT NULL,
    source_artifact_id TEXT NOT NULL REFERENCES core.ingest_artifact(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
ALTER TABLE memory.note ADD COLUMN IF NOT EXISTS note_type TEXT NOT NULL DEFAULT 'observation';
ALTER TABLE memory.note DROP CONSTRAINT IF EXISTS memory_note_note_type_check;
ALTER TABLE memory.note ADD CONSTRAINT memory_note_note_type_check CHECK (note_type IN ('event','fact','observation','feeling','decision_context','task'));

CREATE INDEX IF NOT EXISTS idx_memory_note_ts ON memory.note(ts);
CREATE INDEX IF NOT EXISTS idx_memory_note_source ON memory.note(source_artifact_id);
CREATE INDEX IF NOT EXISTS idx_memory_note_type ON memory.note(note_type);

CREATE TABLE IF NOT EXISTS memory.health_link (
    id BIGSERIAL PRIMARY KEY,
    memory_id BIGINT NOT NULL REFERENCES memory.note(id) ON DELETE CASCADE,
    health_sample_id BIGINT REFERENCES health.sample(id) ON DELETE CASCADE,
    link_day DATE,
    relation TEXT NOT NULL CHECK (relation IN ('mentions','during','correlates_with')),
    confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    created_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS memory.link (
    id BIGSERIAL PRIMARY KEY,
    src_table TEXT NOT NULL,
    src_id TEXT NOT NULL,
    dst_table TEXT NOT NULL,
    dst_id TEXT NOT NULL,
    relation TEXT NOT NULL,
    confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    artifact_id TEXT REFERENCES core.ingest_artifact(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_memory_link_src ON memory.link(src_table, src_id);
CREATE INDEX IF NOT EXISTS idx_memory_link_dst ON memory.link(dst_table, dst_id);

CREATE TABLE IF NOT EXISTS decision.entry (
    id BIGSERIAL PRIMARY KEY,
    artifact_id TEXT REFERENCES core.ingest_artifact(id) ON DELETE SET NULL,
    ts TIMESTAMPTZ NOT NULL,
    domain TEXT NOT NULL,
    context_text TEXT NOT NULL,
    chosen_action TEXT NOT NULL,
    rationale_text TEXT,
    status TEXT NOT NULL DEFAULT 'candidate' CHECK (status IN ('candidate','active','rejected','archived')),
    source_path TEXT,
    source_commit_hash TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_decision_entry_ts ON decision.entry(ts);
CREATE INDEX IF NOT EXISTS idx_decision_entry_status ON decision.entry(status);

CREATE TABLE IF NOT EXISTS decision.review (
    id BIGSERIAL PRIMARY KEY,
    decision_id BIGINT NOT NULL REFERENCES decision.entry(id) ON DELETE CASCADE,
    artifact_id TEXT REFERENCES core.ingest_artifact(id) ON DELETE SET NULL,
    review_ts TIMESTAMPTZ NOT NULL,
    outcome_score REAL,
    self_review TEXT CHECK (self_review IN ('approve','regret','neutral')),
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (decision_id, review_ts)
);

CREATE TABLE IF NOT EXISTS preference.fact (
    id BIGSERIAL PRIMARY KEY,
    artifact_id TEXT REFERENCES core.ingest_artifact(id) ON DELETE SET NULL,
    pref_key TEXT NOT NULL,
    value_json JSONB NOT NULL,
    domain TEXT NOT NULL,
    strength TEXT NOT NULL DEFAULT 'medium' CHECK (strength IN ('weak','medium','strong')),
    status TEXT NOT NULL DEFAULT 'candidate' CHECK (status IN ('candidate','active','deprecated','retracted','rejected')),
    effective_ts TIMESTAMPTZ NOT NULL,
    source_path TEXT,
    source_commit_hash TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_preference_fact_key_status ON preference.fact(pref_key, status);
CREATE INDEX IF NOT EXISTS idx_preference_fact_domain_status ON preference.fact(domain, status);

CREATE TABLE IF NOT EXISTS core.entity (
    id BIGSERIAL PRIMARY KEY,
    entity_type TEXT NOT NULL CHECK (entity_type IN ('person','place','project','org','topic','device','other')),
    canonical_name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active','archived','merged')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_entity_type_name ON core.entity(entity_type, canonical_name);

CREATE TABLE IF NOT EXISTS core.entity_alias (
    id BIGSERIAL PRIMARY KEY,
    entity_id BIGINT NOT NULL REFERENCES core.entity(id) ON DELETE CASCADE,
    alias TEXT NOT NULL,
    source_artifact_id TEXT REFERENCES core.ingest_artifact(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(entity_id, alias)
);

CREATE TABLE IF NOT EXISTS search.embedding_model (
    id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    dimension INTEGER NOT NULL CHECK (dimension > 0),
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active','deprecated')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS search.vector_index (
    id BIGSERIAL PRIMARY KEY,
    target_table TEXT NOT NULL,
    target_id TEXT NOT NULL,
    chunk_idx INTEGER NOT NULL DEFAULT 0,
    text_content TEXT NOT NULL,
    embedding VECTOR(1024) NOT NULL,
    embedding_model_id TEXT NOT NULL REFERENCES search.embedding_model(id),
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    invalidated_at TIMESTAMPTZ,
    extra_metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(target_table, target_id, chunk_idx, embedding_model_id)
);
CREATE INDEX IF NOT EXISTS idx_vector_index_target ON search.vector_index(target_table, target_id);
CREATE INDEX IF NOT EXISTS idx_vector_index_active ON search.vector_index(is_active);

CREATE TABLE IF NOT EXISTS core.report_artifact (
    report_date TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    generator TEXT NOT NULL,
    note TEXT
);