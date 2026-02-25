# MYNAH Specification

Version: 0.7
Status: Active

## 1. One-Line Definition
MYNAH is an open-source, offline-first personal intelligence system that stores health time-series and compacted personal memories locally, then links them for on-device analysis.

## 2. Scope

### 2.1 In Scope (Current Target)
- Linux runtime target (development from Linux and Windows).
- Local-only operation (no cloud dependency in core paths).
- Wearable ingest for health signals and voice transcripts.
- Text ingest from external sources (for example chat exports).
- Timestamp-resolved memory compaction from raw text.
- PostgreSQL + pgvector as local datastore.
- Local UI for status and report output.
- Explicit user-initiated export via USB.

### 2.2 Out of Scope (v0.x)
- Cloud sync, remote accounts, telemetry.
- Medical diagnosis/regulated claims.
- Multi-user identity and sharing.

## 3. Core Principles
- Keep the data model minimal and auditable.
- LLM proposes; deterministic pipeline decides and writes.
- Memory and conversation are separate concerns.
- No silent fallbacks.

## 4. Architecture (Simplified)

### 4.1 Health Store
Stores raw time-series measurements keyed by timestamp.

### 4.2 Memory Store
Stores compacted atomic memory notes as plain text plus vector embeddings.

### 4.3 Link Layer
Links memory notes to health data by timestamp alignment rules.

### 4.4 Unified Pipeline Service
One service (`mynah_agent`) owns ingest, memory pipeline, and reporting APIs.

## 5. Data Model (Minimal)
Schema source of truth:
- `storage/schema.sql` is the single schema definition.
- Services validate schema presence at startup; they do not create tables.

### 5.1 `health.sample`
- `id` (PK)
- `ts` (`timestamptz`, required)
- `metric` (`text`, required)  
  examples: `hr`, `hrv`, `sweat`, `insulin`
- `value_num` (`double precision`, nullable)
- `value_json` (`jsonb`, nullable)
- `unit` (`text`, nullable)
- `quality` (`integer`, nullable)
- `source` (`text`, required)
- Uniqueness: `(source, metric, ts)`

### 5.2 `memory.note`
- `id` (PK)
- `ts` (`timestamptz`, required, single timestamp only)
- `ts_mode` (`exact | day | inferred | upload`, required)
- `text` (`text`, required)
- `embedding` (`vector(N)`, required)
- `source_artifact_id` (`text`, required)

Rules:
- No `ts_end`.
- No metadata column on `memory.note`.
- Same memory text at different timestamps is valid and must be stored.

### 5.3 `memory.health_link`
- `id` (PK)
- `memory_id` (FK -> `memory.note`)
- `health_sample_id` (FK -> `health.sample`, nullable)
- `link_day` (`date`, nullable)
- `relation` (`mentions | during | correlates_with`, required)
- `confidence` (`real`, required)
- `created_at` (`timestamptz`, required)

## 6. Ingest Contract (Required Fields)
Every ingest artifact must include all fields below:
- `source_type` (for example `wearable_transcript`, `chat_export`, `manual_text`)
- `content` (raw text)
- `upload_ts` (`timestamptz`)
- `source_ts` (`timestamptz` or `null`)
- `day_scope` (`boolean`)
- `timezone` (IANA name)

`source_ts` is always present as a field; when unavailable it is `null`.

## 7. Timestamp Resolution Framework (Locked)

Priority order:
1. `source_ts` (exact source timestamp)
2. Script extraction of explicit timestamps from content
3. LLM temporal-hint extraction + deterministic script resolver
4. `upload_ts` fallback

### 7.1 Day Scope
If `day_scope = true`, all notes from that artifact use the same day anchor timestamp.
Recommended anchor: local 12:00 for that day.
`ts_mode = day`.

### 7.2 Inference Rule
When content contains relative expressions (for example "in the morning") and no exact time:
- LLM extracts structured hint.
- Script resolves hint against upload day/time and timezone.
- Result stored with `ts_mode = inferred`.

### 7.3 Timezone Rule
- Store timestamps in UTC.
- Resolve relative expressions using artifact timezone.

## 8. Memory Note Definition and Compaction
A memory note is one atomic compacted statement from raw text.

Atomic rules (v0):
- One event/feeling/observation/decision per note.
- Keep original meaning; no invented content.
- Keep notes concise and plain text.
- Do not merge unrelated topics.
- Advanced compaction heuristics are an improvement lever for later versions.

### 8.1 LLM vs Script Responsibilities
LLM responsibilities:
- Compact raw text into atomic notes.
- Extract temporal hints when required.

Script responsibilities:
- Parse explicit timestamps.
- Apply timestamp precedence rules.
- Resolve inferred timestamps deterministically.
- Validate output shape and required fields.
- Write DB rows and embeddings.
- Link memories to health by timestamp rules.

## 9. Memory-to-Health Linking Rules
- `ts_mode = exact`: link by configurable time window around `memory.note.ts`.
- `ts_mode = inferred`: same as exact, with lower confidence default.
- `ts_mode = day`: link by calendar day bucket.
- `ts_mode = upload`: link by fallback time window, lowest default confidence.

## 10. Model Strategy (Model-Agnostic Framework)
Framework is model-agnostic via configuration.

### 10.1 Default test models
- Generation model: `qwen2.5:7b`
- Embedding model: `qwen2.5:7b` with `dimensions=1024`

### 10.2 Separation rule
Separate models are preferred, but the same local model is allowed in constrained mode to minimize setup overhead.

### 10.3 Embedding dimension lock
- Determine embedding size once at initialization.
- Lock `vector(N)` to that size.
- Any future embedding model change requires re-embedding migration.

## 11. Retry and Failure Policy
- Validation retry limit: 3 attempts.
- If still invalid after retries: fail closed and keep artifact for review.
- Retry strategy is an improvement lever for later versions.

## 12. Retention Policy
- Raw artifact and transcript retention: TBD (keep for now).
- Structured memory notes: retained unless user deletes.

## 13. Security and Runtime Boundaries
- Containers run least-privilege.
- Internal Docker network only for inter-service traffic.
- Model endpoint is not publicly exposed by default.
- Conversational layer and memory-write pipeline remain separate.
- Runtime readiness is split:
  - core readiness (DB + schema)
  - model readiness (required local models present)

## 14. Implementation Boundary
- Memory pipeline owns canonical writes.
- Ingest and summary APIs are part of the same service (`mynah_agent`) to keep the stack minimal.

## 15. Testing State
- Full acceptance suite for this simplified architecture is agreed but will be implemented next.
