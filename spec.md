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

Timestamp resolution is two-step and group-based.

Step A: compute one artifact anchor timestamp:
1. `source_ts` -> `ts_mode = exact`
2. `day_scope = true` -> local 12:00 day anchor -> `ts_mode = day`
3. Script extraction of explicit absolute timestamps from artifact content -> `ts_mode = exact`
4. `upload_ts` fallback -> `ts_mode = upload`

Step B: LLM temporal grouping + deterministic script mapping:
1. LLM returns `groups[]` with:
   - `hint` (from allowed hint options)
   - `items[]` (atomic memory texts)
2. Script resolves each `hint` against `anchor_ts` to one `group_ts`.
3. Every item in that group is written with `ts = group_ts` and mapped `ts_mode`.

This keeps temporal scope consistent across all items in the same group.

### 7.1 Day Scope
If `day_scope = true`, anchor timestamp is local 12:00 of the artifact day (`ts_mode = day`).
Group hints may still override to inferred timestamps.

### 7.2 Hint Mapping Rule
LLM chooses from explicit hint options; script performs all timestamp arithmetic.
Minimum supported hints:
- `default`, `today`, `now`
- `morning`, `afternoon`, `evening`, `night`
- `yesterday`, `yesterday morning`, `yesterday afternoon`, `yesterday evening`, `yesterday night`, `last night`
- `tomorrow`, `tomorrow morning`, `tomorrow afternoon`, `tomorrow evening`, `tomorrow night`
- `at H:MM`, `at H am/pm`, `yesterday at H:MM`, `tomorrow at H:MM`

Example mapping from `anchor_ts = 2026-02-25 14:03:05.6`:
- `today` -> `2026-02-25 14:03:05.6`
- `morning` -> `2026-02-25 09:00:00`
- `afternoon` -> `2026-02-25 15:00:00`
- `evening` -> `2026-02-25 20:00:00`
- `yesterday` -> `2026-02-24 12:00:00`
- `tomorrow` -> `2026-02-26 12:00:00`
- `yesterday evening` -> `2026-02-24 20:00:00`

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
- Group raw text by temporal hint (`groups[].hint`).
- Split group content into atomic memory items (`groups[].items[]`).
- Use only allowed hint labels; use `default` when uncertain.

Script responsibilities:
- Parse explicit timestamps from artifact text.
- Compute artifact anchor timestamp (`exact/day/upload`).
- Resolve each group hint to concrete timestamp deterministically.
- Apply group timestamp to every item in that group.
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
- Embedding model: `qwen3-embedding:0.6b` with `dimensions=1024`

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
