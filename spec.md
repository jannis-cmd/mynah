# MYNAH Specification

Version: 0.8
Status: Active

## 1. One-Line Definition
MYNAH is an open-source, offline-first personal intelligence system that stores health time-series and personal memory artifacts locally, and uses a deterministic pipeline for extraction, indexing, and analysis.

## 2. Scope

### 2.1 In Scope (Current Target)
- Linux runtime target (development from Linux and Windows).
- Local-only operation (no cloud dependency in core paths).
- Wearable ingest for health signals and voice transcripts.
- Text ingest from external sources (for example chat exports).
- `/ME` git repository as canonical human-owned record for preferences, policies, and curated decisions.
- PostgreSQL + pgvector as runtime query/index layer.
- Local UI for status and report output.
- Explicit user-initiated export via USB.

### 2.2 Out of Scope (v0.x)
- Cloud sync, remote accounts, telemetry.
- Medical diagnosis/regulated claims.
- Multi-user identity and sharing.

## 3. Core Principles
- Keep the data model auditable and replayable.
- LLM proposes; deterministic scripts validate and write.
- `/ME` is canonical for human-governed state (values/policies/preferences/curated decisions).
- SQL is operational truth for retrieval and analytics, derived from artifacts and `/ME`.
- No silent fallbacks.

## 4. Architecture

### 4.1 Health Store
Stores raw time-series measurements keyed by timestamp.

### 4.2 Memory Store
Stores atomic memory notes and links.

### 4.3 Decision and Preference Store
Stores structured decisions, reviews, and preference lifecycle state.

### 4.4 Vector Index
Stores embedding index rows referencing target SQL records.

### 4.5 Canonical `/ME` Repo
Git repo with human-readable, versioned files for long-term ownership and audit.

### 4.6 Unified Pipeline Service
One service (`mynah_agent`) owns ingest, extraction orchestration, validation, writes, and reporting APIs.

## 5. Ownership and Derivation Rules (Locked)
1. Raw artifacts are immutable source records in SQL.
2. `/ME` is canonical for preferences, policies, and curated decisions.
3. SQL rows that mirror `/ME` must keep `source_path + source_commit_hash`.
4. Vector index is disposable/derived; it can be rebuilt from SQL text rows.
5. LLM output never writes directly; scripts validate first.

## 6. Ingest Contract (Required Fields)
Every ingest artifact must include:
- `source_type` (for example `wearable_transcript`, `chat_export`, `manual_text`, `me_repo_sync`)
- `content` (raw text)
- `upload_ts` (`timestamptz`)
- `source_ts` (`timestamptz` or `null`)
- `day_scope` (`boolean`)
- `timezone` (IANA name)
- `extractor_version` (for replay/versioning)
- `extraction_schema_version` (for replay/versioning)

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

### 7.3 Timezone Rule
- Store timestamps in UTC.
- Resolve relative expressions using artifact timezone.

## 8. LLM vs Script Responsibilities (Locked)

LLM responsibilities:
- Group raw text by temporal hint (`groups[].hint`).
- Split group content into atomic memory items (`groups[].items[]`).
- Extract structured candidates:
  - memory notes
  - decision candidates
  - preference candidates
  - unresolved questions
- Use only allowed labels; use safe fallback labels when uncertain.

Script responsibilities:
- Parse explicit timestamps from artifact text.
- Compute artifact anchor timestamp (`exact/day/upload`).
- Resolve each group hint to concrete timestamp deterministically.
- Validate JSON shape and field constraints.
- Enforce candidate-to-canonical lifecycle rules.
- Write SQL rows and embedding index rows.
- Write `/ME` canonical updates and record commit references.
- Quarantine invalid outputs instead of dropping them.

## 9. Candidate Lifecycle (Locked)
All extracted high-impact objects must support lifecycle state:
- `candidate` -> `active` -> `deprecated/retracted` (or `rejected`)

Applies to:
- preference facts
- decisions (pre-approval/approval)
- unresolved questions

## 10. Data Model
Schema source of truth:
- `storage/schema.sql` is the single schema definition.
- Services validate schema presence at startup; they do not create tables.

### 10.1 Core
- `core.ingest_artifact`: immutable artifact anchor + ingestion versions.
- `core.artifact_meta`: typed artifact metadata key/value store.
- `core.compaction_attempt`: extraction attempt audit trail.
- `core.extraction_failure`: quarantine table for invalid outputs.
- `core.open_question`: unresolved question lifecycle.

### 10.2 Health (Use Now + Use Soon)
- `health.sample`: timestamped metric values (`value_num` or `value_json`).
- `health.metric_def`: metric dictionary (`unit`, `kind`, expected range, default aggregation).

### 10.3 Memory
- `memory.note`: atomic notes with `ts`, `ts_mode`, `note_type`, and embedding.
- `memory.health_link`: memory to health links with confidence.
- `memory.link`: generic cross-object links (`memory`, `decision`, `entity`, etc.).

`memory.note.note_type` allowed values:
- `event`, `fact`, `observation`, `feeling`, `decision_context`, `task`

### 10.4 Decision
- `decision.entry`: decision event (context, chosen action, rationale, lifecycle status).
- `decision.review`: later review records (outcome, self-review, notes).

### 10.5 Preference
- `preference.fact`: structured preference rows with lifecycle and `/ME` commit pointers.

### 10.6 Entity (Use Soon)
- `core.entity`: canonical people/places/projects/topics/devices.
- `core.entity_alias`: alias mapping for extracted mentions.

### 10.7 Vector Index
- `search.embedding_model`: embedding model registry.
- `search.vector_index`: derived vector rows referencing SQL targets.

## 11. Link and Relation Semantics
`memory.health_link.relation` allowed values:
- `mentions`, `during`, `correlates_with`

`memory.link.relation` is explicit and typed per link row.
All inferred relations require a confidence score and provenance.

## 12. Idempotency and Replay
- Ingestion must be idempotent by artifact identity (`source_type`, `content_hash`, source timestamps).
- Reprocessing with newer models/schemas is expected and must preserve audit history.
- Existing canonical state must not be overwritten silently.

## 13. Model Strategy (Model-Agnostic Framework)
Framework is model-agnostic via configuration.

### 13.1 Default test models
- Generation model: `qwen2.5:7b`
- Embedding model: `qwen3-embedding:0.6b` with `dimensions=1024`

### 13.2 Separation rule
Separate models are preferred, but the same local model is allowed in constrained mode to minimize setup overhead.

### 13.3 Embedding lifecycle
- `search.vector_index` rows carry model identity and active/invalidation state.
- Re-indexing does not require deleting source text rows.

## 14. Retrieval Contract (Use Soon)
Runtime context assembly order:
1. policy and preference constraints (`/ME`-derived active rows)
2. recent decisions and reviews
3. filtered vector retrieval over relevant domains/time windows
4. linked health context if requested

The retrieval contract is deterministic and script-controlled.

## 15. Retry and Failure Policy
- Validation retry limit: 3 attempts.
- If still invalid after retries: fail closed.
- Failed output is written to `core.extraction_failure`.
- No silent drops.

## 16. Security and Runtime Boundaries
- Containers run least-privilege.
- Internal Docker network only for inter-service traffic.
- Model endpoint is not publicly exposed by default.
- Conversational layer and memory-write pipeline remain separate.
- Runtime readiness is split:
  - core readiness (DB + schema)
  - model readiness (required local models present)

## 17. Testing State
- Core architecture and timestamp contract are implemented.
- Candidate lifecycle, quarantine path, `/ME` sync integrity, and vector lifecycle tests are required next.