# MYNAH Specification

Version: 0.9
Status: Active

## 1. One-Line Definition
MYNAH is an open-source, offline-first personal intelligence system that stores health time-series and personal memory artifacts locally, and uses a deterministic pipeline for extraction, indexing, and analysis.

## 2. Scope

### 2.1 In Scope (Current Target)
- Linux runtime target (development from Linux and Windows).
- Local-only operation (no cloud dependency in core paths).
- Wearable ingest for health signals and voice transcripts.
- Wearable-to-compute BLE GATT sync with chunked object transfer (`manifest -> fetch -> commit/wipe`) and compute-side hash verification.
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
- Context must be assembled by scripts with explicit budgets and slot priorities.
- Answer claims must be evidence-backed; unsupported claims are marked uncertain.

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

### 4.7 Retrieval Engine
Retrieval is hybrid by default:
- lexical search for precise keyword/phrase matching
- vector search for semantic similarity
- deterministic fusion and optional reranking for final ordering

### 4.8 Context Assembly and Verification Layer (Use Soon)
Script-owned layer that:
- assembles a bounded context pack from policy, recent state, and retrieved evidence
- enforces deterministic ordering and token budgets per context slot
- validates answer claims against citations before returning final output

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
- `search.vector_index` rows include chunk-level attributes for retrieval scoring.

### 10.8 Retrieval Metadata (Use Soon)
- `search.query_cache`: cache for query embeddings, query expansions, and rerank outputs.
- `search.chunk_meta`: chunk-level metadata (`path`, `section`, `offset`, `token_count`, `content_hash`).
- `search.retrieval_run`: optional audit record for retrieval runs and score components.

### 10.9 Context and Verification (Use Soon)
- `search.context_profile`: named context budget profiles with fixed slot limits.
- `search.answer_verification_run`: claim-to-citation verification outcomes for returned answers.

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
- Generation model: `qwen3.5:35b-a3b`
- Embedding model: `qwen3-embedding:0.6b` with `dimensions=1024`

### 13.2 Separation rule
Separate models are preferred, but the same local model is allowed in constrained mode to minimize setup overhead.

### 13.3 Embedding lifecycle
- `search.vector_index` rows carry model identity and active/invalidation state.
- Re-indexing does not require deleting source text rows.

## 14. Retrieval Contract
The retrieval contract is deterministic and script-controlled.

### 14.1 Query Modes
Supported query modes:
- `lexical`: keyword/phrase-first retrieval.
- `semantic`: embedding similarity-first retrieval.
- `hybrid` (default): lexical + semantic fusion.
- `deep`: hybrid retrieval + query expansion + reranking.

### 14.2 Chunking Contract
Text is chunked before indexing using deterministic rules:
- preserve source boundaries (file path, section, row/document refs)
- prefer semantic boundaries (headings, paragraphs, list items) over blind token splits
- enforce chunk size window and overlap
- store chunk metadata and content hash for replay/reindex

### 14.3 Retrieval Pipeline
Runtime context assembly order:
1. policy and preference constraints (`/ME`-derived active rows)
2. recent decisions and reviews
3. candidate retrieval from lexical index and vector index
4. deterministic score fusion
5. optional rerank pass in `deep` mode
6. linked health context if requested

### 14.4 Score Fusion Rule
Hybrid retrieval must use deterministic fusion (for example rank-based reciprocal fusion).
Script controls fusion weights and boosts; model does not choose ranking policy.

### 14.5 Query Expansion Rule
In `deep` mode, query expansion may generate additional lexical and semantic variants.
Expansion results are validated, bounded, and deduplicated before retrieval.

### 14.6 Rerank Rule
Rerank is optional and only reorders retrieved candidates.
Rerank never introduces new unseen candidates.

### 14.7 Retrieval Output Contract
Every retrieved item returned to the model/UI must include citation metadata:
- source table/id
- source path or artifact reference
- chunk identifier and score components
- content hash/version marker when available

### 14.8 Caching Rule
`search.query_cache` may cache:
- query embeddings
- expansion outputs
- rerank outputs
Cache entries must be invalidated on model/version change.

### 14.9 Evaluation Contract
Retrieval quality is measured with repeatable offline checks:
- recall@k for known-answer sets
- precision@k for high-risk domains
- latency/cost per query mode
- citation coverage (returned answer claims mapped to retrieved evidence)

### 14.10 Context Budget Contract
Context assembly is deterministic and profile-driven:
- use `search.context_profile` to define slot budgets (policy, preferences, recent decisions, evidence, health context)
- do not let the model choose token allocation strategy
- truncate by deterministic priority rules when over budget

### 14.11 Time-Aware Retrieval Rule
Retrieval ranking must include deterministic time signals:
- recency boost for near-term recall tasks
- temporal overlap boost when query includes explicit/implicit time windows
- no boost for stale candidates unless evidence score justifies inclusion

### 14.12 Verification-Before-Trust Rule
Before final response:
- extract answer claims into atomic statements
- require each claim to map to at least one citation from retrieved evidence
- downgrade unmatched claims to `uncertain` instead of presenting as fact
- record verification results for audit and evaluation

### 14.13 Runtime Endpoints
- `POST /pipeline/search/reindex/memory_notes`: refreshes derived vector rows for `memory.note`.
- `POST /tools/retrieve`: retrieval API with `mode`, `limit`, and optional health context attachment.
- `POST /sync/wearable_ble`: wearable BLE object sync (`manifest -> fetch -> commit/wipe`) with hash verification.
- `GET /tools/transcript/recent`: returns recent transcript entries.

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
- Automated pytest status (`2026-02-26`): `16` tests passing.
- Automated coverage currently includes:
  - API route/ready semantics checks
  - selected deterministic timestamp + compaction retry rules
  - BLE sync protocol parsing/chunk transfer with fake transport
- Manual/scripted smoke checks exist for runtime flows (`e2e-smoke`, `timestamp-modes-smoke`, `wearable-ble-sync`, memory E2E harness), but are not CI-gated.
- Required next automated coverage:
  - DB-integrated ingest/process/report endpoint behavior
  - retrieval integration correctness (fusion/rerank/citation assertions)
  - extraction failure quarantine and candidate lifecycle transitions
  - idempotency/replay and `/ME` pointer integrity
