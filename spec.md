# MYNAH Specification

Version: 0.3
Status: Active

## 1. One-Line Definition
MYNAH is an open-source, offline-first personal intelligence system that ingests wearable and human-authored artifacts locally, structures them through a local LLM into a durable memory database, and serves local analysis/reporting on-device.

## 2. Scope

### 2.1 In Scope (Current Solution)
- Linux-targeted local runtime for compute services.
- Development from Linux and Windows hosts.
- Offline-first data flow with no cloud dependency in core paths.
- Wearable ingest placeholders for HR and voice-note audio.
- Human input ingest (`voice_transcript`, `ME.md`, notes/doc text).
- Local LLM processing through Ollama only.
- Structured write pipeline:
  - raw artifact storage,
  - LLM write-plan proposal,
  - deterministic validation,
  - bounded corrective retries,
  - transactional persistence.
- Postgres + pgvector as primary local datastore.
- Local UI dashboard on attached display.
- Explicit user-initiated export via USB.

### 2.2 Out of Scope (v0.x)
- Cloud sync, accounts, telemetry, remote web access.
- Medical diagnosis/regulated medical claims.
- Multi-user identity and role sharing.
- Internet-required operation.

## 3. Platform Contract
- Runtime target: Linux.
- Dev host support: Linux and Windows.
- Compute services run as Docker containers with least privilege.
- LLM serving is local via Ollama on internal container network only.
- No public/LAN exposure of model endpoint in default deployment.

## 4. System Components

### 4.1 Wearable
- Captures HR and voice-note audio.
- Buffers data locally until sync commit is acknowledged.
- Uses BLE-only sync contract.

### 4.2 Compute Daemon (`mynahd`)
- Accepts ingest payloads (fixture and production pathways).
- Persists HR and audio metadata to Postgres.
- Persists audio/transcript fixture artifacts to local artifact storage.
- Provides summary/status endpoints for UI and test loops.

### 4.3 Agent (`mynah_agent`)
- Stores raw artifacts exactly as received.
- Runs LLM extraction into strict structured write plans.
- Validates plans against schema/business rules.
- Returns validator errors to LLM for correction (bounded retries).
- Applies valid plans transactionally into canonical tables.
- Audits every write-plan attempt.
- Generates local daily reports.

### 4.4 UI (`mynah_ui`)
- Appliance-style local UI.
- Reads local daemon and agent status.
- Shows HR summary, recent notes/transcripts, and report history.

### 4.5 Model Provider (Ollama)
- Local-only model runtime.
- Agent uses Ollama for generation and embeddings.
- Startup readiness requires configured model availability.

## 5. Dataflow Contract

### 5.1 Ingestion and Raw Truth
1. Input arrives as audio/transcript/ME.md/document text.
2. Raw artifact is stored first in `artifacts` with hash and metadata.
3. Raw artifact remains the reprocessable source of truth.

### 5.2 Structured Write Pipeline
1. Agent retrieves artifact and semantic neighbors (`pgvector`) for context.
2. Agent prompts LLM to output one strict JSON write plan.
3. Validator checks structure and policy:
  - required fields,
  - allowed actions/types,
  - reference integrity,
  - link sanity.
4. If validation fails, structured errors are fed back to LLM.
5. LLM retries with corrected plan up to configured max attempts.
6. If validation passes, plan is committed transactionally.
7. If retries are exhausted, artifact is marked failed and a candidate record is created.

### 5.3 Trust Rule
- LLM proposes.
- Validator and transaction layer decide.
- No direct LLM DB writes.

## 6. LLM Write-Plan Contract

### 6.1 Required Root Keys
- `artifact_summary`
- `new_entries`
- `new_facts`
- `links`
- `dedupe_candidates`
- `questions`

### 6.2 Entry Taxonomy
Allowed `entry_type` values:
- `memory`
- `health`
- `preference`
- `idea`
- `decision`
- `relationship`
- `event`
- `task`

### 6.3 Action Model
Allowed actions:
- `create`
- `update`
- `link`
- `duplicate`
- `question`

### 6.4 Confidence and Sensitivity
- Confidence is required on created/updated records.
- Sensitive domains (`health`, `decision`) must use conservative confidence when uncertain.

## 7. Database Contract (Postgres + pgvector)

### 7.1 Primary Datastore
- PostgreSQL is the canonical datastore.
- `pgvector` extension is enabled for semantic retrieval.
- No ORM in v0.x; direct SQL via `psycopg`.

### 7.2 Core Tables
- `artifacts`: immutable raw textual inputs and processing state.
- `audio_note`: audio ingest metadata and fixture transcript pointer.
- `transcript`: transcript text artifacts.
- `entries`: canonical structured memory/event/preference rows.
- `entry_version`: append-only snapshots for update audit.
- `facts`: typed structured claims with confidence and status.
- `links`: graph edges between entries.
- `write_plan_audit`: full audit of each LLM/validator attempt.
- `report_artifact`: generated report records.

### 7.3 Persistence Rules
- All successful writes are transactional.
- No silent overwrite of previous semantic state.
- Updates create version history.
- Audit rows are mandatory for every write-plan attempt.

## 8. Retention and Lifecycle

### 8.1 Long-Term Memory
- Structured semantic memory is retained indefinitely by default.
- Why: lifelong continuity is a core product objective.

### 8.2 Raw Audio
- Raw audio may be deleted after successful transcription and integrity checks.
- Why: reduce storage growth while preserving structured truth.

### 8.3 Transcript Artifacts
- Transcript artifacts are retained for 180 days by default.
- After retention window, deletion is allowed once compaction/citation integrity is satisfied.
- Why: retain contestability/reprocessing window without indefinite raw-text growth.

### 8.4 Failed Processing
- Failed artifacts remain auditable.
- Failed writes produce explicit candidate records instead of silent drops.

## 9. Security and Isolation

### 9.1 Runtime Isolation
- Containers run as non-root with `cap_drop: [ALL]`.
- `no-new-privileges` enabled.
- Read-only root filesystem with explicit writable mounts only.
- Internal Docker network marked `internal: true`.

### 9.2 Network Policy
- No default host/LAN port exposure for internal services.
- Agent-to-Ollama traffic stays inside the internal network.

### 9.3 BLE Security Baseline
- LE Secure Connections with MITM-protected pairing.
- Bonding required; explicit physical pairing mode required.
- Replay/downgrade protections enforced.
- Commit/wipe flow idempotent and authenticated.

### 9.4 Export Contract (Current)
- Export medium: user-provided USB storage.
- Export trigger: explicit local UI confirmation.
- Export scope: minimal selected outputs by default.
- Export package: signed bundle with manifest/checksums.

## 10. API Surface (Current Runtime)

### 10.1 Daemon
- `GET /health`
- `GET /ready`
- `POST /ingest/hr`
- `POST /ingest/audio`
- `GET /summary/hr/today`
- `GET /summary/audio/recent`
- `GET /status`

### 10.2 Agent
- `GET /health`
- `GET /ready`
- `POST /pipeline/artifacts/ingest`
- `POST /pipeline/artifacts/process/{artifact_id}`
- `POST /pipeline/me_md/process`
- `POST /pipeline/audio/transcribe`
- `GET /tools/transcript/recent`
- `POST /tools/report_generate`
- `GET /tools/report_recent`
- `GET /status`

### 10.3 UI
- `GET /health`
- `GET /status`
- `GET /`

## 11. Reliability and Failure Model
- Explicit failure is required; no silent fallback implementations.
- Write-plan retries are bounded and auditable.
- Partial/invalid writes do not commit.
- Readiness fails when Postgres or required local model is unavailable.

## 12. Documentation Contract
- `readme.md`: project overview and runtime quickstart.
- `spec.md`: architecture and behavioral contract.
- `testing.md`: active test strategy and execution status.
- These documents must stay synchronized in each architecture change.
