# MYNAH Specification

Version: 0.2 (Draft baseline)
Status: Active

## 1. One-line Definition
MYNAH is an open-source, offline-first personal intelligence system that captures wearable health and voice-note data, syncs it locally to a Linux compute node, builds persistent memory and insights, and presents results on a local display.

## 2. Product Scope

### In Scope (Current Solution)
- Wearable data capture:
  - Heart rate sampling.
  - Voice note recording by hardware button.
- Local buffering on wearable with reliable later sync.
- Local BLE sync from wearable to compute node.
- Local persistence (SQLite + filesystem artifacts).
- Local transcription and memory enrichment pipeline.
- Local dashboard and reports on attached display.
- Explicit user-initiated export only.

### Out of Scope (Current Solution)
- Cloud sync, remote APIs, telemetry, user accounts.
- Clinical/medical diagnosis or regulated health claims.
- Multi-user profile management.
- Internet-dependent operation.

## 3. Platform Targets
- Production target: Linux compute node.
- Development host support: Linux and Windows.
- Wearable firmware stack: ESP-IDF (ESP32-C3 class device).

Notes:
- Compute runtime behavior is Linux-targeted.
- Windows is a supported development environment for authoring, testing (where feasible), and tooling.
- Hardware availability is not required to begin with protocol and service skeletons.

## 4. System Components

### 4.1 Wearable
- Records HR samples at configurable interval (default 1 Hz).
- Records voice notes with start/stop button interaction.
- Stores data locally until durable sync commit.
- Exposes BLE GATT service for status and transfer.
- Wipes only objects acknowledged as committed by compute.

### 4.2 Compute Node
- Runs Linux daemon for BLE sync orchestration.
- Validates object integrity (hash/checksum).
- Persists structured and file artifacts.
- Emits local events for agent and UI.
- Provides internal-only fixture ingest and summary endpoints for deterministic no-wearable E2E loops.
- Why: a closed local debug loop is required before hardware sync is available.

### 4.3 Agent + Memory
- Ingests synced events and artifacts.
- Generates transcripts and derived memory items.
- Supports retrieval and report-oriented analysis.
- Uses SQL-first analytics with read-only default execution.
- Uses evidence-backed memory records with verification before use.

### 4.4 UI
- Local display dashboard with sync/device health.
- Daily HR summaries and note/transcript browsing.
- Report generation and historical report viewing.
- Export section for local explicit export actions.
- UI runtime mode is local kiosk on the attached compute display.
- Why: kiosk mode supports appliance-style operation without publishing network ports.

## 5. Agent Runtime Scaffold (mynah_agent)

### 5.1 Design Goals
- Local-first agent runtime for MYNAH-specific tasks.
- Deterministic tool execution with explicit guardrails.
- Clear separation between:
  - LLM provider,
  - tool layer,
  - memory/data services.

### 5.2 Runtime Modules
- `agent_loop`:
  - Iterative reasoning/tool-call loop.
  - Stops on final answer or max-iteration limit.
- `provider`:
  - Single LLM provider implementation backed by local Ollama.
  - Normalizes tool-call responses to MYNAH internal response objects.
- `tool_registry`:
  - Registers tool schemas and dispatch handlers.
  - Validates parameters before execution.
- `session_store`:
  - Stores per-session message history in local persistent storage.
  - Supports bounded context windows.
- `memory_service`:
  - Reads/writes memory items, links, citations, freshness state.
  - Enforces verification before memory trust.
- `analysis_service`:
  - Handles SQL generation/execution pipeline with read-only policy.
  - Produces structured outputs for UI/report generation.

### 5.3 Tooling Contract (v0.x)
- Tools are first-class runtime functions, not user-installable skill packs.
- Initial built-in tools:
  - `sql_query_readonly`
  - `memory_search`
  - `memory_upsert`
  - `report_generate`
- Excluded from initial runtime:
  - network/web tools,
  - shell execution tools,
  - generic external integrations.
- `memory_upsert` is governance-gated and may reject low-quality writes.
- Internal runtime API exposure for tools:
  - `POST /tools/sql_query_readonly`
  - `GET /tools/query_audit/recent`
  - `POST /tools/memory_upsert`
  - `POST /tools/memory_search`
  - `GET /tools/memory_verify/{memory_id}`
  - `POST /pipeline/audio/transcribe`
  - `GET /tools/transcript/recent`
- Why: explicit local endpoints keep tool behavior inspectable and testable during the closed E2E loop.

### 5.4 Agent Data-Analysis Flow
- User request -> analysis planner.
- Planner requests `sql_query_readonly`.
- SQL validator enforces read-only constraints.
- Query executes against local SQLite.
- Result set returned as structured table payload.
- Agent generates narrative summary with linked evidence/citations.

### 5.7 Memory Retrieval Policy (Locked)
- Retrieval ranking order:
  - verified citation validity,
  - freshness status,
  - semantic relevance score,
  - recency weight.
- Prompt budget control:
  - retrieved memory payload is bounded by token budget, preferring higher-ranked verified items.
- Insight generation rule:
  - derived insights must rely on memories that pass citation verification at read time.
- Why: deterministic ranking and bounded context are required to prevent retrieval drift and low-value memory pollution.

### 5.5 SQL Safety Policy (Locked)
- Read-only enforcement uses a dual gate:
  - SQL parser with statement allowlist,
  - read-only SQLite connection mode.
- Allowed statement types:
  - `SELECT`,
  - `WITH`,
  - `EXPLAIN QUERY PLAN`.
- Forbidden statement/classes:
  - DDL,
  - DML,
  - `ATTACH`/`DETACH`,
  - mutating `PRAGMA`,
  - any construct that can modify database state.
- Query resource limits (per execution):
  - max runtime: 5 seconds,
  - max rows returned: 10,000,
  - max serialized result payload: 5 MB.
- Pagination rule:
  - queries without explicit `LIMIT` are rejected.
- Table/view access policy:
  - only allowlisted analytics-safe tables/views are queryable.
- Audit logging:
  - record normalized query, parameters hash, caller, latency, row count, and outcome.
- Error contract returned to agent:
  - structured error with `code`, `reason`, `retryable`, and `suggestion`.

### 5.6 Failure Handling
- If model/provider fails:
  - return explicit local error,
  - do not write partial memory updates.
- If SQL validation fails:
  - reject query and request reformulation.
- If citation verification fails:
  - downgrade or reject memory-backed conclusions.
- If audio transcription fixture is missing in no-wearable E2E mode:
  - fail explicitly and require `transcript_hint` on ingest.
  - Why: explicit failure keeps the debug loop transparent and avoids hidden fallback behavior.

## 6. Ollama Provider Contract

### 6.1 Provider Scope
- Only local models served through Ollama are supported.
- No cloud provider routing in the MYNAH agent runtime.
- Baseline v0.x model: `qwen2.5:7b-instruct`.
- Fallback model behavior is disabled by default (fail explicitly on model failure).

### 6.2 Connection Model
- Provider endpoint is configurable (default: `http://127.0.0.1:11434`).
- Runtime uses Ollama chat API compatibility for:
  - message exchange,
  - tool/function call support (model-dependent).
- Preferred deployment uses container-internal networking between `mynah_agent` and Ollama.
- Ollama endpoint must not be exposed outside the local Docker network in default deployment.

### 6.3 Provider Requirements
- Configurable:
  - model name,
  - timeout,
  - max tokens,
  - temperature.
- Must return normalized response type:
  - `content`,
  - `tool_calls`,
  - `usage` (if available),
  - `finish_reason`.

### 6.4 Capability Guard
- On startup, provider performs a local capability probe.
- If selected model lacks reliable tool-call behavior:
  - runtime enters constrained mode (analysis disabled or reduced),
  - UI/status exposes capability mismatch.

## 7. Agent Runtime Isolation and Permissions

### 7.1 Containerized Execution
- The agent runtime (`mynah_agent`) runs inside Docker on the Linux compute node.
- Containerization is required to enforce operational boundaries and reduce host-risk exposure.

### 7.2 Strict Permission Model
- Default container policy is least-privilege:
  - non-root runtime user,
  - read-only root filesystem where feasible,
  - explicit volume mounts only for required MYNAH data paths,
  - no privileged mode,
  - no host PID/IPC namespace sharing.
- Network access is restricted to local services required for operation.
- Agent-to-model traffic must remain on an internal Docker network.
- No public or LAN-exposed model-serving port is allowed in default configuration.
- Default deployment publishes no service ports to host/LAN.
- Why: appliance-style local operation minimizes attack surface and avoids unnecessary network exposure.

### 7.3 Transparent Confinement
- Isolation boundaries and allowed resources must be documented and inspectable.
- The runtime must make permission decisions explicit in logs and documentation.
- Any expansion of permissions requires explicit project-level review and documentation update.

### 7.5 Container Topology and Network Policy (Locked)
- Service split:
  - `mynahd`, `mynah_agent`, `mynah_ui`, `ollama` as separate containers.
  - Why: clear isolation boundaries improve security posture and operational clarity.
- Network topology:
  - single internal Docker network for service-to-service communication.
  - Why: one isolated network keeps routing simple while preserving service isolation from host/LAN.
- Port publication:
  - no published ports in default deployment.
  - Why: local display operation does not require external network reachability.
- Inter-service authentication:
  - mTLS between internal services with automatic local certificate issuance and automatic rotation.
  - Why: strong service identity/authentication without requiring manual user key management.
- Host access:
  - only explicitly declared data paths may be mounted.
  - Why: strict mount allowlisting prevents accidental host-scope privilege expansion.

### 7.6 Inter-Service mTLS Operations (Locked)
- Local certificate authority:
  - single local CA generated and stored in compute secure runtime data path (`db` volume).
  - Why: local CA keeps trust roots offline and under device control.
- Certificate rotation:
  - service certificates rotate automatically every 30 days.
  - Why: periodic automated rotation reduces long-lived credential risk without user burden.
- Renewal and rollover:
  - renewal starts 7 days before expiry and supports overlap validity.
  - Why: overlap windows avoid service interruption during rotation.
- Revocation handling:
  - revoked service certs are denied immediately on next handshake.
  - Why: deterministic revocation closes compromised identities quickly.
- User interaction:
  - no manual user action is required for certificate issuance, renewal, or revocation.
  - Why: security maintenance must be automatic to remain reliable in practice.

### 7.4 Platform Contract
- Runtime remains hardware-agnostic within Linux environments.
- No dependency on a specific accelerator or board family is required for core agent behavior.
- Hardware acceleration is optional optimization, not a functional requirement.

## 8. Agentic Memory Principles
MYNAH uses these agentic memory principles for an offline personal system.

### 8.1 Evidence-Backed Memories
- Each memory item stores:
  - Subject/fact.
  - Why it matters.
  - Citations to source evidence in local storage.
  - Immutable citation anchors:
    - `content_hash`,
    - `schema_version`,
    - `snapshot_ref` (timestamp or revision id).
- Memories without valid evidence are not trusted for downstream reasoning.

### 8.2 Just-in-Time Verification
- Memory retrieval and memory trust are separate steps.
- Before a memory influences analysis or report output, the agent re-checks cited evidence against current local data.
- Verification is based on lightweight local reads to keep latency low.

### 8.3 Scoped Memory Boundaries
- Memories are scoped to this local MYNAH instance and are never globally shared by default.
- Exported data does not implicitly include full memory history.
- Any cross-context sharing is explicit and user-initiated.

### 8.4 Freshness and Retention
- Memories are treated as perishable knowledge.
- Freshness is policy-driven and type-specific; semantic memory retention is indefinite by default.
- Frequently re-validated memories are refreshed; stale or unverified memories are deprioritized or expired.

### 8.5 Self-Healing Memory
- If verification contradicts a stored memory, the memory is corrected or superseded by a newer validated record.
- The system preserves provenance and change history so corrections are auditable.

### 8.6 Memory Lifecycle Defaults (Locked)
- Semantic memory retention:
  - retained indefinitely by default.
  - Why: long-term personal continuity is a core product objective.
- Audio artifact retention:
  - raw audio may be deleted after successful transcription and integrity verification.
  - Why: transcripts and structured memory become the durable source while reducing storage growth.
- Transcript artifact retention:
  - transcript files are retained for 180 days, then eligible for deletion after compaction into canonical DB/memory records with citations.
  - Why: fixed retention preserves recheck/contestability window while still controlling long-term storage growth.
- Citation requirement for trusted output:
  - derived insights require at least 2 valid citations; direct facts require at least 1 valid citation.
  - Why: stronger citation thresholds reduce hallucinated conclusions in analytical output.
- Freshness policy:
  - per-type TTL policy is used (facts/events/insights/procedures).
  - Why: different memory classes decay at different rates and need class-specific handling.
- Stale-memory handling:
  - stale memories are excluded from trusted output paths until revalidated.
  - Why: trust guarantees require freshness before use.
- Supersession conflict rule:
  - newest verified memory supersedes conflicting older memory.
  - Why: deterministic conflict resolution improves predictability and auditability.
- User correction precedence:
  - explicit user corrections always override model-derived memory.
  - Why: user intent and correction authority are first-class.
- Deletion model:
  - soft delete with tombstone, followed by retention-policy hard delete where applicable.
  - Why: tombstones preserve audit/history while enabling eventual cleanup.
- Reverification cadence:
  - verification on read plus periodic daily background reverification.
  - Why: combines low-latency trust checks with continuous memory hygiene.

### 8.8 Memory Write Governance (Locked)
- Memory writes (`memory_upsert`) require:
  - salience threshold,
  - confidence floor,
  - dedupe check against recent/semantically similar memories,
  - per-period write rate limits.
- Writes failing governance checks are rejected and recorded in audit state.
- Why: governance gates reduce accumulation of low-quality or duplicate local-model memory artifacts.

### 8.9 Sensitivity and User Control (Locked)
- Memory items are classified by sensitivity level (`low`, `personal`, `sensitive`).
- User controls apply per class:
  - export allow/deny,
  - visibility controls,
  - deletion/forget actions.
- Forget actions create tombstones with audit metadata.
- Why: indefinite semantic retention requires explicit user-governed privacy controls.

### 8.7 Freshness TTL Values (Locked)
- Facts TTL: 365 days.
- Events TTL: 30 days.
- Insights TTL: 14 days.
- Procedures TTL: 90 days.
- Why: stable facts remain relevant longer, while derived insights and events require faster refresh cycles.

## 9. Data and Storage Model

### 9.1 Persistent Stores
- SQLite database for canonical structured records.
- Filesystem object store for audio/transcripts/reports.

### 9.2 Core Record Types
- Device metadata.
- Sync sessions.
- HR samples or HR segments.
- Audio notes and transcript artifacts.
- Memory items and memory links.
- Memory citations/evidence references.
- Memory revision history (corrections/supersession lineage).

### 9.3 Data Integrity Rules
- Synced objects are content-validated before commit.
- Commit and wearable wipe are separate explicit phases.
- Data is never deleted from wearable before commit acknowledgment.

### 9.4 Schema Lock (v0.x)
- HR storage model:
  - Segment table plus optional raw sample table.
  - Why: preserves efficient analytics while keeping raw-detail availability for debugging and reprocessing.
- Transcript storage model:
  - Transcript table plus transcript segment table (`start_ms`, `end_ms`, `text`, `confidence`).
  - Why: time-aligned segments improve citation precision and downstream memory linking.
- Memory tag/entity model:
  - Hybrid approach: normalized truth tables with optional JSON cache fields.
  - Why: normalized data supports robust querying while JSON cache keeps common reads simple.
- Citation model:
  - Normalized `memory_citation` table.
  - Citation rows must store immutable anchors (`content_hash`, `schema_version`, `snapshot_ref`) in addition to relational references.
  - Why: explicit citations are required for verification, auditability, and referential integrity.
- Memory revision model:
  - Append-only `memory_revision` table with current-item pointer.
  - Why: immutable revision history enables transparent self-healing and rollback-safe auditing.
- Sync object tracking:
  - Dedicated `sync_object` table with per-object transfer/validation/commit state.
  - Why: object-level state tracking improves resumability, diagnostics, and idempotency guarantees.
- SQL analysis audit:
  - `query_audit` table capturing query, caller, latency, row count, result hash, and denial reason.
  - Why: audit records are needed to enforce safe SQL behavior and make agent analysis traceable.
- Memory write audit:
  - `memory_write_audit` table capturing attempted writes, governance pass/fail status, rejection reason, and source context.
  - Why: write-path auditability is required to control memory quality over long-lived operation.
- Migration strategy:
  - Raw SQL versioned migration files.
  - Why: keeps schema evolution transparent, tool-light, and aligned with minimal-stack principles.
- Internal compute API contract:
  - daemon exposes:
    - `POST /ingest/hr`,
    - `GET /summary/hr/today`,
    - `POST /ingest/audio`,
    - `GET /summary/audio/recent`
    on the internal runtime network.
  - Why: deterministic fixture ingestion and summary inspection are required to keep the E2E debug loop fast and transparent.

## 10. BLE Sync Contract (Solution-Level)
- Custom GATT service with characteristics for:
  - Device info and capabilities.
  - Device status.
  - Object manifests (HR/audio).
  - Chunk fetch requests.
  - Sync commit and wipe confirmation.
  - Time synchronization.
- Transfer is resumable and chunk-based.
- Protocol supports object-level integrity checks.
- Encrypted BLE link is required.

### 10.1 Wire-Level Decisions (Locked)
- Object identity:
  - Hybrid IDs: `UUIDv7` object ID plus separate `SHA-256` content hash.
  - Why: combines stable transfer identity with strong integrity and dedupe support.
- Chunk sizing:
  - Adaptive transfer chunks: start at 1024 bytes and downshift on error conditions.
  - Why: balances throughput and reliability across variable BLE link quality.
- Resume semantics:
  - Dual resume model: `(object_id, next_offset, object_hash)` plus wearable-issued session token.
  - Why: supports both stateless recovery and stronger session-bound continuity checks.
- Retry/backoff:
  - Exponential backoff with jitter and per-object retry budget.
  - Why: reduces retry storms and improves stability on noisy/intermittent links.
- Timeout defaults:
  - Connect timeout: 15 seconds.
  - Chunk timeout: 10 seconds.
  - Commit timeout: 10 seconds.
  - Why: conservative defaults reduce false failures during early field stabilization.
- Protocol versioning:
  - `major.minor` protocol version plus capability bitmap.
  - Why: enables explicit feature negotiation and safe forward evolution.
- Error contract:
  - Structured errors with numeric code, error class, and retryability hint.
  - Why: deterministic error handling and diagnostics are required for reliable sync.

## 11. BLE Security Profile (Mandatory)

### 11.1 Security Model
- BLE protocol security assumes public source code and private keys.
- Security must not depend on obscurity of protocol details or implementation.

### 11.2 Pairing and Bonding
- LE Secure Connections is required.
- Legacy pairing modes are disallowed.
- MITM-protected pairing is required.
- Bonding is required and limited to explicit trusted owners.
- Pairing is allowed only during explicit physical pair mode with timeout.

### 11.3 Ownership and Access Control
- Wearable maintains an allowlist of trusted compute identities.
- Non-allowlisted peers are rejected before data transfer.
- Pairing attempts are rate-limited with lockout/backoff on repeated failures.

### 11.4 Application-Layer Session Security
- BLE link encryption is required but not sufficient.
- Sync traffic uses application-layer authenticated encryption per session.
- Session establishment includes ephemeral key exchange and mutual authentication.
- Each encrypted frame includes unique nonce/counter; reuse is forbidden.

### 11.5 Replay and Downgrade Protection
- Protocol version is explicit and validated on handshake.
- Downgrade to weaker protocol/security modes is rejected.
- Monotonic counters or equivalent anti-replay primitives are required.
- Stale session IDs, stale counters, or duplicate commit tokens are rejected.

### 11.6 Data and Commit Authenticity
- Object manifests include authenticated integrity metadata.
- Commit/wipe operations are bound to authenticated session state.
- Wearable wipes only after validated durable commit acknowledgment.
- Commit/wipe flow is idempotent and auditable.

### 11.7 Device Key Protection
- Device long-term keys must be protected at rest.
- Secure boot and flash encryption are required on production firmware profiles.
- Key material must not be logged or exposed through debug interfaces in production mode.

### 11.8 Privacy Hardening
- BLE address privacy features are enabled (resolvable private addresses).
- Advertising payload minimizes identifying information.
- Operational metadata exposure is minimized to least required fields.

### 11.9 Security Verification Requirements
- Automated tests must cover:
  - replay attempts,
  - downgrade attempts,
  - MITM-oriented pairing failures,
  - interrupted transfer recovery,
  - commit/wipe atomicity.
- Protocol parser and message handling should include fuzzing coverage.
- BLE fuzzing acceptance gates:
  - minimum 100,000 generated protocol inputs per CI fuzz run,
  - no crashes, panics, or memory-safety faults,
  - no silent acceptance of malformed authenticated frames.
- Why: measurable fuzz gates are required to make parser hardening enforceable.

### 11.10 Key Lifecycle Policy (Final)
- Ownership provisioning uses explicit physical pair mode only:
  - button hold for 10 seconds to enter pair mode,
  - pair window expires after 5 minutes,
  - single-owner binding is enforced.
- Trust anchors are asymmetric:
  - wearable stores trusted compute public key(s),
  - compute stores wearable public key.
- Key rotation is automatic and mandatory; user action is not required.
- Revocation uses an automatic policy engine with deterministic rules and full audit logging.
- Lost compute recovery is supported via physical recovery mode on wearable.
- Lost wearable handling on compute:
  - mark device as revoked,
  - preserve existing data immutably,
  - allow replacement wearable onboarding.
- Anti-replay/session counters are persisted durably:
  - encrypted NVS on wearable,
  - durable SQLite state on compute.
- Development and production key domains are strictly separated.
- Production profiles must not expose debug key material or key-sensitive logs.

## 12. Security and Trust Boundaries
- Offline-first operation by default.
- No implicit external network dependencies for core workflows.
- Export is user-initiated and explicit.
- Local threat model assumes trusted home environment for v0.x.
- Physical anti-tamper hardening is not a v0.x goal.
- Memory scope is local-instance bounded by default.
- Memory usage requires citation validation at use time.

### 12.1 Export Contract (Locked, v0.x)
- Export medium:
  - user-provided USB storage only.
  - Why: removable media supports explicit, user-controlled offline transfer.
- Trigger model:
  - export is initiated manually from local UI with explicit confirmation.
  - Why: deliberate user action prevents accidental bulk data disclosure.
- Export package format:
  - signed archive bundle containing selected report/data artifacts plus manifest.
  - Why: signed bundles preserve integrity and provenance during offline transfer.
- Scope defaults:
  - exports are minimal by default (selected reports/derived summaries), not full raw memory history.
  - Why: least-data export reduces privacy exposure risk.
- Verifiability:
  - each export includes checksum manifest and signature verification metadata.
  - Why: receiving systems must be able to validate authenticity and integrity offline.
- Future compatibility:
  - additional export channels may be introduced later but must preserve explicit user-initiation and minimal-data defaults.
  - Why: future expansion must not weaken trust and containment guarantees.

## 13. Reliability Requirements
- Wearable must survive intermittent connectivity and power cycles.
- Compute sync process must be idempotent at session/object level.
- System must recover cleanly from partial transfers.
- UI must expose meaningful sync and storage health states.
- Invalid or stale memories must not silently influence outputs.

## 14. Operational Constraints
- Compute solution remains Linux-compatible as primary runtime contract.
- Development workflow remains usable from both Linux and Windows hosts.
- Hardware-specific acceleration (e.g., Jetson class devices) is optional, not mandatory for core architecture.

## 15. Software Stack Decisions (v0.x)

### 15.1 Core Languages and Runtime
- Compute services use Python 3.11+.
- Wearable firmware uses ESP-IDF (C/C++).
- Runtime target is Linux.

### 15.2 Data Layer
- Primary datastore is SQLite.
- DB access is direct SQL through Python `sqlite3` (no ORM in v0.x).
- SQL-first analytics is the default analysis path.

### 15.3 UI Stack
- Local UI uses FastAPI with server-rendered templates.
- Avoid heavy frontend frameworks in v0.x.
- UI remains local-only and offline-capable.

### 15.4 Agent Tooling Model
- Use built-in function tools (not external skill/plugin systems) for core operations.
- Initial tool set remains minimal and domain-specific:
  - `sql_query_readonly`
  - `memory_search`
  - `memory_upsert`
  - `report_generate`

### 15.5 LLM and Model Serving
- LLM runtime uses local models served through Ollama only (v0.x).
- Agent-to-Ollama traffic stays on internal Docker network only.
- No public/LAN exposure of model-serving endpoints in default deployment.
- Baseline model for v0.x is `qwen2.5:7b-instruct`.
- Additional fallback models are not included unless explicitly approved.

### 15.6 Transcription Stack
- Baseline transcription engine for v0.x is `whisper.cpp`.
- Why: minimal dependency footprint, open/local execution, and strong alignment with offline-first constraints.

### 15.7 Packaging and Deployment
- Compute services are containerized with Docker.
- Default container base image is Debian slim for compatibility and maintenance balance.
- Least-privilege runtime settings are mandatory (see runtime isolation section).
- Writable data is split into dedicated volumes:
  - `db`,
  - `artifacts`,
  - `logs`,
  - `ollama_models`.
- Why: separated volumes improve backup/recovery hygiene and reduce cross-service coupling.
- Root filesystems are read-only by default, with explicit writable mounts only.
- Why: read-only roots reduce persistence opportunities for runtime compromise.
- All services run as non-root users.
- Why: non-root execution reduces container breakout impact.
- Runtime hardening defaults:
  - `cap_drop: [ALL]`,
  - `no-new-privileges: true`,
  - default seccomp profile unless explicitly documented exceptions are required.
- Why: hardened runtime defaults reduce privilege escalation pathways.
- Health management:
  - healthchecks on all services,
  - restart policy `unless-stopped`.
- Why: explicit liveness checks and controlled restart policy improve unattended reliability.

### 15.8 Dependency Policy
- Standard library first, then small curated third-party dependencies.
- Add dependencies only with explicit justification and maintenance review.
- Avoid introducing heavy distributed infrastructure dependencies (for example: Kubernetes, Redis, Kafka) in v0.x.

### 15.9 Testing Stack
- Automated tests use `pytest`.
- Focus on unit + integration coverage for sync, data integrity, memory verification, and SQL guardrails.
- Testing output and status reporting must be explicit and transparent.

## 16. Documentation Contract
The project maintains the following canonical docs:
- `readme.md`: project overview and navigation.
- `spec.md`: product and architecture solution contract.
- `testing.md`: test strategy and tracking state.
- `docs/agentic-memory.md`: memory principles, lifecycle, and governance details.

## 17. Decision Register

### 17.1 Decision Recording Rule
- Every major architecture/security/runtime decision must be recorded in this section.
- Each entry must include:
  - Decision
  - Choice
  - Why (one sentence)
  - Status (`locked` or `draft`)

### 17.2 Placement Rule (Locked)
- Locked decisions must be written in their respective functional chapters.
- This section exists only as a recording rule and index anchor, not as a duplicate decision dump.
