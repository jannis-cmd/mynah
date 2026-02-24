# MINAH Specification

Version: 0.2 (Draft baseline)
Status: Active

## 1. One-line Definition
MINAH is an open-source, offline-first personal intelligence system that captures wearable health and voice-note data, syncs it locally to a Linux compute node, builds persistent memory and insights, and presents results on a local display.

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

## 5. Agent Runtime Scaffold (mynah_agent)

### 5.1 Design Goals
- Local-first agent runtime for MINAH-specific tasks.
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
  - Normalizes tool-call responses to MINAH internal response objects.
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

### 5.4 Agent Data-Analysis Flow
- User request -> analysis planner.
- Planner requests `sql_query_readonly`.
- SQL validator enforces read-only constraints.
- Query executes against local SQLite.
- Result set returned as structured table payload.
- Agent generates narrative summary with linked evidence/citations.

### 5.5 Failure Handling
- If model/provider fails:
  - return explicit local error,
  - do not write partial memory updates.
- If SQL validation fails:
  - reject query and request reformulation.
- If citation verification fails:
  - downgrade or reject memory-backed conclusions.

## 6. Ollama Provider Contract

### 6.1 Provider Scope
- Only local models served through Ollama are supported.
- No cloud provider routing in the MINAH agent runtime.

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
  - explicit volume mounts only for required MINAH data paths,
  - no privileged mode,
  - no host PID/IPC namespace sharing.
- Network access is restricted to local services required for operation.
- Agent-to-model traffic must remain on an internal Docker network.
- No public or LAN-exposed model-serving port is allowed in default configuration.

### 7.3 Transparent Confinement
- Isolation boundaries and allowed resources must be documented and inspectable.
- The runtime must make permission decisions explicit in logs and documentation.
- Any expansion of permissions requires explicit project-level review and documentation update.

### 7.4 Platform Contract
- Runtime remains hardware-agnostic within Linux environments.
- No dependency on a specific accelerator or board family is required for core agent behavior.
- Hardware acceleration is optional optimization, not a functional requirement.

## 8. Agentic Memory Principles
MINAH uses these agentic memory principles for an offline personal system.

### 8.1 Evidence-Backed Memories
- Each memory item stores:
  - Subject/fact.
  - Why it matters.
  - Citations to source evidence in local storage (for example: DB row IDs, file paths, transcript offsets, report sections).
- Memories without valid evidence are not trusted for downstream reasoning.

### 8.2 Just-in-Time Verification
- Memory retrieval and memory trust are separate steps.
- Before a memory influences analysis or report output, the agent re-checks cited evidence against current local data.
- Verification is based on lightweight local reads to keep latency low.

### 8.3 Scoped Memory Boundaries
- Memories are scoped to this local MINAH instance and are never globally shared by default.
- Exported data does not implicitly include full memory history.
- Any cross-context sharing is explicit and user-initiated.

### 8.4 Freshness and Retention
- Memories are treated as perishable knowledge.
- Retention/expiry is policy-driven (default window is finite and configurable).
- Frequently re-validated memories are refreshed; stale or unverified memories are deprioritized or expired.

### 8.5 Self-Healing Memory
- If verification contradicts a stored memory, the memory is corrected or superseded by a newer validated record.
- The system preserves provenance and change history so corrections are auditable.

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

## 12. Security and Trust Boundaries
- Offline-first operation by default.
- No implicit external network dependencies for core workflows.
- Export is user-initiated and explicit.
- Local threat model assumes trusted home environment for v0.x.
- Physical anti-tamper hardening is not a v0.x goal.
- Memory scope is local-instance bounded by default.
- Memory usage requires citation validation at use time.

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

### 15.6 Packaging and Deployment
- Compute services are containerized with Docker.
- Default container base image is Debian slim for compatibility and maintenance balance.
- Least-privilege runtime settings are mandatory (see runtime isolation section).

### 15.7 Dependency Policy
- Standard library first, then small curated third-party dependencies.
- Add dependencies only with explicit justification and maintenance review.
- Avoid introducing heavy distributed infrastructure dependencies (for example: Kubernetes, Redis, Kafka) in v0.x.

### 15.8 Testing Stack
- Automated tests use `pytest`.
- Focus on unit + integration coverage for sync, data integrity, memory verification, and SQL guardrails.

## 16. Documentation Contract
The project maintains the following canonical docs:
- `readme.md`: project overview and navigation.
- `spec.md`: product and architecture solution contract.
- `testing.md`: test strategy and tracking state.
- `docs/agentic-memory.md`: memory principles, lifecycle, and governance details.
