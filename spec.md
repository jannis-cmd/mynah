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

### 4.4 UI
- Local display dashboard with sync/device health.
- Daily HR summaries and note/transcript browsing.
- Report generation and historical report viewing.
- Export section for local explicit export actions.

## 5. Data and Storage Model

### 5.1 Persistent Stores
- SQLite database for canonical structured records.
- Filesystem object store for audio/transcripts/reports.

### 5.2 Core Record Types
- Device metadata.
- Sync sessions.
- HR samples or HR segments.
- Audio notes and transcript artifacts.
- Memory items and memory links.

### 5.3 Data Integrity Rules
- Synced objects are content-validated before commit.
- Commit and wearable wipe are separate explicit phases.
- Data is never deleted from wearable before commit acknowledgment.

## 6. BLE Sync Contract (Solution-Level)
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

## 7. Security and Trust Boundaries
- Offline-first operation by default.
- No implicit external network dependencies for core workflows.
- Export is user-initiated and explicit.
- Local threat model assumes trusted home environment for v0.x.
- Physical anti-tamper hardening is not a v0.x goal.

## 8. Reliability Requirements
- Wearable must survive intermittent connectivity and power cycles.
- Compute sync process must be idempotent at session/object level.
- System must recover cleanly from partial transfers.
- UI must expose meaningful sync and storage health states.

## 9. Operational Constraints
- Compute solution remains Linux-compatible as primary runtime contract.
- Development workflow remains usable from both Linux and Windows hosts.
- Hardware-specific acceleration (e.g., Jetson class devices) is optional, not mandatory for core architecture.

## 10. Documentation Contract
The project maintains the following canonical docs:
- `readme.md`: project overview and navigation.
- `spec.md`: product and architecture solution contract.
- `testing.md`: test strategy and tracking state.