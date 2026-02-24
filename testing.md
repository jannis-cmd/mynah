# Testing Strategy and Tracking

This document defines how MINAH testing is performed and how status is tracked over time.

## 1. Testing Goals
- Verify correctness of offline sync and storage behavior.
- Protect data integrity across interrupted sessions.
- Validate Linux runtime behavior while keeping dev workflow usable on Windows.
- Ensure regressions are detected early with automated tests.
- Ensure agentic memory behavior is evidence-backed, verifiable, and auditable.

## 2. Test Levels

### 2.1 Unit Tests
Scope:
- Data model validation.
- Object integrity/hash checks.
- Memory item normalization and linking helpers.
- Report generation logic and SQL guardrails.
- Citation parser/validator behavior.
- Memory freshness/expiry policy logic.

Primary environment:
- Linux and Windows (where runtime dependencies allow).

### 2.2 Integration Tests
Scope:
- Daemon sync session lifecycle.
- Manifest/chunk/commit behavior.
- SQLite and filesystem writes.
- Agent ingest from daemon-emitted events.
- Just-in-time memory verification against local evidence.
- Memory supersession (self-healing) on contradiction.

Primary environment:
- Linux (required), Windows optional via mocks/stubs.

### 2.3 End-to-End Tests
Scope:
- Simulated wearable data to final UI/report artifacts.
- HR + voice-note ingest to transcript + memory records.
- Recovery from interrupted transfer and resume.

Primary environment:
- Linux.

### 2.4 Hardware-Dependent Tests
Scope:
- BLE pairing and encrypted transfer on physical device.
- Sensor fidelity and timing behavior.
- Power-cycle behavior and buffer durability.

Primary environment:
- Linux with connected ESP-IDF target hardware.

## 3. Non-Hardware Early Testing Approach
Until hardware is connected, use:
- Protocol simulators/mocks for wearable manifests and chunk responses.
- Fixture-based HR/audio payloads.
- Deterministic transcript/memory test doubles.

This allows compute, storage, agent, and UI behavior to progress before full hardware validation.

## 4. Minimum Acceptance Criteria (Current)
- Sync pipeline handles partial transfer and successful resume.
- No wearable object is marked wiped before commit confirmation path is complete.
- Ingested artifacts are queryable from SQLite and visible to downstream report logic.
- Linux-targeted services execute in automated CI.
- Memory-backed outputs include valid citations to local evidence.
- Unverified/stale memories are excluded from trusted output paths.

## 5. Test Matrix

| Area | Linux | Windows Dev | Hardware Required |
|---|---|---|---|
| Unit tests | Required | Recommended | No |
| Integration tests | Required | Optional | No |
| End-to-end (simulated) | Required | Optional | No |
| BLE/device validation | Required | N/A | Yes |

## 6. Tracking

Use this table to track test implementation and execution status.

| ID | Test Area | Type | Status | Last Run | Notes |
|---|---|---|---|---|---|
| T-001 | Data integrity checks | Unit | Planned | - | Hash/commit logic |
| T-002 | Sync session lifecycle | Integration | Planned | - | Manifest/chunk/commit |
| T-003 | Resume after interruption | End-to-End | Planned | - | Simulated drop/reconnect |
| T-004 | BLE encrypted transfer | Hardware | Blocked | - | Awaiting connected device |
| T-005 | Voice ingest to transcript | Integration | Planned | - | Local pipeline |
| T-006 | Citation verification path | Integration | Planned | - | Retrieval vs trust checks |
| T-007 | Memory freshness/expiry | Unit | Planned | - | Stale memory handling |
| T-008 | Self-healing supersession | Integration | Planned | - | Contradiction correction |

Status values:
- Planned
- In Progress
- Passing
- Failing
- Blocked

## 7. Reporting Expectations
- Every substantial change should include corresponding test updates.
- New behaviors should add or update at least one test case.
- Failures should be recorded in the tracking table until resolved.
