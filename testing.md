# MYNAH Testing Strategy and Tracking

This document defines the active test concept for the Postgres + write-plan pipeline branch.
Current priority is a closed compute-side loop (no physical wearable required).

## 1. Scope

### 1.1 In Scope
- `mynahd` daemon API
- `mynah_agent` write-plan pipeline
- `mynah_ui` status/dashboard surface
- Postgres + pgvector data flow
- Ollama local-model integration

### 1.2 Out of Scope (Current Phase)
- Real wearable hardware validation
- Real BLE radio transfer validation
- Cloud and remote access scenarios

## 2. Environment Under Test
- Runtime: Docker on Linux target.
- Dev execution from Linux or Windows host.
- Network: internal Docker network only.
- Datastore: Postgres (`pgvector`) in Docker volume.
- Artifact storage: Docker volume (`artifacts`).

## 3. Test Policy
- E2E loop is primary gate during active development.
- No silent fallbacks in runtime or tests.
- A behavior change is complete only after at least one matching E2E scenario is re-run.

## 4. E2E Scenarios (Current Contract)

| ID | Scenario | Expected Result |
|---|---|---|
| E2E-001 | Service readiness (`mynahd`, `mynah_agent`, `mynah_ui`) | All healthy/ready with Postgres + Ollama model available |
| E2E-002 | HR ingest (`POST /ingest/hr`) | Samples persisted, daily summary returns correct aggregates |
| E2E-003 | Audio ingest (`POST /ingest/audio`) | Audio metadata and transcript fixture persisted |
| E2E-004 | Transcript processing (`POST /pipeline/audio/transcribe`) | Transcript row created/updated and artifact processed |
| E2E-005 | Artifact write-plan process (`POST /pipeline/artifacts/process/{id}`) | Valid plan writes entries/facts/links transactionally |
| E2E-006 | Validation retry path | Invalid LLM output triggers validator errors and bounded retry |
| E2E-007 | Retry exhaustion path | Artifact marked failed and candidate row created |
| E2E-008 | Report generation (`POST /tools/report_generate`) | Markdown report persisted and listable via `/tools/report_recent` |
| E2E-009 | UI status aggregation (`GET /status`) | UI reflects daemon + agent + summaries/reports |

## 5. Validation and Audit Requirements
- Every write-plan attempt must be recorded in `write_plan_audit`.
- Validation errors returned to LLM must be structured (code/field/reason/suggestion).
- Failed artifacts must remain auditable (`processing_state = failed`).
- Successful writes must be atomic (no partial commit).

## 6. Dataset Evaluation Tracks

### 6.1 Current State
- Prior SQLite-era quality harnesses (100/200 transcript cycles) are legacy.
- They are not authoritative for this branch until migrated to Postgres/write-plan endpoints.

### 6.2 Migration Goal
- Reintroduce corpus runs on new pipeline with:
  - seeded trend ground truth,
  - blind prompt ladder,
  - script-vs-agent metric comparison,
  - reproducible markdown/json result artifacts.

## 7. Commands

Start stack:
- Linux/macOS: `sh compute/scripts/stack-up.sh`
- Windows PowerShell: `./compute/scripts/stack-up.ps1`

Smoke loop:
- Linux/macOS: `sh compute/scripts/e2e-smoke.sh`
- Windows PowerShell: `./compute/scripts/e2e-smoke.ps1`

Stop stack:
- Linux/macOS: `sh compute/scripts/stack-down.sh`
- Windows PowerShell: `./compute/scripts/stack-down.ps1`

## 8. Tracking Table

| ID | Status | Last Run | Notes |
|---|---|---|---|
| E2E-001..E2E-005 | In Progress | 2026-02-25 | Branch rebuild in progress; scenarios being rebaselined to Postgres pipeline |
| E2E-006..E2E-007 | Planned | - | Validator retry/exhaustion assertions to be codified in smoke/integration tests |
| E2E-008..E2E-009 | In Progress | 2026-02-25 | Report and UI checks retained, now against Postgres-backed agent |
| QLT-PG-001 corpus cycle | Planned | - | Legacy quality runs require endpoint/script migration |

Status values:
- Planned
- In Progress
- Passing
- Failing
- Blocked
