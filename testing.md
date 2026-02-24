# Testing Strategy and Tracking

This document defines the active MYNAH testing concept for implementation.
Current priority is a tight debug loop on compute-side components, excluding wearable hardware.

## 1. Current Testing Scope
- In scope now:
  - `mynahd` (compute daemon)
  - `mynah_agent` (agent runtime)
  - `mynah_ui` (local UI)
  - local Ollama integration
  - SQLite + artifact storage behavior
- Out of scope for this phase:
  - physical wearable firmware/hardware validation
  - BLE radio behavior on real wearable

## 2. E2E-First Development Policy
- E2E tests are the primary gate for feature progress in this phase.
- Unit and integration tests support E2E but do not replace it.
- A feature is considered complete only when its E2E path is passing.

## 3. Runtime Test Environment (On Device)

### 3.1 Deployment Mode
- Run compute services in Docker on Linux target device.
- Use local Docker networking only.
- No external network dependencies for E2E tests.

### 3.2 Database and Storage (Current Decision)
- Use local SQLite DB inside Docker-managed storage for now.
- External SSD is deferred for a later phase.
- Use dedicated Docker volumes:
  - `db`
  - `artifacts`
  - `logs`

### 3.3 Test Data Sources
- Wearable data is simulated using deterministic fixtures:
  - HR fixture streams
  - audio note fixture files
  - manifest/chunk fixture payloads

## 4. E2E Test Scenarios (No Wearable Hardware)

### 4.1 Core Pipeline
- E2E-001: Simulated HR ingest -> DB persistence -> UI summary visible.
- E2E-002: Simulated audio ingest -> transcription -> memory creation -> UI note visibility.
- E2E-003: Report generation from local DB -> report artifact saved and listed.

### 4.2 Agent and Memory Trust Path
- E2E-004: Agent query -> `sql_query_readonly` -> deterministic result + narrative output.
- E2E-005: Citation verification at read-time blocks unverified memory from trusted output.
- E2E-006: Supersession flow updates conflicting memory and preserves revision history.

### 4.3 Safety and Guardrail Path
- E2E-007: Write-like SQL attempt is rejected by SQL safety policy.
- E2E-008: Query without `LIMIT` is rejected.
- E2E-009: Query audit record is written for accepted and rejected requests.

### 4.4 Resilience Path
- E2E-010: Interrupted ingest session resumes and completes idempotently.
- E2E-011: Service restart during processing preserves durable state and recovers cleanly.
- E2E-012: Stale memory excluded from trusted output until reverification.

## 5. Debug Loop (Mandatory During Development)
- Loop target:
  - modify code -> run focused E2E scenario -> inspect logs/state -> fix -> rerun.
- Each implementation PR should include:
  - scenario IDs executed,
  - pass/fail outcomes,
  - failure notes and fixes if applicable.
- Fast loop expectation:
  - at least one relevant E2E scenario run per substantial change.

## 6. Test Execution Cadence
- Per commit (local/dev):
  - relevant focused E2E scenario(s)
  - impacted unit/integration tests
- Per merge candidate:
  - full E2E set (E2E-001..E2E-012)
- Nightly/on-demand:
  - extended resilience reruns and log review

## 7. Dataset-Driven Evaluation Cycle (100 Transcripts)

### 7.1 Purpose
- Validate analytical quality and memory behavior beyond pass/fail functional checks.
- Measure whether the agent can discover meaningful correlations with verified evidence.

### 7.2 Corpus Definition
- Evaluation corpus size: 100 transcript fixtures.
- Corpus composition:
  - varied topics, routines, activities, and time spans,
  - controlled overlap themes to enable correlation testing,
  - noise/ambiguity cases to test false insight resistance.
- Data source policy:
  - anonymized or synthetic text only for shared/committed fixtures.

### 7.3 Gold Expectations
- Maintain expected labels for each transcript:
  - entities,
  - tags/themes,
  - time context.
- Maintain expected cross-item relations:
  - known correlations,
  - known non-correlations (negative controls).
- Maintain expected citation coverage:
  - each accepted derived insight should map to valid citations.

### 7.4 Evaluation Workflow
- Ingest all 100 transcripts through the same production ingestion path.
- Run memory enrichment and linking.
- Execute predefined analysis prompts/questions.
- Compare outputs to gold expectations and record metrics.

### 7.5 Quality Metrics
- Correlation precision.
- Correlation recall.
- False-insight rate.
- Citation-validity rate.
- Stale-memory leakage rate.
- End-to-end runtime for corpus pass.

### 7.6 Threshold Template (Initial)
- Correlation precision: target >= 0.75
- Correlation recall: target >= 0.70
- False-insight rate: target <= 0.10
- Citation-validity rate: target >= 0.98
- Stale-memory leakage rate: target = 0.00 in trusted outputs

Thresholds are initial defaults and should be tightened as the system matures.

### 7.7 Execution Cadence
- Not required for every local commit.
- Required on:
  - merge-candidate validation,
  - nightly scheduled run,
  - model/version changes,
  - memory/ranking/governance logic changes.

### 7.8 Regression Policy
- If any metric falls below threshold:
  - mark evaluation run as failing,
  - block merge for affected changes unless explicitly waived,
  - log root cause and corrective action.

## 8. Transparency Requirements
- Test outputs must be explicit and traceable:
  - scenario ID
  - execution time
  - result (pass/fail)
  - failure reason (if fail)
- No silent fallback behavior in test harnesses.
- Regressions must be recorded and remain visible until resolved.

## 9. Exit Criteria for This Phase
- All no-wearable E2E scenarios passing on Linux device.
- SQL safety constraints proven in E2E path.
- Memory trust/citation/supersession path passing.
- Restart/resume behavior stable under repeated runs.

## 10. Deferred Hardware Phase
- Hardware BLE/firmware tests are tracked separately and activated when wearable is connected.
- Existing E2E fixtures remain as regression baseline even after hardware integration begins.

## 11. Tracking Table

| ID | Scenario | Type | Status | Last Run | Notes |
|---|---|---|---|---|---|
| E2E-001 | HR ingest to UI summary | End-to-End | Planned | - | Simulated fixture stream |
| E2E-002 | Audio ingest to transcript/memory | End-to-End | Planned | - | Simulated audio fixture |
| E2E-003 | Report generation artifact path | End-to-End | Planned | - | Report visible in UI |
| E2E-004 | Agent SQL analysis path | End-to-End | Planned | - | `sql_query_readonly` |
| E2E-005 | Citation verification gate | End-to-End | Planned | - | Unverified memory rejected |
| E2E-006 | Memory supersession path | End-to-End | Planned | - | Revision lineage preserved |
| E2E-007 | SQL write-attempt rejection | End-to-End | Planned | - | Guardrail enforcement |
| E2E-008 | SQL no-LIMIT rejection | End-to-End | Planned | - | Policy enforcement |
| E2E-009 | Query audit persistence | End-to-End | Planned | - | Accept/reject audit entries |
| E2E-010 | Interrupted ingest recovery | End-to-End | Planned | - | Idempotent resume |
| E2E-011 | Restart durability recovery | End-to-End | Planned | - | Durable state restore |
| E2E-012 | Stale memory exclusion | End-to-End | Planned | - | Reverify required |
| QLT-001 | 100 transcript corpus ingest/run | Quality Eval | Planned | - | Dataset-driven evaluation |
| QLT-002 | Correlation quality metrics gate | Quality Eval | Planned | - | Precision/recall thresholds |
| QLT-003 | Citation-validity quality gate | Quality Eval | Planned | - | Trusted output citation check |
| QLT-004 | False-insight and leakage gate | Quality Eval | Planned | - | Hallucination/stale-memory control |
| HW-001 | BLE encrypted transfer on device | Hardware | Blocked | - | Awaiting connected wearable |

Status values:
- Planned
- In Progress
- Passing
- Failing
- Blocked
