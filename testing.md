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

## 7. Dataset-Driven Evaluation Cycles

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

### 7.9 Execution Command
- Linux/macOS shell:
  - `sh compute/scripts/quality-eval.sh`
- Windows PowerShell:
  - `./compute/scripts/quality-eval.ps1`
- Output contract:
  - prints metrics and thresholds,
  - writes JSON report to `/home/appuser/data/artifacts/reports/quality/<run_id>.json`,
  - exits non-zero if any threshold gate fails.

### 7.10 Longitudinal Human-Transcript Cycle (200 Transcripts)
- Purpose:
  - validate month-scale trend extraction on realistic human-style transcripts with feelings, memory context, sleep/pain/stress, and activity details.
- Commands:
  - Linux/macOS shell: `sh compute/scripts/human-transcript-eval.sh`
  - Windows PowerShell: `./compute/scripts/human-transcript-eval.ps1`
- Run contract:
  - ingests 200 day-spaced transcript fixtures over multiple months,
  - transcribes and upserts note memories through the production pipeline,
  - evaluates seeded numeric trends plus LLM trend extraction checks,
  - writes one Markdown report to `reports/human-transcript-trend-report.md`,
  - cleans inserted DB rows and artifacts after run.

### 7.11 Universal Agent Tool-Pipeline Cycle (200 Transcripts, Blind Prompt)
- Purpose:
  - validate that the agent can process open human trend prompts through its own typed tool pipeline rather than direct prompt-only generation.
- Agent path under test:
  - `POST /analysis/trends`
- Run contract:
  - ingest + transcribe + memory-upsert 200 longitudinal transcripts,
  - send human prompt requesting trend detection with no embedded expected values,
  - agent infers/executes metric plan via typed operators and returns deterministic metrics,
  - compare agent metrics against script-derived ground truth with tolerance gates,
  - write Markdown report to `reports/human-transcript-trend-report.md`,
  - cleanup run rows/artifacts (`audio_note`, `transcript`, `memory_item`, `memory_citation`, `analysis_run`, `analysis_step`).

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
| E2E-001 | HR ingest to UI summary | End-to-End | Passing | 2026-02-24 | `POST /ingest/hr` -> `GET /summary/hr/today` -> UI `/status` `hr_today` |
| E2E-002 | Audio ingest to transcript/memory | End-to-End | Passing | 2026-02-24 | `POST /ingest/audio` -> `POST /pipeline/audio/transcribe` -> transcript + note memory + UI `audio_recent` |
| E2E-003 | Report generation artifact path | End-to-End | Passing | 2026-02-24 | `POST /tools/report_generate` -> `GET /tools/report_recent` -> UI `reports_recent` |
| E2E-004 | Agent SQL analysis path | End-to-End | Passing | 2026-02-24 | `POST /tools/sql_query_readonly` read path over `hr_sample` |
| E2E-005 | Citation verification gate | End-to-End | Passing | 2026-02-24 | `memory_upsert` citation minimum + `memory_verify` trusted-active checks |
| E2E-006 | Memory supersession path | End-to-End | Passing | 2026-02-24 | `supersedes_memory_id` flow + `memory_search` excludes superseded items |
| E2E-007 | SQL write-attempt rejection | End-to-End | Passing | 2026-02-24 | `DELETE ...` blocked by SQL validator |
| E2E-008 | SQL no-LIMIT rejection | End-to-End | Passing | 2026-02-24 | `SELECT` without `LIMIT` rejected |
| E2E-009 | Query audit persistence | End-to-End | Passing | 2026-02-24 | `GET /tools/query_audit/recent` shows accepted/rejected rows |
| E2E-010 | Interrupted ingest recovery | End-to-End | Passing | 2026-02-24 | `POST /ingest/audio_chunk` resume + duplicate chunk idempotency + status inspection |
| E2E-011 | Restart durability recovery | End-to-End | Passing | 2026-02-24 | daemon restart between chunk uploads still completes persisted `sync_object` |
| E2E-012 | Stale memory exclusion | End-to-End | Passing | 2026-02-24 | stale memory blocked from trusted search until `POST /tools/memory_reverify/{memory_id}` |
| QLT-001 | 100 transcript corpus ingest/run | Quality Eval | Passing | 2026-02-24 | `compute/scripts/quality-eval.*` ingests and transcribes 100 themed fixtures |
| QLT-002 | Correlation quality metrics gate | Quality Eval | Passing | 2026-02-24 | Precision/recall thresholds enforced by non-zero exit on failure |
| QLT-003 | Citation-validity quality gate | Quality Eval | Passing | 2026-02-24 | `citation_validity >= 0.98` enforced by quality harness |
| QLT-004 | False-insight and leakage gate | Quality Eval | Passing | 2026-02-24 | `false_insight_rate <= 0.10` and `stale_memory_leakage_rate == 0.00` |
| QLT-005 | 200 transcript longitudinal trend extraction | Quality Eval | Passing | 2026-02-24 | `human-transcript-eval.*` validates month-scale trends, LLM extraction, and cleanup |
| QLT-006 | Universal blind prompt trend analysis | Quality Eval | Passing | 2026-02-24 | `/analysis/trends` typed DAG output matches script ground truth on new trend set |
| HW-001 | BLE encrypted transfer on device | Hardware | Blocked | - | Awaiting connected wearable |

Status values:
- Planned
- In Progress
- Passing
- Failing
- Blocked
