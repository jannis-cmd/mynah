# MYNAH Memory Ingest Implementation and Test Plan

Status: Active (branch `experiment/memory-e2e-datasets-and-writeplan`)

## Goal
Implement and validate a full local ingest loop for three realistic input streams:
1. Agent chat history exports (Codex JSONL sessions)
2. Human wearable-style voice transcripts (200 entries with timestamps)
3. Health tracker HRV time series across the same timeline

The loop must produce structured memory notes in PostgreSQL/pgvector through the existing LLM+script pipeline and generate a reproducible test report.

## Constraints
- Follow `spec.md` as source of truth.
- Force structured JSON from the model (schema-validated).
- Keep prompts adjustable and documented.
- Use scripts for deterministic operations (timestamp math, validation, writes, metrics).
- Use LLM for semantic tasks (temporal grouping + atomic note extraction).
- No silent fallback behavior.

## Phase 1: Pipeline Hardening
- [ ] Force JSON output mode in model generation calls for compaction.
- [ ] Keep compaction schema strict and explicit.
- [ ] Extend extracted note payload to include `note_type` for typed memory rows.
- [ ] Keep retry/fail-closed behavior with compaction audit trail.
- [ ] Add/adjust unit tests for schema-driven compaction behavior.

## Phase 2: Dataset Generation
- [ ] Create dataset root: `storage/test_data/memory_e2e/`.
- [ ] Copy Codex session JSONL files from local `.codex/sessions` into dataset.
- [ ] Generate `200` human transcripts with:
  - exact timestamp per transcript
  - consistent persona and medium technicality
  - mixed content domains (feelings, events, food, pain, ideas, plans)
- [ ] Generate HRV dataset (`hrv_rmssd_ms`) over the same time window.
- [ ] Write dataset manifest with counts and time span.

## Phase 3: E2E Ingest Harness
- [ ] Build script to copy datasets into running agent container volume path.
- [ ] Ingest HRV data via `/ingest/health` in chunks.
- [ ] Ingest/process wearable transcripts via `/pipeline/artifacts/ingest` + `/pipeline/artifacts/process`.
- [ ] Ingest/process Codex history as chunked chat artifacts.
- [ ] Collect runtime metrics from API + SQL.

## Phase 4: Evaluation and Report
- [ ] Define pass criteria (no failed artifacts, expected note counts, links present, etc.).
- [ ] Produce markdown report in `reports/memory-e2e-report.md`.
- [ ] Include:
  - dataset summary
  - ingest counts by source
  - ts_mode distribution
  - note_type distribution
  - failure/retry stats
  - pass/fail outcome

## Phase 5: Iteration Until Pass
- [ ] If criteria fail, inspect compaction failures and adjust prompt/schema logic.
- [ ] Re-run full E2E loop after each meaningful change.
- [ ] Stop once criteria pass without test-specific hacks.

## Deliverables
- Pipeline code updates (agent).
- Dataset generation scripts + generated test datasets.
- E2E ingest script(s) for Linux and Windows.
- Evaluation report markdown.
- Synced docs (`spec.md`, `readme.md`, `testing.md`) reflecting behavior.
