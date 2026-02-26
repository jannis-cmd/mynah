# MYNAH Memory Ingest Implementation and Test Plan

Status: Active and partially completed (last reviewed `2026-02-26`, branch `experiment/memory-e2e-datasets-and-writeplan`)

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

## Progress Summary

Completed:
- JSON response-format enforcement and strict compaction schema path are implemented.
- Temporal note payload includes strict `note_type`.
- Compaction retry/fail-closed behavior is implemented and unit-tested.
- Dataset tooling exists (`tools/test-harness/memory_e2e/generate_testsets.py`) and dataset root is in use.
- E2E ingest harness exists (`tools/test-harness/memory_e2e/run_ingest_and_report.py` + `compute/scripts/memory-e2e-run.*`).
- Report outputs are being generated (`reports/memory-e2e-partial-report.md`, `reports/codex-quarter-ingest-report.md`, `reports/transcript-grouping-audit.md`).

In progress:
- Full transcript + Codex ingest run to clean pass state (no failed/pending artifacts).
- Stabilizing compaction behavior on harder Codex subsets.

Not complete yet:
- Final all-green acceptance report (`reports/memory-e2e-report.md`) with all pass criteria satisfied.
- CI-gated automated execution of this full E2E harness.

## Phase Checklist

### Phase 1: Pipeline Hardening
- [x] Force JSON output mode in model generation calls for compaction.
- [x] Keep compaction schema strict and explicit.
- [x] Extend extracted note payload to include `note_type` for typed memory rows.
- [x] Keep retry/fail-closed behavior with compaction audit trail.
- [x] Add/adjust unit tests for schema-driven compaction behavior.

### Phase 2: Dataset Generation
- [x] Create dataset root: `storage/test_data/memory_e2e/`.
- [x] Copy Codex session JSONL files from local `.codex/sessions` into dataset.
- [x] Generate `200` human transcripts with timestamped entries and mixed content domains.
- [x] Generate HRV dataset (`hrv_rmssd_ms`) over the same time window.
- [x] Write dataset manifest with counts and time span.

### Phase 3: E2E Ingest Harness
- [x] Build script to copy datasets into running agent container volume path.
- [x] Ingest HRV data via `/ingest/health` in chunks.
- [x] Ingest/process wearable transcripts via `/pipeline/artifacts/ingest` + `/pipeline/artifacts/process`.
- [x] Ingest/process Codex history as chunked chat artifacts.
- [x] Collect runtime metrics from API + SQL.

### Phase 4: Evaluation and Report
- [x] Define pass criteria (no failed artifacts, expected note counts, links present, etc.).
- [x] Produce interim markdown reports under `reports/`.
- [ ] Produce final full-pass report in `reports/memory-e2e-report.md`.

### Phase 5: Iteration Until Pass
- [x] Track failures/pending work and inspect compaction outcomes.
- [x] Re-run ingestion on meaningful changes.
- [ ] Reach stable pass state without failed/pending artifacts in target run.

## Deliverables
- Pipeline code updates (agent).
- Dataset generation scripts + generated test datasets.
- E2E ingest script(s) for Linux and Windows.
- Evaluation report markdown (interim complete, final full-pass pending).
- Synced docs (`spec.md`, `readme.md`, `testing.md`) reflecting behavior.
