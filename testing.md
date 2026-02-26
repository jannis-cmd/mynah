# MYNAH Testing Strategy and Tracking

Last verified: `2026-02-26`  
Automated run: `cd compute/agent/mynah_agent && pytest -q` -> `16 passed, 2 warnings`

This document tracks what is actually tested today for the v0.9 codebase, and what is still untested.

## 1. Automated Coverage (Pytest)

### `tests/test_api_contract.py` (3 tests)
- Verifies required API route surface exists (subset contract).
- Verifies `/ready` is core-runtime readiness (not model-strict).
- Verifies `/ready/model` fails when required model state is missing.

### `tests/test_pipeline_rules.py` (11 tests)
- Anchor timestamp precedence (`source_ts` > day-scope anchor > explicit candidates > upload).
- Selected group-hint timestamp mapping (`yesterday evening`, `today`, `at 2pm`).
- Compaction retry path fails closed after max retries.
- Strict `note_type` validation on temporal extraction items.
- Retrieval helper behavior:
  - query normalization
  - recency hint detection
  - primary model text extraction
  - deep-mode query expansion retry with strict JSON expectation

### `tests/test_ble_sync.py` (2 tests)
- BLE manifest parsing for HR and audio objects.
- End-to-end BLE chunk fetch/verify/commit flow with fake transport (no hardware dependency).

## 2. Manual/Scripted Checks (Not CI-Gated)
- `compute/scripts/e2e-smoke.sh` / `.ps1`: stack health + key ingest/transcribe/report paths.
- `compute/scripts/timestamp-modes-smoke.sh` / `.ps1`: DB-backed timestamp mode smoke validation.
- `compute/scripts/wearable-ble-sync.sh` / `.ps1`: calls `POST /sync/wearable_ble` against running stack/device.
- `compute/scripts/memory-e2e-run.sh` / `.ps1`: dataset ingest harness and report generation.

These scripts are useful operator checks but are not part of the automated pytest run.

## 3. Covered vs Not Covered

Currently covered:
- Core API readiness semantics.
- Selected deterministic timestamp mapping rules.
- Compaction retry fail-closed behavior.
- Basic retrieval helper logic.
- BLE sync protocol parsing and chunk-transfer logic (fake transport).

Not covered yet (or not automated yet):
- Endpoint behavior tests for:
  - `/ingest/hr`, `/ingest/health`, `/ingest/audio`
  - `/pipeline/artifacts/ingest`, `/pipeline/artifacts/process/{artifact_id}`
  - `/pipeline/me_md/process`
  - `/pipeline/audio/transcribe`
  - `/tools/transcript/recent`
  - `/tools/report_generate`, `/tools/report_recent`
  - `/status`
  - `/sync/wearable_ble` API integration to DB write path
- Retrieval integration correctness:
  - SQL retrieval scoring/fusion correctness
  - citation payload integrity under live DB data
  - deep-mode rerank behavior and bounds
- Data lifecycle/integrity:
  - idempotency and replay checks
  - extraction failure quarantine integration (`core.extraction_failure`)
  - candidate lifecycle transitions (`candidate -> active/deprecated/retracted/rejected`)
  - `/ME` commit pointer integrity checks
- Firmware/hardware:
  - on-device BLE + sensor behavior is manual only
  - no hardware-in-the-loop automated tests

## 4. Next Test Additions
1. Add API integration tests for ingest/process/retrieval/report endpoints against a test DB.
2. Add deterministic retrieval integration tests (lexical/semantic/hybrid/deep + citation assertions).
3. Add lifecycle tests for extraction failure quarantine and candidate transitions.
4. Add idempotency/replay tests for artifact ingest.
5. Keep manual smoke scripts, but add at least one CI-safe smoke profile that runs in Docker without physical hardware.
