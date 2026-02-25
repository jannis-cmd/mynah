# MYNAH Testing Strategy and Tracking

This document tracks testing for the simplified MYNAH architecture.

## 1. Scope
In scope:
- health ingest and storage (`health.sample`)
- memory compaction and storage (`memory.note`)
- timestamp resolution framework (`exact | day | inferred | upload`)
- memory-to-health linking (`memory.health_link`)
- local model integration (generation + embedding)

Out of scope for now:
- hardware BLE tests on physical wearable
- cloud/remote access scenarios

## 2. Required Pipeline Checks
- Raw artifact ingest with required fields.
- Timestamp resolution precedence is enforced.
- Day-scope ingest uses one day anchor timestamp for all extracted notes.
- No cross-timestamp dedupe of memory notes.
- Memory notes written with one timestamp only (no `ts_end`).
- Links to health data follow `ts_mode` rules.
- Retry policy enforces 3 attempts then fail-closed.

## 3. Model Checks
- Generation and embedding models are separate.
- Embedding dimension is stable with DB `vector(N)`.
- Readiness fails if required configured models are unavailable.

## 4. Current Status
- Simplified spec is locked.
- Implemented checks in code:
  - unit tests for timestamp precedence and retry fail-closed logic (`test_pipeline_rules.py`)
  - runtime smoke checks for daemon ingest and summary endpoints in Docker
  - runtime smoke check for artifact process fail-closed behavior when required model is missing
- Full acceptance test suite for this model is still in progress.

## 5. Planned Acceptance Suite (Next)
1. Artifact ingest contract validation.
2. Timestamp precedence unit tests.
3. Compaction output-shape tests.
4. Day-scope batch ingest test.
5. Linking correctness tests (`exact/day/inferred/upload`).
6. Retry + fail-closed behavior tests.
7. End-to-end pipeline test on Docker runtime.
