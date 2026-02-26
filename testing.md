# MYNAH Testing Strategy and Tracking

This document tracks testing for the v0.8 MYNAH architecture.

## 1. Scope
In scope:
- artifact-centric ingest and provenance (`core.ingest_artifact`, `core.artifact_meta`)
- extraction audit + quarantine (`core.compaction_attempt`, `core.extraction_failure`)
- timestamp grouping and deterministic mapping (`exact | day | inferred | upload`)
- memory storage and linking (`memory.note`, `memory.health_link`, `memory.link`)
- decision lifecycle (`decision.entry`, `decision.review`)
- preference lifecycle and `/ME` references (`preference.fact`)
- health metric definitions (`health.metric_def`)
- vector lifecycle (`search.embedding_model`, `search.vector_index`)
- hybrid retrieval pipeline (lexical + semantic + fusion + optional rerank)
- chunking and citation integrity for retrieval results
- deterministic context assembly with profile budgets
- verification-before-trust (claim to citation mapping)

Out of scope for now:
- hardware BLE tests on physical wearable
- cloud/remote access scenarios

## 2. Required Pipeline Checks
- Raw artifact ingest with required fields and extraction version metadata.
- Temporal grouping output (`groups[].hint + groups[].items[]`) is validated and mapped deterministically to timestamps.
- Candidate lifecycle is enforced (`candidate -> active/deprecated/retracted/rejected`).
- Failed extraction outputs are quarantined in `core.extraction_failure` (no silent drop).
- Decision writes are append-safe (`decision.entry` and later `decision.review` rows).
- Preference rows include `/ME` reference integrity (`source_path`, `source_commit_hash`) when canonicalized.
- Memory notes are typed (`note_type`) and written with one timestamp only.
- Links to health data follow `ts_mode` rules.
- Vector rows support model identity and invalidation lifecycle.
- Hybrid score fusion is deterministic and reproducible.
- Query expansion/rerank outputs are bounded, cached, and invalidated correctly on model/version change.
- Retrieval responses include citation metadata (source row/path + chunk id + score components).
- Context pack assembly follows fixed slot budgets and deterministic truncation.
- Final answer claims are checked against retrieved evidence; unmatched claims are downgraded to uncertain.

## 3. Model Checks
- Runtime default generation model is `qwen3.5:35b-a3b`.
- Generation and embedding model configuration is explicit (can be same or separate).
- Embedding dimension is stable with DB vector dimensions.
- Readiness fails if required configured models are unavailable.

## 4. Current Status
- v0.8 spec is locked.
- Implemented checks in code:
  - API contract route coverage and readiness semantics tests (`test_api_contract.py`)
  - unit tests for group-hint timestamp mapping and retry fail-closed logic (`test_pipeline_rules.py`)
  - runtime smoke checks for unified agent ingest and summary endpoints in Docker
  - timestamp mode smoke test scripts (`compute/scripts/timestamp-modes-smoke.ps1`, `.sh`)
  - retrieval runtime endpoints:
    - `POST /pipeline/search/reindex/memory_notes`
    - `POST /tools/retrieve`
  - retrieval mode smoke run on ingested dataset (`lexical`, `semantic`, `hybrid`, `deep`)
- Candidate lifecycle and quarantine integration tests are next.
- Retrieval quality evaluation harness (recall@k/precision@k/citation coverage) is next.
- Context-budget determinism and claim verification audit tests are next.

## 5. Planned Acceptance Suite (Next)
1. Artifact ingest contract + idempotency tests.
2. Group-hint timestamp mapping unit tests.
3. Candidate lifecycle promotion/rejection tests.
4. Extraction failure quarantine tests.
5. Decision entry/review split integrity tests.
6. `/ME` canonical pointer integrity tests (`path + commit_hash`).
7. Health metric definition validation tests.
8. Generic link table semantics tests (`memory.link`).
9. Vector invalidation/re-index tests.
10. End-to-end Docker runtime tests.
11. Chunking boundary/overlap determinism tests.
12. Hybrid fusion reproducibility tests.
13. Deep mode expansion + rerank boundedness tests.
14. Retrieval citation integrity tests.
15. Retrieval quality benchmark tests (recall@k, precision@k, citation coverage).
16. Context profile budget/truncation determinism tests.
17. Verification-before-trust tests (claim extraction, citation mapping, uncertainty downgrade).
