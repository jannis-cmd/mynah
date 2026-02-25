# MYNAH

MYNAH is an open-source, offline-first personal intelligence system.
It stores health time-series and personal memory artifacts locally, with deterministic extraction and indexing.

## Documentation
- Project spec: [spec.md](spec.md)
- Testing strategy and status: [testing.md](testing.md)

## System Model (v0.8)
- `/ME` git repo is canonical for values, policies, preferences, and curated decisions.
- PostgreSQL stores operational rows for ingest, memory, decisions, and linking.
- pgvector index is derived for semantic retrieval and can be rebuilt.

Core data layers:
- `core.*`: artifact provenance, extraction audit, quarantine
- `health.*`: measurements + metric definitions
- `memory.*`: atomic notes + links
- `decision.*`: decision entries + later reviews
- `preference.*`: preference lifecycle with `/ME` commit references
- `search.*`: embedding model registry + vector index rows

## Timestamp Framework
Two-step resolution:
1. Compute one artifact anchor timestamp (`exact/day/upload`).
2. LLM returns temporal groups (`hint + items[]`), script maps hint to timestamp deterministically.

LLM classifies/groups; script does timestamp math and writes.

## Retrieval Engine
Retrieval is hybrid by default:
- lexical retrieval for exact phrase/keyword intent
- semantic retrieval for conceptual similarity
- deterministic score fusion
- optional query expansion + rerank in deep mode

Query modes:
- `lexical`
- `semantic`
- `hybrid` (default)
- `deep`

All retrieval results are returned with citations (source row/path + chunk metadata).

## Context and Trust Controls
- Context assembly is script-owned and budgeted by profile (not model-autonomous).
- Context slots are deterministic (policy, preferences, recent decisions, evidence, optional health context).
- Final answers use verification-before-trust: claims without evidence are marked uncertain.

## Runtime Stack
- `mynah_agent` (ingest, extraction orchestration, deterministic writes, reports)
- `mynah_ui` (local display)
- `postgres` + `pgvector`
- `db_init` (schema migration from `storage/schema.sql`)
- `ollama`

Readiness:
- `/ready` = core runtime readiness (DB + schema)
- `/ready/model` = strict model readiness (generation + embedding present)

Defaults for testing:
- generation: `qwen3.5:35b-a3b`
- embedding: `qwen3-embedding:0.6b` with `OLLAMA_EMBED_DIM=1024`

## Quick Start
Linux/macOS:
```bash
sh compute/scripts/stack-up.sh
```

Windows PowerShell:
```powershell
./compute/scripts/stack-up.ps1
```

Stop:

Linux/macOS:
```bash
sh compute/scripts/stack-down.sh
```

Windows PowerShell:
```powershell
./compute/scripts/stack-down.ps1
```
