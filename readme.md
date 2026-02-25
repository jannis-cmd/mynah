# MYNAH

MYNAH is an open-source, offline-first personal intelligence system.
It stores health time-series and compacted memory notes locally, then links them for on-device analysis.

## Documentation
- Project spec: [spec.md](spec.md)
- Testing strategy and status: [testing.md](testing.md)

## Simplified System Model
- `health.sample`: timestamped health measurements.
- `memory.note`: plain-text atomic memories with one timestamp and embedding.
- `memory.health_link`: links memory notes to health data by time alignment.

Core simplifications:
- single timestamp per memory note (`ts`), no `ts_end`
- no metadata column on memory notes
- same memory text at different timestamps is valid and kept
- separate generation model and embedding model

## Timestamp Framework
Priority order:
1. source timestamp
2. explicit timestamp extracted from content
3. inferred timestamp from LLM hint + deterministic resolver
4. upload timestamp fallback

`day_scope=true` forces all extracted notes to one day anchor timestamp.

## Runtime Stack
- `mynah_agent` (ingest, compaction, timestamp resolution, linking, reports)
- `mynah_ui` (local display)
- `postgres` + `pgvector`
- `db_init` (one-shot schema migration from `storage/schema.sql`)
- `ollama`

Readiness:
- `/ready` = core runtime readiness (DB + schema).
- `/ready/model` = strict model readiness (generation + embedding present).

Defaults for testing:
- generation: `qwen2.5:7b`
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
