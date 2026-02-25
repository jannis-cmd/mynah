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
Two-step resolution:
1. Compute one artifact anchor timestamp (`exact/day/upload`):
   - `source_ts` -> `exact`
   - `day_scope=true` -> local day anchor (12:00) -> `day`
   - explicit absolute timestamp in artifact text -> `exact`
   - otherwise `upload_ts` -> `upload`
2. LLM groups text by temporal hint and returns atomic items per group:
   - each group has `hint`
   - each group has `items[]`
   - script resolves `hint + anchor_ts -> group_ts` deterministically
   - all items in that group inherit `group_ts`

LLM chooses hints, script does all timestamp math.

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
