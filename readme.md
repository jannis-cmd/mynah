# MYNAH

MYNAH is an open-source, offline-first personal intelligence system.
It ingests personal artifacts locally (wearable streams, voice transcripts, ME.md-like notes), structures them through a local LLM, and stores durable memory in a local database for on-device analysis and reporting.

Core rule: data stays local by default.

## Documentation
- Project spec: [spec.md](spec.md)
- Testing strategy and status: [testing.md](testing.md)
- Agentic memory notes: [docs/agentic-memory.md](docs/agentic-memory.md)

## Current Runtime Stack
- `mynahd` (daemon API)
- `mynah_agent` (artifact processing + structured writes)
- `mynah_ui` (local display UI)
- `postgres` (primary datastore with `pgvector`)
- `ollama` (local model serving)

Runtime decisions:
- Linux-targeted runtime.
- Development from Linux and Windows hosts.
- Local models through Ollama only.
- Agent-to-model traffic on internal Docker network only.

## Quick Start

Linux/macOS:
```bash
sh compute/scripts/stack-up.sh
```

Windows PowerShell:
```powershell
./compute/scripts/stack-up.ps1
```

Stop stack:

Linux/macOS:
```bash
sh compute/scripts/stack-down.sh
```

Windows PowerShell:
```powershell
./compute/scripts/stack-down.ps1
```

## Smoke Loop

Linux/macOS:
```bash
sh compute/scripts/e2e-smoke.sh
```

Windows PowerShell:
```powershell
./compute/scripts/e2e-smoke.ps1
```

Smoke loop currently validates:
- service health/readiness,
- HR ingest and daily summary,
- audio ingest with transcript hint,
- transcript pipeline with write-plan processing,
- report generation and listing,
- UI status aggregation from daemon + agent.

## Structured Write Pipeline
1. Store raw artifact in `artifacts`.
2. Retrieve semantically similar prior entries using pgvector.
3. Ask local LLM for strict JSON write plan.
4. Run deterministic validator.
5. If invalid, return structured validator errors to LLM and retry (bounded).
6. Commit validated plan transactionally.
7. Persist full attempt lineage in `write_plan_audit`.

This keeps the LLM as proposer and the backend as gatekeeper.

## Repository Layout

```text
mynah/
  readme.md
  spec.md
  testing.md

  docs/
  wearable/
    hardware/
    firmware/

  compute/
    daemon/mynahd/
    agent/mynah_agent/
    ui/mynah_ui/
    scripts/

  storage/
  tools/
  .github/workflows/
```

## Current Status
- Branch includes rebuilt Postgres + pgvector foundation.
- Agent memory writes now follow write-plan + validator-retry governance.
- Legacy SQLite-era quality scripts/tests are being migrated to the new pipeline.
