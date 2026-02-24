# MYNAH

MYNAH is an open-source, offline-first personal intelligence system for wearable health and voice-note capture, local sync, persistent memory, and local reporting.

Core principle: personal data stays local by default. There is no cloud dependency in the core workflow.

## Documentation
- Project specification: [spec.md](spec.md)
- Testing strategy and tracking: [testing.md](testing.md)
- Agentic memory concepts: [docs/agentic-memory.md](docs/agentic-memory.md)

## Local Runtime (Docker)

MYNAH now includes a runtime skeleton with:
- `mynahd` (daemon service)
- `mynah_agent` (agent service, Ollama-backed)
- `mynah_ui` (local UI service)
- `ollama` (local model server)

`mynah_agent` now includes a typed trend-analysis endpoint:
- `POST /analysis/trends` (tool-oriented DAG pipeline over local transcripts with run/step lineage)

Default topology uses an internal Docker network with no published ports.

Quick start:
```bash
docker compose up -d --build
docker compose ps
```

Windows PowerShell:
```powershell
./compute/scripts/stack-up.ps1
```

Smoke test:
```bash
sh compute/scripts/e2e-smoke.sh
```

Windows PowerShell:
```powershell
./compute/scripts/e2e-smoke.ps1
```

Quality evaluation (100 transcript cycle):
```bash
sh compute/scripts/quality-eval.sh
```

Windows PowerShell:
```powershell
./compute/scripts/quality-eval.ps1
```

Longitudinal human-transcript evaluation (200 notes across multiple months):
```bash
sh compute/scripts/human-transcript-eval.sh
```

Windows PowerShell:
```powershell
./compute/scripts/human-transcript-eval.ps1
```

Smoke coverage currently includes:
- health checks for daemon/agent/ui,
- simulated HR ingest into `mynahd`,
- HR summary visibility through daemon and UI status,
- simulated audio ingest into `mynahd` with fixture transcript,
- chunked audio ingest (`/ingest/audio_chunk`) with idempotent resume checks,
- daemon restart durability checks for partial chunk sessions,
- audio transcription pipeline (`/pipeline/audio/transcribe`) and transcript visibility checks,
- note-memory creation from transcript and UI recent-note visibility,
- report generation (`/tools/report_generate`) with persisted markdown artifact checks,
- report listing visibility (`/tools/report_recent`) through UI status,
- SQL tool guardrails (`sql_query_readonly`) with accept/reject path checks,
- query audit visibility for accepted and rejected SQL requests,
- memory tool governance checks (`memory_upsert`) including citation minimum enforcement,
- memory verification/supersession path (`memory_verify`, `memory_search`),
- stale-memory exclusion checks with explicit reverification (`memory_reverify`),
- agent analyze round-trip against local Ollama.

Quality-eval coverage includes:
- 100 synthetic transcript ingest/transcribe/memory writes,
- theme correlation precision/recall gate,
- citation-validity gate,
- false-insight rate gate,
- stale-memory leakage gate,
- JSON report artifact output at `/home/appuser/data/artifacts/reports/quality/<run_id>.json`.

Longitudinal human-transcript eval coverage includes:
- 200 human-like daily transcripts (memories, feelings, sleep, pain, stress, exercise) over multiple months,
- open human-prompt trend requests executed via `/analysis/trends`,
- deterministic metric comparison against script-derived ground truth,
- guaranteed cleanup of inserted DB rows/artifacts after run,
- Markdown report output at `reports/human-transcript-trend-report.md`.

Stop:
```bash
docker compose down
```

Windows PowerShell:
```powershell
./compute/scripts/stack-down.ps1
```

## Current Direction
- Runtime target is Linux for compute services.
- Development is supported from Linux and Windows hosts.
- ESP-IDF firmware work can start as protocol and service skeletons even when hardware is not currently connected.

## Repository Structure

```text
mynah/
  readme.md
  spec.md
  testing.md

  docs/                    # Architecture, threat model, protocol and design docs

  wearable/
    hardware/              # Schematics, PCB, BOM, enclosure assets
    firmware/              # ESP-IDF project(s)

  compute/
    daemon/
      mynahd/              # Daemon runtime service (Dockerized)
    agent/
      mynah_agent/         # Agent runtime service (Dockerized)
    ui/
      mynah_ui/            # Local UI runtime service (Dockerized)
    scripts/               # Runtime helper scripts (up/down/smoke)

  storage/                 # DB schema, migrations, sample fixtures
  tools/                   # Optional CLI/test harnesses
  .github/workflows/       # CI pipelines
```

## System Summary
- Wearable captures HR and voice notes, buffers locally.
- Compute daemon performs secure BLE sync and durability checks.
- Agent builds searchable memory and report artifacts.
- UI presents status, notes, trends, and report generation locally.

## Agentic Memory Concepts
- Evidence-backed memory: memories include citations to local source artifacts.
- Just-in-time verification: memory is re-validated before being used in analysis/output.
- Scoped memory: memory remains local to this MYNAH instance unless explicitly exported.
- Self-healing updates: contradicted memories are corrected with provenance preserved.
- Freshness policy: stale/unverified memory is downgraded or expired.

## Project Status
This repository now has a runnable compute runtime skeleton with vertical slices for HR ingest/UI summary, chunked audio resume/restart durability, audio fixture ingest/transcript/memory flow, report artifact generation/listing, agent SQL guardrails/audit, memory governance/verification/supersession/freshness, a 100-transcript quality-eval gate, a 200-transcript longitudinal human-eval loop, and a universal typed trend-analysis pipeline with run/step lineage.
