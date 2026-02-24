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
This repository is currently in initial setup and specification phase.
