# MINAH

MINAH is an open-source, offline-first personal intelligence system for wearable health and voice-note capture, local sync, persistent memory, and local reporting.

Core principle: personal data stays local by default. There is no cloud dependency in the core workflow.

## Documentation
- Project specification: [spec.md](spec.md)
- Testing strategy and tracking: [testing.md](testing.md)

## Current Direction
- Runtime target is Linux for compute services.
- Development is supported from Linux and Windows hosts.
- ESP-IDF firmware work can start as protocol and service skeletons even when hardware is not currently connected.

## Repository Structure

```text
minah/
  readme.md
  spec.md
  testing.md

  docs/                    # Architecture, threat model, protocol and design docs

  wearable/
    hardware/              # Schematics, PCB, BOM, enclosure assets
    firmware/              # ESP-IDF project(s)

  compute/
    daemon/                # Local sync/orchestration service
    agent/                 # Memory + transcript/insight pipeline
    ui/                    # Local dashboard/report interface
    scripts/               # Dev/install helpers

  storage/                 # DB schema, migrations, sample fixtures
  tools/                   # Optional CLI/test harnesses
  .github/workflows/       # CI pipelines
```

## System Summary
- Wearable captures HR and voice notes, buffers locally.
- Compute daemon performs secure BLE sync and durability checks.
- Agent builds searchable memory and report artifacts.
- UI presents status, notes, trends, and report generation locally.

## Project Status
This repository is currently in initial setup and specification phase.