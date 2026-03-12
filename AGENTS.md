# MYNAH Repo Overview

This repository defines the planning baseline for MYNAH.

MYNAH is a secure, tenant-aware agent framework for tailored agents such as:
- horse twin agents
- receptionist agents
- later multimodal agents with custom hardware

The repository is intentionally documentation-first right now.
Use the linked files below as the starting points:

- `VISION.md` - product vision and strategic direction
- `DESIGN.md` - design direction and visual principles
- `DEVELOP.md` - development rules and working expectations
- `EVALS.md` - evaluation and experiment design rules
- `SPEC.md` - full architecture and system specification

Start simple, keep the system inspectable, and prefer explicit decisions over hidden behavior.

Spec visualization rule:
- `architecture-overview.html` is renamed to `spec-overview.html`.
- `spec-overview.html` represents the current implementation truthfully at a high level and gradually becomes more granular over time.
- When a subsystem is deep enough to inspect, its block in `spec-overview.html` should open a popup with more detail instead of linking to a new page.
- Popup detail can be nested conceptually, but the visual spec remains a single HTML artifact.
