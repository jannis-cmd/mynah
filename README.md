# MYNAH

MYNAH is a secure, tenant-aware agent framework for building tailored agents for users and businesses.

## Status

The repository is currently in a planning and design phase. The implementation was intentionally cleared so the project can be rebuilt against a simpler and more defensible framework scope.

## Current Direction

- Tenant-aware agent runtime with hard policy boundaries
- Memory-first agents by default, with optional sealed skills
- Dedicated device adapters for multimodal use cases
- Security, sandboxing, stability, low cost, and scalability as primary design drivers

## Source of Truth

- `SPEC.md` contains the active product and architecture specification
- `AGENTS.md` contains contribution rules and repository workflow expectations

## Near-Term Focus

- define the core tenant, agent, subject, memory, skill, and device model
- define the sandboxing architecture for secure skill execution
- define the first deployment shape for shared and dedicated tenants
- define the first reference solution built on top of the framework

The repository should stay intentionally small until those decisions are locked.
