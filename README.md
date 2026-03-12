# MYNAH

MYNAH is a secure, tenant-aware agent framework for building tailored agents for users and businesses.

## Status

The repository is moving from planning into an initial local prototype phase.

## Current Direction

- Tenant-aware agent runtime with hard policy boundaries
- Memory-first agents by default, with optional sealed skills
- Dedicated device adapters for multimodal use cases
- Security, sandboxing, stability, low cost, and scalability as primary design drivers

## Source of Truth

- `SPEC.md` contains the active product and architecture specification
- `AGENTS.md` contains contribution rules and repository workflow expectations

## Near-Term Focus

- build the first local closed loop in Go
- maintain `AGENT_PROFILE.md`, shared `MEMORY.md`, per-user `USER.md`, and SQLite session history per agent
- validate OpenAI-backed response generation, scoped recall, and LLM-driven memory routing
- reject generic, low-value, or unsafe memory/profile writes and keep stored agent context high signal
- keep the prototype cheap and easy to debug before adding sandboxed skills

## Local Prototype

The first runnable target is a local CLI:

- `mynah init --tenant demo --agent bella`
- `mynah chat --tenant demo --agent bella --user anna`
- `mynah show --tenant demo --agent bella --user anna`
- `mynah eval --tenant demo --agent bella --cases evals\horse-bella.json`

Environment:

- `OPENAI_API_KEY` can still be used
- otherwise MYNAH will read the key from `%USERPROFILE%\.mynah\secrets\openai_api_key`
- this is the only file-based key location MYNAH uses
- `OPENAI_BASE_URL` defaults to `https://api.openai.com/v1`
- `OPENAI_MODEL` defaults to `gpt-4.1-mini`

The CLI stores agent data in `.mynah/` by default.

Memory contract in the current prototype:

- `AGENT_PROFILE.md` is developer-defined framing
- `MEMORY.md` is shared durable memory for the agent
- `USER.md` is durable memory for the current identified user
- post-turn routing keeps shared facts and shared outcomes in `MEMORY.md`
- post-turn routing keeps user-specific identity and preferences in the current user's `USER.md`
- SQLite session history is the deeper recall archive searched on demand
- memory and user-memory writes are validated before persistence
- memory routing is LLM-guided but corrected by deterministic post-validation routing rules
- memory and user-memory writes are durable for future turns but do not alter the current turn's already-built prompt

## Setup

Go is required to run the prototype.
Once Go is installed:

```powershell
go mod tidy
go run ./cmd/mynah init --tenant demo --agent bella
go run ./cmd/mynah chat --tenant demo --agent bella --user anna
go run ./cmd/mynah show --tenant demo --agent bella --user anna
```

## Secret Handling

Preferred local setup:

```powershell
$dir = "$HOME\.mynah\secrets"
New-Item -ItemType Directory -Force -Path $dir | Out-Null
Set-Content -Path "$dir\openai_api_key" -Value "YOUR_OPENAI_KEY"
```

This stores the key in:

- `%USERPROFILE%\.mynah\secrets\openai_api_key`

Advantages:

- works across PowerShell shells without re-exporting
- keeps the key out of the repo
- keeps the key out of command history
- avoids depending on a shell-specific session variable

Cleanup:

```powershell
Remove-Item "$HOME\.mynah\secrets\openai_api_key" -Force -ErrorAction SilentlyContinue
[Environment]::SetEnvironmentVariable('OPENAI_API_KEY', $null, 'User')
[Environment]::SetEnvironmentVariable('OPENAI_API_KEY', $null, 'Machine')
if (Test-Path Env:OPENAI_API_KEY) { Remove-Item Env:OPENAI_API_KEY }
```

## Debugging

Use `--debug` on `init`, `chat`, or `eval` to print:

- recall hits
- prompt context sizes
- assistant reply size
- memory revision reason and output sizes
- when stored memory/profile docs are ignored for being low-value or generic

Example:

```powershell
go run ./cmd/mynah chat --tenant demo --agent bella --user anna --message "What did we do yesterday?" --debug
```

## Tiny Eval Harness

The repository includes a small starter eval file:

- [evals/horse-bella.json](/Users/janni/workspace/PRIVAT/mynah/evals/horse-bella.json)

Run it with:

```powershell
go run ./cmd/mynah eval --tenant demo --agent bella --cases evals\horse-bella.json
```

## Live Smoke Tests

The default Go test suite uses deterministic fakes for repeatable contract checks.
An opt-in live smoke test also exists for the real OpenAI path:

```powershell
$env:MYNAH_LIVE_TESTS="1"
go test ./internal/app -run Live
```

The live test:

- uses a temp sandbox and cleans up after itself
- checks user-scoped memory routing and isolation with the real model
- includes a broader 20-variant robustness pass for paraphrased prompts and interleaved users
- uses broad assertions so it validates behavior without overfitting to one exact phrasing
