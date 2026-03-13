# MYNAH

MYNAH is a secure, tenant-aware agent framework for building tailored agents for users and businesses.

## Status

The repository is moving from planning into an initial local prototype phase.

## Current Direction

- Tenant-aware agent runtime with explicit scope boundaries
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
- `mynah serve --listen :8080`

The runtime API now exposes:

- `GET /healthz`
- `POST /v1/agents/init`
- `POST /v1/sessions`
- `POST /v1/chat`
- `GET /v1/inspect`

Environment:

- `OPENAI_API_KEY` can still be used
- otherwise MYNAH will read the key from `%USERPROFILE%\.mynah\secrets\openai_api_key`
- this is the only file-based key location MYNAH uses
- `OPENAI_BASE_URL` defaults to `https://api.openai.com/v1`
- `OPENAI_MODEL` defaults to `gpt-4.1-mini`

The CLI stores agent data in `.mynah/` by default.

Runtime session model:

- sessions are now expected to be created by the runtime through `POST /v1/sessions`
- `POST /v1/chat` expects an existing `session_id`
- `session_id` remains explicit in the API so later runtime, audit, and sandbox boundaries can rely on it cleanly

Memory contract in the current prototype:

- `AGENT_PROFILE.md` is developer-defined framing
- `MEMORY.md` is shared durable memory for the agent
- `USER.md` is durable memory for the current identified user
- `target=user` always resolves to the current session user's `USER.md`
- memory updates happen through a direct memory tool contract: `target`, `action`, `content`, and optional `old_text`
- shared facts and shared outcomes are accepted only for `MEMORY.md`
- user-specific identity and preferences are accepted only for the current user's `USER.md`
- SQLite session history is the deeper recall archive searched on demand
- memory and user-memory writes are validated before persistence
- the model chooses when to call the memory tool and whether to write to shared or user memory
- the memory store validates target, action, content safety, duplicate handling, and bounded document size before writing
- memory and user-memory writes are durable for future turns but do not alter the current turn's already-built prompt

Inspection in the current prototype:

- `show` prints `AGENT_PROFILE.md`, `MEMORY.md`, the selected user's `USER.md`, and recent session history
- this gives a quick view of what the agent currently knows

## Setup

Go is required to run the prototype.
Once Go is installed:

```powershell
go mod tidy
go run ./cmd/mynah init --tenant demo --agent bella
go run ./cmd/mynah chat --tenant demo --agent bella --user anna
go run ./cmd/mynah show --tenant demo --agent bella --user anna
```

## Container Dev

Local container dev now uses:

- [Dockerfile](/C:/Users/janni/workspace/PRIVAT/mynah/Dockerfile)
- [compose.yaml](/C:/Users/janni/workspace/PRIVAT/mynah/compose.yaml)

The compose setup:

- runs `mynah serve --listen :8080 --data .mynah`
- binds the repo-local `.mynah/` into the container for storage parity
- reads `OPENAI_API_KEY` from your shell environment

Start it with:

```powershell
$env:OPENAI_API_KEY=(Get-Content "$HOME\.mynah\secrets\openai_api_key" -Raw).Trim()
docker compose up --build
```

## Runtime API Flow

Basic runtime flow:

1. Initialize an agent:

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8080/v1/agents/init `
  -ContentType 'application/json' `
  -Body '{"tenant_id":"demo","agent_id":"bella"}'
```

2. Start a session:

```powershell
$session = Invoke-RestMethod -Method Post -Uri http://localhost:8080/v1/sessions `
  -ContentType 'application/json' `
  -Body '{"tenant_id":"demo","agent_id":"bella","user_id":"anna","source":{"type":"chat_platform","subject":"telegram:user:12345","session_ref":"telegram:chat:777"}}'
```

The `source` object is currently optional and can carry minimal session-origin metadata such as:

- `type`
- `subject`
- `session_ref`

3. Chat on that session:

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8080/v1/chat `
  -ContentType 'application/json' `
  -Body (@{
    tenant_id = 'demo'
    agent_id = 'bella'
    user_id = 'anna'
    session_id = $session.session_id
    message = 'My name is Anna and I prefer concise answers.'
  } | ConvertTo-Json)
```

4. Inspect current state:

```powershell
Invoke-RestMethod -Method Get -Uri "http://localhost:8080/v1/inspect?tenant_id=demo&agent_id=bella&user_id=anna"
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
- memory tool target and action
- when stored profile docs are ignored for being low-value or generic

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
- checks the structured memory operation contract directly, including `target`, `add`, `replace`, and `remove` behavior
- verifies update/removal scenarios by inspecting stored `MEMORY.md` and `USER.md` state, not only final reply wording
- uses broad assertions so it validates behavior without overfitting to one exact phrasing
