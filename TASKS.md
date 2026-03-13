# Next Tasks

## Current Track

Package the runtime service for local container development and keep the runtime/API contract stable.

## Implementation

1. Add a Dockerfile for `mynah serve`.
2. Add a small `docker-compose` or equivalent local dev setup for the runtime service.
3. Make the container runtime use the same env vars and `.mynah/` data path contract as local CLI runs.
4. Document the HTTP runtime flow:
   - `POST /v1/agents/init`
   - `POST /v1/sessions`
   - `POST /v1/chat`
   - `GET /v1/inspect`
5. Decide whether to mount `.mynah/` directly in dev or use a named volume.
6. Keep the current memory core unchanged underneath the service boundary.

## After That

1. Add a small auth/channel envelope to the runtime API.
2. Attach channel metadata to runtime-created sessions.
3. Keep session ownership server-side and reject cross-user session reuse.
4. Prepare the runtime contract for later isolated execution and stronger hosted deployment boundaries.

## Tests

1. Keep current deterministic suites passing:
   - `./internal/runtimeapi`
   - `./internal/app`
   - `./internal/memory`
   - `./internal/storage`
   - `./cmd/mynah`
2. Keep current live OpenAI app tests passing:
   - `TestLiveOpenAIMemoryRoutingAndIsolation`
   - `TestLiveOpenAIRobustness20Variants`
   - `TestLiveOpenAIMemoryOperationContractTargetsAndUpdates`
   - `TestLiveOpenAIReplaceAndRemoveEndToEnd`
3. Add smoke checks for containerized startup:
   - runtime boots
   - `GET /healthz` responds
   - `POST /v1/sessions` and `POST /v1/chat` work against mounted local storage

