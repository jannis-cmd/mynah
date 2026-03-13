# Next Tasks

## Current Track

Attach a minimal channel envelope to runtime-created sessions.

## Implementation

1. Add optional channel metadata to `POST /v1/sessions`.
2. Persist channel metadata on the session record.
3. Return channel metadata in the runtime-created session response.
4. Keep session ownership server-side and reject cross-user session reuse.
5. Keep the current memory core and runtime API contract stable underneath.

## After That

1. Expand the channel envelope into a small auth/channel story for the runtime API.
2. Thread channel metadata through inspect and future audit surfaces.
3. Prepare the runtime contract for later isolated execution and stronger hosted deployment boundaries.
4. Define the first boundary between the runtime service and future sandboxed execution.

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
3. Keep container smoke checks healthy:
   - runtime boots
   - `GET /healthz` responds
   - `POST /v1/sessions` and `POST /v1/chat` work against mounted local storage
