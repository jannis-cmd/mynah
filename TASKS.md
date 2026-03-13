# Next Tasks

## Current Track

Generalize session origin metadata into a stable source envelope.

## Implementation

1. Replace the narrow `channel` shape on `POST /v1/sessions` with a more universal `source` object.
2. Persist source metadata on the session record.
3. Return source metadata in the runtime-created session response.
4. Keep session ownership server-side and reject cross-user session reuse.
5. Keep the current memory core and runtime API contract stable underneath.

## After That

1. Decide whether source metadata should be exposed through inspect and future audit surfaces.
2. Add a small adapter contract around source normalization without overfitting to any one product channel.
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
