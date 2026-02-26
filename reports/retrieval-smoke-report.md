# Retrieval Smoke Report

- Dataset state at run: memory.note=238, search.vector_index(active memory.note)=238.
- Reindex endpoint: POST /pipeline/search/reindex/memory_notes returned delta 0.
- Retrieval endpoint: POST /tools/retrieve.

## Mode: lexical
- Query: knee pain yesterday
- Result count: 5
- Top notes:
  - 230 (2025-12-03T09:00:00+00:00) Morning pain is worse.
  - 213 (2025-11-23T18:10:00+00:00) Pain in my back is persistent.
  - 206 (2025-11-20T14:30:00+00:00) Pain in my jaw is more noticeable, but manageable for now.

## Mode: hybrid
- Query: sleep stress mood trend
- Result count: 5
- Top notes:
  - 164 (2025-10-26T20:45:00+00:00) Need to find ways to de-stress before it impacts sleep tonight.
  - 143 (2025-10-12T22:00:00+00:00) Stressed again. Tired from lack of good sleep.
  - 209 (2025-11-22T20:35:00+00:00) Stress is creeping in.

## Mode: semantic
- Query: times I felt anxious after caffeine
- Result count: 5
- Top notes:
  - 3 (2025-08-01T08:35:00+00:00) Feeling a bit wired already, even with the caffeine.
  - 78 (2025-09-10T20:40:00+00:00) The caffeine helped me stay focused.
  - 163 (2025-10-25T22:00:00+00:00) Tonight was a social event and I felt wired from caffeine earlier.

## Mode: deep
- Query: what patterns explain my bad days recently
- Result count: 5
- Retrieval itself worked, but query expansion failed in this environment.
- Diagnostic: query_expansion_error = HTTP 500 Internal Server Error.
- Root cause from Ollama response: unable to load qwen3.5:35b-a3b model blob on this machine.

## Summary
- Retrieval works on ingested data for lexical, hybrid, and semantic modes.
- Derived vector index is connected and populated.
- Remaining runtime blocker is local generation model loadability for deep query expansion.
