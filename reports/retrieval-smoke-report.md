# Retrieval Smoke Report

- Dataset state: `memory.note=238`, `search.vector_index(active memory.note)=238`
- Endpoint under test: `POST /tools/retrieve`
- Reindex endpoint: `POST /pipeline/search/reindex/memory_notes`

## Mode: lexical
- Query: `knee pain yesterday`
- Result count: `5`
- Diagnostics: `query_expansion_used=false`, `candidate_limit=20`
- Top 3:
  - `230` (`2025-12-03T09:00:00+00:00`): `Morning pain is worse.`
  - `213` (`2025-11-23T18:10:00+00:00`): `Pain in my back is persistent.`
  - `206` (`2025-11-20T14:30:00+00:00`): `Pain in my jaw is more noticeable, but manageable for now.`

## Mode: hybrid
- Query: `sleep stress mood trend`
- Result count: `5`
- Diagnostics: `query_expansion_used=false`, `candidate_limit=20`
- Top 3:
  - `164` (`2025-10-26T20:45:00+00:00`): `Need to find ways to de-stress before it impacts sleep tonight.`
  - `143` (`2025-10-12T22:00:00+00:00`): `Stressed again. Tired from lack of good sleep`
  - `209` (`2025-11-22T20:35:00+00:00`): `Stress is creeping in.`

## Mode: semantic
- Query: `times I felt anxious after caffeine`
- Result count: `5`
- Diagnostics: `query_expansion_used=false`, `candidate_limit=20`
- Top 3:
  - `3` (`2025-08-01T08:35:00+00:00`): `Feeling a bit wired already, even with the caffeine.`
  - `78` (`2025-09-10T20:40:00+00:00`): `The caffeine helped me stay focused`
  - `163` (`2025-10-25T22:00:00+00:00`): `Tonight was a social event. Had a good time but felt wired from the caffeine earlier.`

## Mode: deep
- Query: `what patterns explain my bad days recently`
- Result count: `5`
- Diagnostics: `query_expansion_used=true`, `candidate_limit=20`
- Query expansions:
  - `what explains my recent bad days`
  - `patterns behind my recent bad days`
  - `what causes my recent bad days`
  - `why am I having bad days lately`
- Top 3:
  - `190` (`2025-11-10T14:50:00+00:00`): `Spent some time on a short walk, which really helped my mood.`
  - `108` (`2025-09-25T16:50:00+00:00`): `Sleep's been decent lately`
  - `194` (`2025-11-12T18:05:00+00:00`): `Gotta get back to some exercise to help clear my head and manage stress better.`

## Summary
- Expansion failure (`no JSON object`) is resolved.
- With `qwen3.5:35b-a3b` on Ollama `0.17.1`, deep mode now performs expansion and returns bounded variants.
- Retrieval behavior is stable across all four modes on the current ingested dataset.
