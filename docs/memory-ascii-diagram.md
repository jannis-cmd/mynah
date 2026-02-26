# MYNAH Memory System (Current) - ASCII Diagram

## 1) Ingest -> Memory Flow

```text
                +--------------------+
                |  Input Sources     |
                |--------------------|
                | wearable transcript|
                | audio + hint       |
                | health samples     |
                | chat / markdown    |
                +---------+----------+
                          |
                          v
             +--------------------------+
             | core.ingest_artifact     |
             | (raw artifact anchor)    |
             +------------+-------------+
                          |
                          v
             +--------------------------+
             | LLM temporal compaction  |
             | + script validation      |
             +------------+-------------+
                          |
          +---------------+----------------+
          |                                |
          v                                v
+----------------------+        +------------------------+
| core.compaction_attempt|      | core.extraction_failure|
| (accepted/rejected)   |      | (fail-closed records)  |
+----------------------+        +------------------------+
                          |
                          v
               +----------------------+
               | memory.note          |
               | ts, ts_mode, type,   |
               | text, embedding, src |
               +----------+-----------+
                          |
            +-------------+------------------+
            |                                |
            v                                v
+------------------------+         +-------------------------+
| memory.health_link     |         | search.vector_index     |
| (note -> health.sample)|         | (derived retrieval rows)|
+-----------+------------+         +-----------+-------------+
            |                                  |
            v                                  v
   +--------------------+             +----------------------+
   | health.sample      |             | /tools/retrieve      |
   | hrv/hr/etc         |             | lexical/semantic/... |
   +--------------------+             +----------------------+
```

## 2) Core Table Relationships

```text
core.ingest_artifact (id)
  |--< core.artifact_meta (artifact_id)
  |--< core.compaction_attempt (artifact_id)
  |--< core.extraction_failure (artifact_id)
  |--< core.open_question (artifact_id)
  |--< memory.note (source_artifact_id)
  |--< decision.entry (artifact_id)
  |--< decision.review (artifact_id)
  |--< preference.fact (artifact_id)
  |--< memory.link (artifact_id)
  \--< core.entity_alias (source_artifact_id)

core.audio_note (id)
  \--1 core.transcript (audio_id)

memory.note (id)
  |--< memory.health_link (memory_id) >-- health.sample (id)
  \--< search.vector_index (target_table='memory.note', target_id=id)

search.embedding_model (id)
  \--< search.vector_index (embedding_model_id)

decision.entry (id)
  \--< decision.review (decision_id)
```

## 3) Retrieval Path (Current)

```text
user query
   |
   v
mode = lexical | semantic | hybrid | deep
   |
   +--> lexical: memory.note text match / token fallback
   |
   +--> semantic: query embedding -> search.vector_index similarity
   |
   +--> hybrid/deep: deterministic fusion (RRF + time boost)
               |
               +--> deep: bounded query expansion (retry + JSON validation)
   |
   v
citation-bearing results
  - source_table/source_id
  - source_artifact_id
  - chunk_id
  - score components
```

## 4) Notes On Current State

- `search.vector_index` is derived from `memory.note` and can be refreshed via:
  - `POST /pipeline/search/reindex/memory_notes`
- Retrieval endpoint:
  - `POST /tools/retrieve`
- `decision.*`, `preference.*`, and `core.entity*` tables are in schema and ready, but primary active loop today is artifact -> memory.note -> retrieval.
