# Agentic Memory Concepts

MYNAH defines these agentic memory concepts for a local, offline personal intelligence product.

## 1. Memory Model
- Memories are explicit records, not hidden prompt state.
- A memory has:
  - Assertion (fact/insight/procedure).
  - Context.
  - Evidence citations.
  - Confidence.
  - Freshness metadata.

## 2. Evidence and Citations
- Every non-trivial memory should cite local evidence:
  - SQLite object identifiers.
  - Audio/transcript artifacts.
  - Derived report sections.
- Citations allow direct re-checking and auditability.

## 3. Retrieval vs Trust
- Retrieval finds candidate memories.
- Trust is granted only after just-in-time verification against cited evidence.
- If evidence is missing or contradictory, memory is downgraded or rejected.

## 4. Scope
- Memory scope is local-instance by default.
- Memory is not automatically shared with other systems or users.
- Exporting memory is explicit and user-controlled.

## 5. Freshness and Decay
- Memories have bounded validity windows.
- Stale memories are deprioritized and can expire.
- Re-validated memories are refreshed and remain active.

## 6. Self-Healing
- When new evidence contradicts old memory, MYNAH updates or supersedes the memory.
- Supersession history is retained so corrections are traceable.

## 7. Offline Adaptation Notes
- Verification and citation checks are fully local.
- No external network lookup is required for memory trust decisions.

## 8. Current Implementation Status (`2026-02-26`)
- Retrieval endpoints and citation-bearing responses are implemented.
- Full verification-before-trust enforcement is still in progress and not fully covered by automated tests yet.
- See `testing.md` for current covered vs uncovered test areas.
