# Codex Quarter Ingest Report

Snapshot scope: point-in-time harness output; this file is not a full automated coverage report. See `testing.md` for current tested vs untested areas.

- Generated at (UTC): `2026-02-26T07:40:33.990407+00:00`
- Dataset root: `/home/appuser/data/artifacts/test_data/memory_e2e`
- Selected files: `6`
- Outcomes: ok=`5`, failed=`1`, skipped=`0`

## Per File Result

| file | status | artifact_id | notes_created | error |
|---|---|---|---:|---|
| 2026/02/09/rollout-2026-02-09T08-47-31-019c415e-b49e-70d2-8f5c-be6162b2eb8b.jsonl | ok | d62d24d2-813f-4818-babc-23fbeedf07f6 | 1 |  |
| 2026/02/09/rollout-2026-02-09T09-18-24-019c417a-fa32-74a3-be05-427961ba8233.jsonl | ok | b7c2730a-5706-46ad-98af-f31961d29f4e | 1 |  |
| 2026/02/11/rollout-2026-02-11T10-10-29-019c4bf7-60ff-7641-bdbc-894174348d53.jsonl | failed_closed | 786415e6-c798-45c8-8629-225004588e39 | 0 | compaction failed after 3 retries |
| 2026/02/11/rollout-2026-02-11T11-00-14-019c4c24-ecaa-7661-92fa-6c3895ecbddc.jsonl | ok | 61517e17-3590-40f5-b817-44a760b15b99 | 1 |  |
| 2026/02/11/rollout-2026-02-11T14-40-37-019c4cee-b1e5-7ab3-93b4-d67e32e3229e.jsonl | ok | bc612951-ee24-42c9-8180-f33b6e5f25dc | 4 |  |
| 2026/02/12/rollout-2026-02-12T13-03-55-019c51bc-865a-7c70-b997-d347e5f5e2ab.jsonl | ok | 5f6b3554-d805-41ae-b67f-301f9e33018c | 1 |  |

## DB Delta (status endpoint)

| metric | before | after | delta |
|---|---:|---:|---:|
| artifact_count | 126 | 132 | 6 |
| memory_note_count | 230 | 238 | 8 |
| memory_health_link_count | 264 | 272 | 8 |
| compaction_attempt_count | 143 | 155 | 12 |
