# Transcript Grouping Audit

Snapshot scope: point-in-time audit output; this file is not a full automated coverage report. See `testing.md` for current tested vs untested areas.

- Generated at (UTC): `2026-02-26T07:22:04.857239+00:00`
- Processed wearable transcripts audited: `123`
- Artifacts with accepted compaction output: `123`

## Core Metrics

| Metric | Value |
|---|---:|
| total_notes (DB) | 230 |
| total_group_items (LLM output) | 230 |
| text_alignment_mismatches | 0 |
| default_only_artifacts_with_temporal_cues | 8 |

## Distributions

### ts_mode

| ts_mode | count |
|---|---:|
| exact | 154 |
| inferred | 76 |

### note_type

| note_type | count |
|---|---:|
| observation | 230 |

### hints

| hint | count |
|---|---:|
| default | 49 |
| today | 29 |
| tonight | 20 |
| morning | 13 |
| evening | 9 |
| night | 5 |
| this morning | 5 |
| afternoon | 4 |
| now | 4 |
| tomorrow | 4 |
| last night | 3 |
| this afternoon | 3 |
| this evening | 2 |
| tomorrow morning | 2 |
| yesterday | 2 |
| yesterday evening | 1 |

### hint_quality

| quality | count |
|---|---:|
| ok | 149 |
| weak | 6 |

## Failures and Risks

### Failed/Pending Artifacts

| artifact_id | source_ts | state |
|---|---|---|
| 98f6b0b7-73fc-43f0-a3a8-fc4b179a414e | 2025-11-07 20:20:00+00 | failed |
| fc935dd5-f3bc-4cae-baaf-92036715b53d | 2025-11-28 18:35:00+00 | failed |
| 5bc21eae-3268-4b0c-80ea-198657db83d9 | 2025-12-04 09:15:00+00 | pending |

### Rejected Attempt Error Categories

| category | count |
|---|---:|
| json_parse | 15 |
| unsupported_hint | 3 |
| other | 2 |

### Weak Hint Examples

- `93d77d68-2b94-4103-a2c1-52fa960d906e` @ `2025-08-16 22:15:00+00` hint=`evening`: evening hint without cue | excerpt: Had a late meeting. Drank some water to stay hydrated, but I’m exhausted. Hoping for a good night’s sleep.
- `91663bdc-4c4d-43af-a0e2-e5363d480210` @ `2025-08-21 08:30:00+00` hint=`morning`: morning hint without morning cue | excerpt: Another day, another cup of coffee. It's 8 AM and I'm wired up already. The caffeine buzz is great for focus but I know it'll hit me hard later tonight when trying to sleep. Gotta keep an eye on that tomorrow.
- `80e3f41f-e07c-44b8-bc8c-20ff85563b36` @ `2025-08-23 18:30:00+00` hint=`this evening`: evening hint without cue | excerpt: Today was all about brainstorming sessions. My head's spinning with ideas but also feeling a bit drained. Might need to hit the gym later tonight for some stress relief. Planning on hitting that elliptical machine, yeah.
- `5722feb0-ae94-441f-b431-af40c5eaf000` @ `2025-08-24 22:35:00+00` hint=`evening`: evening hint without cue | excerpt: Just finished dinner, and it's a good one. Grilled chicken and veggies. Helps clear the mind for some serious coding tonight. But man, that caffeine is really starting to kick in now. Could be trouble sleeping later.
- `14a59aa8-c8cf-4056-9ba9-dc9c8f9e16b7` @ `2025-10-02 14:35:00+00` hint=`this afternoon`: afternoon hint without cue | excerpt: Great meeting with the team today. Actually had some cool ideas for a new project proposal. Just hope it gets approved. Decided to grab an early dinner and head home earlier tonight. Need to get good rest, feeling pretty
- `562f00a2-4be9-49dd-b142-6328a931a3e6` @ `2025-10-04 08:30:00+00` hint=`morning`: morning hint without morning cue | excerpt: Woke up with a bit of a headache. Probably from all the stress and not enough sleep last night. Decided to get an early start on some exercise, quick jog around the park. Helps clear my head and feels good. Need that boo

### Alignment Mismatch Examples

- none
