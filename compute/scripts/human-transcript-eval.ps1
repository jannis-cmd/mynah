@'
import base64
import json
import random
import sqlite3
import statistics
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

RUN_ID = f"hlt_{int(time.time())}"
NOTE_PREFIX = f"{RUN_ID}_note_"
TOTAL_NOTES = 200
START_DAY = date(2025, 7, 1)
ARTIFACTS_ROOT = Path("/home/appuser/data/artifacts")
DB_PATH = Path("/home/appuser/data/db/mynah.db")
REPORT_PATH = ARTIFACTS_ROOT / "reports" / "quality" / "human-transcript-trend-report.md"

RNG = random.Random(4206)

FEELINGS = [
    "grounded",
    "anxious",
    "hopeful",
    "tired",
    "restless",
    "calm",
    "motivated",
    "frustrated",
]
MEMORY_SNIPPETS = [
    "I remembered an old project lesson and adjusted my routine",
    "I forgot where I left my keys and felt scattered",
    "I had a clear memory of a productive morning rhythm",
    "I kept replaying a stressful conversation from work",
    "I noticed I felt better after a short outdoor walk",
    "I remembered to stretch before bed which helped me settle",
]


def post(url: str, payload: dict, timeout: int = 20) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {url} -> {exc.code}: {body}") from exc


def clamp_int(value: float, low: int, high: int) -> int:
    return max(low, min(high, int(round(value))))


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(ys) < 2 or len(xs) != len(ys):
        return 0.0
    mx = statistics.mean(xs)
    my = statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = sum((x - mx) ** 2 for x in xs) ** 0.5
    den_y = sum((y - my) ** 2 for y in ys) ** 0.5
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / den_x / den_y


def monthly_averages(records: list[dict]) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[dict]] = {}
    for rec in records:
        key = rec["day"][:7]
        buckets.setdefault(key, []).append(rec)
    out: dict[str, dict[str, float]] = {}
    for key in sorted(buckets.keys()):
        items = buckets[key]
        out[key] = {
            "sleep_h": round(statistics.mean(r["sleep_h"] for r in items), 2),
            "pain": round(statistics.mean(r["pain"] for r in items), 2),
            "mood": round(statistics.mean(r["mood"] for r in items), 2),
            "stress": round(statistics.mean(r["stress"] for r in items), 2),
            "exercise_min": round(statistics.mean(r["exercise_min"] for r in items), 2),
            "late_caffeine_rate": round(
                sum(r["caffeine_late"] for r in items) / float(len(items)),
                2,
            ),
        }
    return out


def evaluate_llm_output(text: str) -> dict[str, bool]:
    low = text.lower()
    mapping = {
        "sleep_vs_pain": "trend_sleep_pain:",
        "caffeine_vs_sleep_stress": "trend_caffeine_sleep_stress:",
        "exercise_vs_mood_pain": "trend_exercise_mood_pain:",
        "weekend_mood_lift": "trend_weekend_mood:",
    }
    checks: dict[str, bool] = {}
    for key, token in mapping.items():
        line = ""
        for candidate in low.splitlines():
            if token in candidate:
                line = candidate.strip()
                break
        checks[key] = bool(line) and ("confirmed" in line) and ("not_confirmed" not in line)
    return checks


records: list[dict] = []
ingest_ok = 0
transcribe_ok = 0
audio_ids: list[str] = []
memory_ids: list[str] = []
analysis_output = ""
analysis_error = ""
cleanup = {"audio_deleted": 0, "transcript_deleted": 0, "memory_deleted": 0, "citation_deleted": 0}

audio_bytes = base64.b64encode(b"RIFF_HUMAN_TRANSCRIPT_EVAL").decode("ascii")

try:
    for idx in range(TOTAL_NOTES):
        day = START_DAY + timedelta(days=idx)
        progress = idx / float(TOTAL_NOTES - 1)
        weekend = 1 if day.weekday() >= 5 else 0

        caffeine_prob = max(0.10, 0.58 - (0.38 * progress))
        caffeine_late = 1 if RNG.random() < caffeine_prob else 0

        exercise_min = clamp_int(18 + (30 * progress) + (10 if weekend else 0) + RNG.uniform(-6, 6), 5, 90)
        sleep_h = round(max(4.8, min(8.5, 5.7 + (1.45 * progress) - (0.55 if caffeine_late else 0.0) + RNG.uniform(-0.3, 0.3))), 1)
        pain = clamp_int(7.3 - (2.0 * progress) - (0.018 * exercise_min) + (0.6 if sleep_h < 6.1 else 0.0) + RNG.uniform(-0.5, 0.5), 1, 9)
        stress = clamp_int(7.6 - (1.9 * progress) + (0.5 if caffeine_late else 0.0) + (0.4 if sleep_h < 6.0 else -0.2) + RNG.uniform(-0.6, 0.6), 1, 10)
        mood = clamp_int(3.7 + (2.4 * progress) + (0.03 * exercise_min) + (0.6 if weekend else 0.0) - (0.5 if sleep_h < 6.0 else 0.0) + RNG.uniform(-0.7, 0.7), 1, 10)
        wakeups = clamp_int(3.0 - (1.8 * progress) + (0.7 if caffeine_late else 0.0) + RNG.uniform(-0.5, 0.5), 0, 5)

        feeling = FEELINGS[(idx + mood + stress) % len(FEELINGS)]
        memory_line = MEMORY_SNIPPETS[(idx + pain) % len(MEMORY_SNIPPETS)]
        note_id = f"{NOTE_PREFIX}{idx:03d}"
        caller = f"{RUN_ID}_c{idx // 50}"
        start_ts = datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc).replace(hour=20, minute=15)
        end_ts = start_ts + timedelta(minutes=1)

        transcript = (
            f"Daily reflection on {day.isoformat()}. I felt {feeling} today. "
            f"Mood was {mood}/10 and stress {stress}/10. "
            f"I slept {sleep_h:.1f} hours with {wakeups} wake-ups. "
            f"Lower-back pain reached {pain}/10. "
            f"I did {exercise_min} minutes of movement and breathing work. "
            f"Late caffeine: {'yes' if caffeine_late else 'no'}. "
            f"Memory: {memory_line}. "
            f"[run_id={RUN_ID};sleep_h={sleep_h:.1f};pain={pain};mood={mood};stress={stress};"
            f"exercise_min={exercise_min};caffeine_late={caffeine_late};weekend={weekend}]"
        )

        ingest_payload = {
            "note_id": note_id,
            "device_id": "fixture_wearable_01",
            "start_ts": start_ts.isoformat(),
            "end_ts": end_ts.isoformat(),
            "audio_b64": audio_bytes,
            "transcript_hint": transcript,
            "source": "human_longitudinal_eval",
        }
        post("http://mynah_agent:8002/ingest/audio", ingest_payload, timeout=20)
        ingest_ok += 1
        audio_ids.append(note_id)

        transcribed = post(
            "http://mynah_agent:8002/pipeline/audio/transcribe",
            {"audio_id": note_id, "caller": caller, "force": True},
            timeout=20,
        )
        transcribe_ok += 1
        if transcribed.get("memory_id"):
            memory_ids.append(transcribed["memory_id"])

        records.append(
            {
                "day": day.isoformat(),
                "note_id": note_id,
                "sleep_h": sleep_h,
                "pain": pain,
                "mood": mood,
                "stress": stress,
                "exercise_min": exercise_min,
                "caffeine_late": caffeine_late,
                "weekend": weekend,
            }
        )

    month_stats = monthly_averages(records)
    sleep_vals = [r["sleep_h"] for r in records]
    pain_vals = [float(r["pain"]) for r in records]
    mood_vals = [float(r["mood"]) for r in records]
    exercise_vals = [float(r["exercise_min"]) for r in records]
    stress_vals = [float(r["stress"]) for r in records]
    caffeine_1_sleep = [r["sleep_h"] for r in records if r["caffeine_late"] == 1]
    caffeine_0_sleep = [r["sleep_h"] for r in records if r["caffeine_late"] == 0]
    caffeine_1_stress = [float(r["stress"]) for r in records if r["caffeine_late"] == 1]
    caffeine_0_stress = [float(r["stress"]) for r in records if r["caffeine_late"] == 0]
    weekend_mood = [float(r["mood"]) for r in records if r["weekend"] == 1]
    weekday_mood = [float(r["mood"]) for r in records if r["weekend"] == 0]

    metrics = {
        "sleep_pain_corr": round(pearson(sleep_vals, pain_vals), 3),
        "exercise_mood_corr": round(pearson(exercise_vals, mood_vals), 3),
        "sleep_stress_corr": round(pearson(sleep_vals, stress_vals), 3),
        "late_caffeine_sleep_delta_h": round(statistics.mean(caffeine_1_sleep) - statistics.mean(caffeine_0_sleep), 3),
        "late_caffeine_stress_delta": round(statistics.mean(caffeine_1_stress) - statistics.mean(caffeine_0_stress), 3),
        "weekend_mood_delta": round(statistics.mean(weekend_mood) - statistics.mean(weekday_mood), 3),
    }

    month_lines = []
    for month, vals in month_stats.items():
        month_lines.append(
            f"- {month}: sleep={vals['sleep_h']}h, pain={vals['pain']}, mood={vals['mood']}, "
            f"stress={vals['stress']}, exercise_min={vals['exercise_min']}, late_caffeine_rate={vals['late_caffeine_rate']}"
        )
    sample_lines = []
    for rec in records[::40]:
        sample_lines.append(
            f"- {rec['day']} sleep={rec['sleep_h']} pain={rec['pain']} mood={rec['mood']} "
            f"stress={rec['stress']} exercise={rec['exercise_min']} caffeine_late={rec['caffeine_late']}"
        )

    analyze_prompt = (
        f"You are reviewing longitudinal personal diary data for run_id {RUN_ID}. "
        "Identify concrete trends and relationships as concise bullets.\n\n"
        "Seeded expectation classes:\n"
        "1) sleep improves while pain decreases across months\n"
        "2) late caffeine worsens sleep and increases stress\n"
        "3) more exercise increases mood and lowers pain\n"
        "4) weekends show a mood lift\n\n"
        "Monthly aggregates:\n"
        + "\n".join(month_lines)
        + "\n\nComputed correlations:\n"
        + json.dumps(metrics, indent=2)
        + "\n\nSample daily rows:\n"
        + "\n".join(sample_lines)
        + "\n\nReturn exactly four lines in this exact format (no extra text):\n"
        + "TREND_SLEEP_PAIN: confirmed|not_confirmed - short evidence\n"
        + "TREND_CAFFEINE_SLEEP_STRESS: confirmed|not_confirmed - short evidence\n"
        + "TREND_EXERCISE_MOOD_PAIN: confirmed|not_confirmed - short evidence\n"
        + "TREND_WEEKEND_MOOD: confirmed|not_confirmed - short evidence"
    )
    try:
        analysis_resp = post("http://mynah_agent:8002/analyze", {"prompt": analyze_prompt}, timeout=60)
        analysis_output_raw = analysis_resp.get("response", "").strip()
        analysis_output = analysis_output_raw.encode("ascii", errors="replace").decode("ascii")
    except Exception as exc:  # noqa: BLE001
        analysis_error = str(exc).encode("ascii", errors="replace").decode("ascii")

    llm_checks = evaluate_llm_output(analysis_output if analysis_output else analysis_error)
    llm_pass = all(llm_checks.values()) and bool(analysis_output)
    month_first = month_stats[min(month_stats.keys())]
    month_last = month_stats[max(month_stats.keys())]
    trend_seed_pass = (
        month_last["sleep_h"] > month_first["sleep_h"]
        and month_last["pain"] < month_first["pain"]
        and month_last["mood"] > month_first["mood"]
        and metrics["late_caffeine_sleep_delta_h"] < 0
    )

    report_lines = [
        f"# Human Transcript Longitudinal Evaluation",
        "",
        f"- Run ID: `{RUN_ID}`",
        f"- Dataset: `{TOTAL_NOTES}` human-like daily transcripts",
        f"- Time range: `{records[0]['day']}` to `{records[-1]['day']}`",
        f"- Pipeline path: ingest audio -> transcribe -> memory upsert",
        "",
        "## Seeded Trends",
        "- Sleep improves over months while pain drops.",
        "- Late caffeine reduces sleep quality and increases stress.",
        "- More exercise improves mood and lowers pain.",
        "- Weekends should show mood lift.",
        "",
        "## Ingest Results",
        f"- Audio ingest success: `{ingest_ok}/{TOTAL_NOTES}`",
        f"- Transcribe success: `{transcribe_ok}/{TOTAL_NOTES}`",
        f"- Memory items created: `{len(memory_ids)}`",
        "",
        "## Monthly Averages",
    ]
    report_lines.extend(month_lines)
    report_lines.extend(
        [
            "",
            "## Numeric Trend Metrics",
            f"- sleep_pain_corr: `{metrics['sleep_pain_corr']}` (expected negative)",
            f"- exercise_mood_corr: `{metrics['exercise_mood_corr']}` (expected positive)",
            f"- sleep_stress_corr: `{metrics['sleep_stress_corr']}` (expected negative)",
            f"- late_caffeine_sleep_delta_h: `{metrics['late_caffeine_sleep_delta_h']}` (expected negative)",
            f"- late_caffeine_stress_delta: `{metrics['late_caffeine_stress_delta']}` (expected positive)",
            f"- weekend_mood_delta: `{metrics['weekend_mood_delta']}` (expected positive)",
            "",
            "## LLM Trend Extraction",
        ]
    )
    if analysis_output:
        report_lines.extend(["```text", analysis_output, "```"])
    else:
        report_lines.extend(["```text", analysis_error, "```"])
    report_lines.extend(
        [
            "",
            "## LLM Trend Checks",
            f"- sleep_vs_pain: `{llm_checks['sleep_vs_pain']}`",
            f"- caffeine_vs_sleep_stress: `{llm_checks['caffeine_vs_sleep_stress']}`",
            f"- exercise_vs_mood_pain: `{llm_checks['exercise_vs_mood_pain']}`",
            f"- weekend_mood_lift: `{llm_checks['weekend_mood_lift']}`",
            f"- llm_overall_pass: `{llm_pass}`",
            f"- seeded_numeric_trends_pass: `{trend_seed_pass}`",
            "",
            "## Result",
            f"- overall_pass: `{bool(llm_pass and trend_seed_pass and transcribe_ok == TOTAL_NOTES)}`",
        ]
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

finally:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        like_pattern = f"{NOTE_PREFIX}%"

        note_rows = conn.execute(
            "SELECT id, audio_path, transcript_hint_path FROM audio_note WHERE id LIKE ?",
            (like_pattern,),
        ).fetchall()
        transcript_rows = conn.execute(
            "SELECT audio_id, path FROM transcript WHERE audio_id LIKE ?",
            (like_pattern,),
        ).fetchall()
        memory_rows = conn.execute(
            "SELECT id FROM memory_item WHERE summary LIKE ?",
            (f"%{RUN_ID}%",),
        ).fetchall()
        memory_ids_to_delete = [row["id"] for row in memory_rows]

        for row in note_rows:
            for p in [row["audio_path"], row["transcript_hint_path"]]:
                if p:
                    path = Path(p)
                    if path.exists():
                        path.unlink()

        for row in transcript_rows:
            path = Path(row["path"])
            if path.exists():
                path.unlink()

        citation_deleted = 0
        revision_deleted = 0
        if memory_ids_to_delete:
            placeholders = ",".join("?" for _ in memory_ids_to_delete)
            citation_deleted = conn.execute(
                f"DELETE FROM memory_citation WHERE memory_id IN ({placeholders})",
                memory_ids_to_delete,
            ).rowcount
            revision_deleted = conn.execute(
                f"DELETE FROM memory_revision WHERE memory_id IN ({placeholders})",
                memory_ids_to_delete,
            ).rowcount
            cleanup["memory_deleted"] = conn.execute(
                f"DELETE FROM memory_item WHERE id IN ({placeholders})",
                memory_ids_to_delete,
            ).rowcount
        cleanup["citation_deleted"] = citation_deleted
        _ = revision_deleted

        cleanup["audio_deleted"] = conn.execute(
            "DELETE FROM audio_note WHERE id LIKE ?",
            (like_pattern,),
        ).rowcount
        cleanup["transcript_deleted"] = conn.execute(
            "DELETE FROM transcript WHERE audio_id LIKE ?",
            (like_pattern,),
        ).rowcount
        conn.execute(
            "DELETE FROM memory_write_audit WHERE caller LIKE ?",
            (f"{RUN_ID}%",),
        )
        conn.execute(
            "DELETE FROM agent_run_log WHERE prompt LIKE ?",
            (f"%{RUN_ID}%",),
        )
        conn.commit()

    print(f"run_id={RUN_ID}")
    print(f"report_path={REPORT_PATH}")
    print(f"cleanup={json.dumps(cleanup, sort_keys=True)}")
'@ | docker compose exec -T mynah_ui python -

New-Item -ItemType Directory -Force reports | Out-Null
docker compose cp mynah_ui:/home/appuser/data/artifacts/reports/quality/human-transcript-trend-report.md reports/human-transcript-trend-report.md | Out-Null
Write-Output "host_report_path=reports/human-transcript-trend-report.md"
