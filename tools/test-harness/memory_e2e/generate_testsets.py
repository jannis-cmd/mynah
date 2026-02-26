#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")

PERSONA = {
    "name": "Mara",
    "profile": "30-year-old product-minded engineer, reflective, medium technical language (5/10), concise but expressive",
    "style": "first-person spoken notes captured from a wearable throughout daily life",
}

TREND_ANCHORS = [
    "Evening caffeine tends to reduce sleep quality and increase next-day stress.",
    "Short outdoor walks improve mood and reduce perceived pain.",
    "High workload days correlate with neck/jaw tension and lower recovery.",
    "Consistent bedtime and hydration improve recovery and clarity.",
]

TOPIC_CYCLE = [
    "feelings",
    "events",
    "ideas",
    "food",
    "pain",
    "sleep",
    "stress",
    "exercise",
    "social",
    "focus",
]


@dataclass
class TranscriptSeed:
    note_id: str
    ts: datetime
    topics: list[str]


def _strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = JSON_BLOCK_RE.search(stripped)
    if not match:
        raise ValueError("no JSON object found in model output")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("model output JSON is not an object")
    return parsed


def _run_ollama_json(
    ollama_container: str,
    model: str,
    prompt: str,
    retries: int = 3,
    timeout_sec: int = 420,
) -> dict[str, Any]:
    last_error = "unknown"
    active_prompt = prompt
    for attempt in range(1, retries + 1):
        result = subprocess.run(
            ["docker", "exec", ollama_container, "ollama", "run", model, active_prompt],
            capture_output=True,
            text=False,
            timeout=timeout_sec,
            check=False,
        )
        stdout_text = (result.stdout or b"").decode("utf-8", errors="ignore")
        stderr_text = (result.stderr or b"").decode("utf-8", errors="ignore")
        if result.returncode != 0:
            last_error = f"exit={result.returncode}, stderr={stderr_text.strip()[:600]}"
            active_prompt += (
                "\n\nPrevious run failed. Return only JSON matching the required output shape."
                f"\nFailure: {last_error}\n"
            )
            continue
        raw = _strip_ansi(stdout_text)
        try:
            return _extract_json_object(raw)
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            active_prompt += (
                "\n\nPrevious run returned non-parseable JSON."
                "\nReturn one JSON object only. No markdown, no prose, no code fences."
                f"\nError: {last_error}\n"
            )
    raise RuntimeError(f"failed to get valid JSON from model after {retries} retries: {last_error}")


def _build_seeds(start_day: date, count: int, tz: ZoneInfo, seed: int) -> list[TranscriptSeed]:
    rng = random.Random(seed)
    seeds: list[TranscriptSeed] = []
    for idx in range(count):
        day = start_day + timedelta(days=idx)
        # Bias towards daytime wearable notes.
        hour = rng.choices([7, 8, 9, 12, 14, 16, 18, 20, 22], weights=[2, 3, 4, 3, 4, 4, 4, 3, 2])[0]
        minute = rng.choice([0, 5, 10, 15, 20, 30, 35, 40, 45, 50, 55])
        ts = datetime.combine(day, time(hour=hour, minute=minute), tz)
        topic_a = TOPIC_CYCLE[idx % len(TOPIC_CYCLE)]
        topic_b = TOPIC_CYCLE[(idx + 3) % len(TOPIC_CYCLE)]
        topic_c = TOPIC_CYCLE[(idx + 6) % len(TOPIC_CYCLE)]
        seeds.append(TranscriptSeed(note_id=f"voice_{idx + 1:04d}", ts=ts, topics=[topic_a, topic_b, topic_c]))
    return seeds


def _generate_transcript_batch(
    ollama_container: str,
    model: str,
    batch: list[TranscriptSeed],
    tz_name: str,
) -> tuple[list[dict[str, Any]], int]:
    seed_payload = [
        {
            "id": item.note_id,
            "ts": item.ts.isoformat(),
            "topics": item.topics,
        }
        for item in batch
    ]
    prompt = (
        "Generate wearable-style voice transcript notes for one persona.\n"
        "Return one JSON object with key `entries`.\n"
        "Output shape:\n"
        '{"entries":[{"id":"voice_0001","ts":"<use exact ts from input>","timezone":"Europe/Berlin","text":"..."}]}\n'
        "Rules:\n"
        "- Keep `id` and `ts` exactly as provided.\n"
        "- `timezone` must be exactly the provided timezone.\n"
        "- `text` must be first-person, natural spoken language, 90-170 words.\n"
        "- Technicality level 5/10: practical language, occasional technical terms.\n"
        "- Include memory-worthy details across feelings, events, ideas, food, pain, sleep, stress, exercise.\n"
        "- Do not output markdown or explanations.\n"
        "- Do not fabricate external references.\n"
        f"Persona: {json.dumps(PERSONA)}\n"
        f"Longitudinal trends to weave in probabilistically: {json.dumps(TREND_ANCHORS)}\n"
        f"timezone: {tz_name}\n"
        f"input_entries: {json.dumps(seed_payload, ensure_ascii=True)}\n"
    )
    payload = _run_ollama_json(ollama_container=ollama_container, model=model, prompt=prompt)
    entries = payload.get("entries")
    if not isinstance(entries, list) or len(entries) != len(batch):
        raise ValueError(f"invalid transcript batch size: expected={len(batch)}, got={len(entries) if isinstance(entries, list) else 'invalid'}")

    by_id = {item.note_id: item for item in batch}
    out: list[dict[str, Any]] = []
    ts_mismatch_count = 0
    for row in entries:
        if not isinstance(row, dict):
            raise ValueError("entry is not an object")
        note_id = str(row.get("id", "")).strip()
        if note_id not in by_id:
            raise ValueError(f"unexpected id from model: {note_id}")
        expected = by_id[note_id]
        ts = str(row.get("ts", "")).strip()
        if ts != expected.ts.isoformat():
            ts_mismatch_count += 1
        text = str(row.get("text", "")).strip()
        if len(text) < 60:
            raise ValueError(f"text too short for {note_id}")
        out.append(
            {
                "id": note_id,
                "ts": expected.ts.isoformat(),
                "timezone": tz_name,
                "text": text,
                "topics": expected.topics,
            }
        )
    out.sort(key=lambda x: x["id"])
    return out, ts_mismatch_count


def _copy_codex_history(sessions_root: Path, output_root: Path) -> tuple[int, int]:
    target_root = output_root / "agent_history" / "sessions"
    target_root.mkdir(parents=True, exist_ok=True)
    file_count = 0
    total_bytes = 0
    for src in sorted(sessions_root.rglob("*.jsonl")):
        rel = src.relative_to(sessions_root)
        dst = target_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        file_count += 1
        total_bytes += src.stat().st_size
    return file_count, total_bytes


def _generate_hrv(transcripts: list[dict[str, Any]], output_csv: Path, seed: int) -> int:
    rng = random.Random(seed + 101)
    rows: list[tuple[str, str, float, str, int, str]] = []
    for idx, row in enumerate(transcripts):
        ts = datetime.fromisoformat(row["ts"])
        base = 54.0 + 7.5 * math.sin(idx / 18.0)
        stress_wave = 4.0 * math.sin(idx / 7.0)
        daily_adjust = rng.uniform(-6.0, 6.0)
        # Six points around the same day as transcript.
        for h, m in [(6, 30), (9, 30), (13, 30), (17, 30), (21, 15), (23, 0)]:
            sample_ts = ts.replace(hour=h, minute=m, second=0, microsecond=0)
            value = max(18.0, min(125.0, base - stress_wave + daily_adjust + rng.uniform(-4.0, 4.0)))
            rows.append((sample_ts.isoformat(), "hrv_rmssd_ms", round(value, 2), "ms", 95, "synthetic_hrv_v1"))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ts", "metric", "value_num", "unit", "quality", "source"])
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MYNAH memory E2E datasets.")
    parser.add_argument("--output-root", default="storage/test_data/memory_e2e", help="Dataset output root")
    parser.add_argument("--sessions-root", default=str(Path.home() / ".codex" / "sessions"), help="Codex sessions root")
    parser.add_argument("--ollama-container", default="mynah-ollama-1")
    parser.add_argument("--model", default="qwen3.5:35b-a3b")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--start-date", default="2025-08-01")
    parser.add_argument("--timezone", default="UTC")
    parser.add_argument("--seed", type=int, default=20260225)
    parser.add_argument("--batch-size", type=int, default=10)
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    sessions_root = Path(args.sessions_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if not sessions_root.exists():
        raise FileNotFoundError(f"sessions root not found: {sessions_root}")

    if args.timezone.upper() == "UTC":
        tz = timezone.utc
    else:
        try:
            tz = ZoneInfo(args.timezone)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"timezone '{args.timezone}' unavailable on this host. "
                "Use --timezone UTC or install tzdata for IANA zones."
            ) from exc
    start_day = date.fromisoformat(args.start_date)
    seeds = _build_seeds(start_day=start_day, count=args.count, tz=tz, seed=args.seed)

    copied_files, copied_bytes = _copy_codex_history(sessions_root=sessions_root, output_root=output_root)

    checkpoint_path = output_root / "voice_transcripts.checkpoint.jsonl"
    transcripts: list[dict[str, Any]] = []
    ts_mismatch_total = 0
    if checkpoint_path.exists():
        for raw in checkpoint_path.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if raw:
                transcripts.append(json.loads(raw))
        transcripts.sort(key=lambda item: item["id"])
        print(f"resuming from checkpoint entries={len(transcripts)}")

    index = len(transcripts)
    for i in range(index):
        expected_id = seeds[i].note_id
        if transcripts[i]["id"] != expected_id:
            raise RuntimeError(
                f"checkpoint mismatch at index={i}: expected id={expected_id}, got={transcripts[i]['id']}"
            )
    logical_batch = 0
    while index < len(seeds):
        remaining = len(seeds) - index
        batch_size = min(args.batch_size, remaining)
        while True:
            batch = seeds[index : index + batch_size]
            try:
                generated, ts_mismatch_count = _generate_transcript_batch(
                    ollama_container=args.ollama_container,
                    model=args.model,
                    batch=batch,
                    tz_name=args.timezone,
                )
                break
            except Exception as exc:  # noqa: BLE001
                if batch_size == 1:
                    raise RuntimeError(f"failed to generate transcript for seed index={index}: {exc}") from exc
                batch_size = max(1, batch_size // 2)
                print(f"batch generation failed at index={index}, reducing batch_size to {batch_size}: {exc}")
        logical_batch += 1
        ts_mismatch_total += ts_mismatch_count
        transcripts.extend(generated)
        with checkpoint_path.open("a", encoding="utf-8") as cp_handle:
            for row in generated:
                cp_handle.write(json.dumps(row, ensure_ascii=True) + "\n")
        index += len(generated)
        print(
            f"generated transcript batch {logical_batch}: "
            f"+{len(generated)} (total={len(transcripts)}), ts_mismatch_batch={ts_mismatch_count}"
        )

    transcripts.sort(key=lambda item: item["id"])
    transcript_path = output_root / "voice_transcripts.jsonl"
    with transcript_path.open("w", encoding="utf-8") as handle:
        for row in transcripts:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    hrv_path = output_root / "hrv_samples.csv"
    hrv_count = _generate_hrv(transcripts=transcripts, output_csv=hrv_path, seed=args.seed)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "persona": PERSONA,
        "trend_anchors": TREND_ANCHORS,
        "model": args.model,
        "timezone": args.timezone,
        "transcript_count": len(transcripts),
        "transcript_start_ts": transcripts[0]["ts"] if transcripts else None,
        "transcript_end_ts": transcripts[-1]["ts"] if transcripts else None,
        "hrv_sample_count": hrv_count,
        "transcript_ts_mismatch_count_model_vs_seed": ts_mismatch_total,
        "agent_history_file_count": copied_files,
        "agent_history_total_bytes": copied_bytes,
        "paths": {
            "voice_transcripts_jsonl": str(transcript_path.relative_to(output_root)),
            "hrv_samples_csv": str(hrv_path.relative_to(output_root)),
            "agent_history_root": "agent_history/sessions",
        },
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
