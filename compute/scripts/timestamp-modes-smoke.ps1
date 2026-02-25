Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

@'
import json
import os
import time
import urllib.request
from datetime import datetime, timezone

import psycopg

AGENT_URL = "http://mynah_agent:8002"
DATABASE_DSN = os.environ["MYNAH_DATABASE_DSN"]
RUN_ID = f"tsm_{int(time.time())}"


def post(url: str, payload: dict, timeout: int = 30) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_note(note_id: int) -> dict:
    with psycopg.connect(DATABASE_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, ts_mode, ts::text, text
                FROM memory.note
                WHERE id = %s
                LIMIT 1
                """,
                (note_id,),
            )
            row = cur.fetchone()
            if not row:
                raise RuntimeError(f"missing memory.note id={note_id}")
            return {
                "note_id": int(row[0]),
                "ts_mode": row[1],
                "ts": row[2],
                "text_preview": row[3][:100],
            }


def process_case(name: str, content: str, source_ts, day_scope: bool) -> dict:
    ingest_payload = {
        "source_type": "timestamp_mode_eval",
        "content": content,
        "upload_ts": datetime.now(timezone.utc).isoformat(),
        "source_ts": source_ts,
        "day_scope": day_scope,
        "timezone": "UTC",
        "caller": "timestamp_modes_smoke",
    }
    ingested = post(f"{AGENT_URL}/pipeline/artifacts/ingest", ingest_payload)
    artifact_id = ingested["artifact_id"]
    processed = post(
        f"{AGENT_URL}/pipeline/artifacts/process/{artifact_id}",
        {"caller": "timestamp_modes_smoke"},
        timeout=120,
    )
    if processed.get("status") != "ok":
        raise RuntimeError(f"{name} failed: {processed}")
    note_id = int(processed["note_ids"][0])
    note = fetch_note(note_id)
    return {
        "case": name,
        "artifact_id": artifact_id,
        "note_id": note_id,
        "ts_mode": note["ts_mode"],
        "ts": note["ts"],
        "text_preview": note["text_preview"],
    }


artifacts = []
results = []

try:
    exact = process_case(
        name="exact",
        content=(
            f"[run_id={RUN_ID}] I felt focused after journaling and a short walk. "
            "This is one single reflection from the wearable transcript."
        ),
        source_ts="2026-02-24T07:45:00+00:00",
        day_scope=False,
    )
    artifacts.append(exact["artifact_id"])
    results.append(exact)

    day = process_case(
        name="day",
        content=(
            f"[run_id={RUN_ID}] Day summary: stress was lower, mood more stable, "
            "and shoulder pain reduced after stretching."
        ),
        source_ts=None,
        day_scope=True,
    )
    artifacts.append(day["artifact_id"])
    results.append(day)

    inferred = process_case(
        name="inferred",
        content=(
            f"[run_id={RUN_ID}] Time hint yesterday morning. "
            "I woke up stiff and improved after tea."
        ),
        source_ts=None,
        day_scope=False,
    )
    artifacts.append(inferred["artifact_id"])
    results.append(inferred)

    upload = process_case(
        name="upload",
        content=f"[run_id={RUN_ID}] I felt calm after tea and sketching with no explicit time references.",
        source_ts=None,
        day_scope=False,
    )
    artifacts.append(upload["artifact_id"])
    results.append(upload)

    expected = {"exact": "exact", "day": "day", "inferred": "inferred", "upload": "upload"}
    for row in results:
        want = expected[row["case"]]
        if row["ts_mode"] != want:
            raise RuntimeError(f"case={row['case']} expected={want} got={row['ts_mode']}")

    print("timestamp_mode_test=PASS")
    for row in results:
        print(
            f"case={row['case']} mode={row['ts_mode']} ts={row['ts']} "
            f"artifact={row['artifact_id']} note_id={row['note_id']}"
        )

finally:
    with psycopg.connect(DATABASE_DSN, autocommit=False) as conn:
        with conn.cursor() as cur:
            for artifact_id in artifacts:
                cur.execute("DELETE FROM memory.note WHERE source_artifact_id = %s", (artifact_id,))
                cur.execute("DELETE FROM core.compaction_attempt WHERE artifact_id = %s", (artifact_id,))
                cur.execute("DELETE FROM core.ingest_artifact WHERE id = %s", (artifact_id,))
        conn.commit()
'@ | docker compose exec -T mynah_agent python -
if ($LASTEXITCODE -ne 0) { throw "timestamp modes smoke failed" }
