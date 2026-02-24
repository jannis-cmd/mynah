import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI

SERVICE = os.getenv("MYNAH_SERVICE_NAME", "mynahd")
DB_PATH = Path(os.getenv("MYNAH_DB_PATH", "/data/db/mynah.db"))

app = FastAPI(title="mynahd", version="0.1.0")


def _init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS service_heartbeat (
                service_name TEXT PRIMARY KEY,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def _heartbeat() -> str:
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO service_heartbeat(service_name, updated_at)
            VALUES(?, ?)
            ON CONFLICT(service_name) DO UPDATE SET updated_at=excluded.updated_at
            """,
            (SERVICE, now),
        )
        conn.commit()
    return now


@app.on_event("startup")
def startup() -> None:
    _init_db()
    _heartbeat()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": SERVICE}


@app.get("/ready")
def ready() -> dict:
    ts = _heartbeat()
    return {"status": "ready", "service": SERVICE, "db_path": str(DB_PATH), "heartbeat_at": ts}


@app.get("/status")
def status() -> dict:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT updated_at FROM service_heartbeat WHERE service_name = ?",
            (SERVICE,),
        ).fetchone()
    return {"service": SERVICE, "last_heartbeat": row[0] if row else None}
