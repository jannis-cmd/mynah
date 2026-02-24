import json
import os
import sqlite3
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

SERVICE = os.getenv("MYNAH_SERVICE_NAME", "mynah_agent")
DB_PATH = Path(os.getenv("MYNAH_DB_PATH", "/data/db/mynah.db"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

app = FastAPI(title="mynah_agent", version="0.1.0")


class AnalyzeRequest(BaseModel):
    prompt: str


def _init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_run_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                prompt TEXT NOT NULL,
                model TEXT NOT NULL,
                response TEXT NOT NULL
            )
            """
        )
        conn.commit()


def _ollama_tags() -> dict:
    req = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/tags", method="GET")
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _model_available(model_name: str) -> bool:
    tags = _ollama_tags()
    models = [m.get("name") for m in tags.get("models", [])]
    return model_name in models


def _ollama_generate(prompt: str) -> str:
    payload = json.dumps({"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("response", "").strip()


@app.on_event("startup")
def startup() -> None:
    _init_db()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": SERVICE}


@app.get("/ready")
def ready() -> dict:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("SELECT 1").fetchone()
    try:
        tags = _ollama_tags()
    except urllib.error.URLError as exc:
        raise HTTPException(status_code=503, detail=f"ollama unavailable: {exc.reason}") from exc
    models = [m.get("name") for m in tags.get("models", [])]
    if OLLAMA_MODEL not in models:
        raise HTTPException(status_code=503, detail=f"required model missing: {OLLAMA_MODEL}")
    return {"status": "ready", "service": SERVICE, "model": OLLAMA_MODEL, "available_models": models}


@app.post("/analyze")
def analyze(req: AnalyzeRequest) -> dict:
    if not _model_available(OLLAMA_MODEL):
        raise HTTPException(status_code=503, detail=f"required model missing: {OLLAMA_MODEL}")
    response_text = _ollama_generate(req.prompt)
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO agent_run_log(created_at, prompt, model, response) VALUES(?, ?, ?, ?)",
            (now, req.prompt, OLLAMA_MODEL, response_text),
        )
        conn.commit()
    return {"model": OLLAMA_MODEL, "response": response_text}
