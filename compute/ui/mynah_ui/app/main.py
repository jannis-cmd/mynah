import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

SERVICE = os.getenv("MYNAH_SERVICE_NAME", "mynah_ui")
DAEMON_URL = os.getenv("MYNAH_DAEMON_URL", "http://mynahd:8001")
AGENT_URL = os.getenv("MYNAH_AGENT_URL", "http://mynah_agent:8002")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

app = FastAPI(title="mynah_ui", version="0.1.0")


def _probe(url: str) -> tuple[str, str]:
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return "up", payload.get("status", "ok")
    except (urllib.error.URLError, TimeoutError):
        return "down", "unreachable"


def _fetch_json(url: str) -> dict | None:
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": SERVICE}


@app.get("/status")
def status() -> JSONResponse:
    daemon_state, daemon_detail = _probe(f"{DAEMON_URL}/ready")
    agent_state, agent_detail = _probe(f"{AGENT_URL}/ready")
    hr_today = _fetch_json(f"{DAEMON_URL}/summary/hr/today")
    payload = {
        "service": SERVICE,
        "daemon": {"state": daemon_state, "detail": daemon_detail},
        "agent": {"state": agent_state, "detail": agent_detail},
        "hr_today": hr_today,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return JSONResponse(payload)


@app.get("/")
def home(request: Request):
    daemon_state, daemon_detail = _probe(f"{DAEMON_URL}/ready")
    agent_state, agent_detail = _probe(f"{AGENT_URL}/ready")
    hr_today = _fetch_json(f"{DAEMON_URL}/summary/hr/today") or {}
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "service": SERVICE,
            "daemon_state": daemon_state,
            "daemon_detail": daemon_detail,
            "agent_state": agent_state,
            "agent_detail": agent_detail,
            "hr_today": hr_today,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )
