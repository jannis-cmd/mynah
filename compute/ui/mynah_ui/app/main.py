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
AGENT_URL = os.getenv("MYNAH_AGENT_URL", "http://mynah_agent:8002")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

app = FastAPI(title="mynah_ui", version="0.1.0")


def _probe(url: str) -> tuple[str, str]:
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return "up", payload.get("status", "ok")
    except urllib.error.HTTPError as exc:
        detail = exc.reason
        try:
            payload = json.loads(exc.read().decode("utf-8"))
            detail = payload.get("detail", detail)
        except Exception:  # noqa: BLE001
            pass
        return "down", str(detail)
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
    pipeline_state, pipeline_detail = _probe(f"{AGENT_URL}/ready")
    model_state, model_detail = _probe(f"{AGENT_URL}/ready/model")
    hr_today = _fetch_json(f"{AGENT_URL}/summary/hr/today")
    audio_recent = _fetch_json(f"{AGENT_URL}/summary/audio/recent?limit=5")
    reports_recent = _fetch_json(f"{AGENT_URL}/tools/report_recent?limit=5")
    payload = {
        "service": SERVICE,
        "pipeline": {"state": pipeline_state, "detail": pipeline_detail},
        "model": {"state": model_state, "detail": model_detail},
        "hr_today": hr_today,
        "audio_recent": audio_recent,
        "reports_recent": reports_recent,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return JSONResponse(payload)


@app.get("/")
def home(request: Request):
    pipeline_state, pipeline_detail = _probe(f"{AGENT_URL}/ready")
    model_state, model_detail = _probe(f"{AGENT_URL}/ready/model")
    hr_today = _fetch_json(f"{AGENT_URL}/summary/hr/today") or {}
    audio_recent = _fetch_json(f"{AGENT_URL}/summary/audio/recent?limit=5") or {"entries": []}
    reports_recent = _fetch_json(f"{AGENT_URL}/tools/report_recent?limit=5") or {"entries": []}
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "service": SERVICE,
            "pipeline_state": pipeline_state,
            "pipeline_detail": pipeline_detail,
            "model_state": model_state,
            "model_detail": model_detail,
            "hr_today": hr_today,
            "audio_recent": audio_recent,
            "reports_recent": reports_recent,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )
