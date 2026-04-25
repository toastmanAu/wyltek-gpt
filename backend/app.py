from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

import httpx
import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from backend.converters import Registry, run_conversion
from backend.host_context import host_context_block

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("local-chatbot")

ROOT = Path(__file__).resolve().parent.parent
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())
OLLAMA_URL = CONFIG["ollama"]["url"]
WORKSPACE = ROOT / "workspaces"
WORKSPACE.mkdir(exist_ok=True)

REGISTRY = Registry(CONFIG.get("converters", []))
log.info("converters: %d enabled, %d disabled", len(REGISTRY.enabled), len(REGISTRY.missing))

app = FastAPI(title="local-chatbot")


def _full_system_prompt() -> str:
    """Base prompt from config.yaml + dynamically-gathered host facts.
    Built fresh per call so date/time stay current across long-running sessions."""
    return CONFIG["assistant"]["system_prompt"] + host_context_block()


@app.get("/api/config")
async def get_config():
    return {
        "default_theme": CONFIG["ui"]["default_theme"],
        "system_prompt": _full_system_prompt(),
    }


@app.get("/api/themes")
async def list_themes():
    return sorted(p.stem for p in (ROOT / "frontend" / "themes").glob("*.css"))


@app.get("/api/models")
async def list_models():
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"{OLLAMA_URL}/api/tags")
            r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except httpx.HTTPError as e:
        raise HTTPException(503, f"Ollama unreachable at {OLLAMA_URL}: {e}")


@app.post("/api/chat")
async def chat(payload: dict):
    model = payload.get("model")
    incoming = payload.get("messages", [])
    if not model or not incoming:
        raise HTTPException(400, "Missing 'model' or 'messages'")

    # Strip any system messages the client sent and prepend a freshly-built
    # one. Backend is the single source of truth for the system prompt so
    # facts like the current date never go stale during long sessions.
    user_and_assistant = [m for m in incoming if m.get("role") != "system"]
    messages = [{"role": "system", "content": _full_system_prompt()}, *user_and_assistant]

    async def stream():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                f"{OLLAMA_URL}/api/chat",
                json={"model": model, "messages": messages, "stream": True},  # noqa: B023
            ) as r:
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    chunk = data.get("message", {}).get("content")
                    if chunk:
                        yield chunk
                    if data.get("done"):
                        return

    return StreamingResponse(stream(), media_type="text/plain")


@app.get("/api/converters")
async def list_converters():
    return {
        "enabled": [
            {
                "id": c.id,
                "from": list(c.sources),
                "to": list(c.targets),
                "params": c.params,
            }
            for c in REGISTRY.enabled
        ],
        "disabled": [{"id": cid, "missing": req} for cid, req in REGISTRY.missing],
    }


def _resolve_in_workspace(session_id: str, name: str) -> tuple[Path, Path]:
    """Returns (session_dir, file_path) both resolved and validated."""
    session_dir = (WORKSPACE / session_id).resolve()
    if WORKSPACE.resolve() not in session_dir.parents and session_dir != WORKSPACE.resolve():
        raise HTTPException(400, "invalid session_id")
    target = (session_dir / Path(name).name).resolve()
    try:
        target.relative_to(session_dir)
    except ValueError:
        raise HTTPException(400, "invalid path")
    return session_dir, target


@app.post("/api/upload")
async def upload(session_id: str = "default", file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "missing filename")
    session_dir, target = _resolve_in_workspace(session_id, file.filename)
    session_dir.mkdir(parents=True, exist_ok=True)
    target.write_bytes(await file.read())
    return {
        "name": target.name,
        "size": target.stat().st_size,
        "session_id": session_id,
    }


@app.get("/api/file-info")
async def file_info(session_id: str = "default", name: str = ""):
    if not name:
        raise HTTPException(400, "missing 'name'")
    _, target = _resolve_in_workspace(session_id, name)
    if not target.exists():
        raise HTTPException(404, "not found")
    return {"name": target.name, "size": target.stat().st_size, "session_id": session_id}


@app.post("/api/convert")
async def convert(payload: dict):
    session_id = payload.get("session_id", "default")
    name = payload.get("name")
    target_ext = (payload.get("target") or "").lstrip(".").lower()
    if not name or not target_ext:
        raise HTTPException(400, "missing 'name' or 'target'")

    session_dir, input_path = _resolve_in_workspace(session_id, name)
    if not input_path.exists():
        raise HTTPException(404, f"file not found in session workspace: {name}")

    src_ext = input_path.suffix.lstrip(".").lower()
    converter = REGISTRY.find(src_ext, target_ext)
    if not converter:
        reachable = sorted(REGISTRY.reachable_from(src_ext))
        raise HTTPException(
            400,
            f"no converter for .{src_ext} → .{target_ext}. "
            f"From .{src_ext} you can reach: {reachable or '(nothing)'}",
        )

    user_params = payload.get("params") or {}
    if not isinstance(user_params, dict):
        raise HTTPException(400, "'params' must be an object")

    try:
        out = run_conversion(converter, input_path, session_dir, target_ext, user_params)
    except subprocess.TimeoutExpired:
        raise HTTPException(504, f"{converter.id} timed out")
    except (RuntimeError, ValueError) as e:
        raise HTTPException(500, str(e))

    return {
        "name": out.name,
        "session_id": session_id,
        "size": out.stat().st_size,
        "via": converter.id,
    }


@app.get("/api/files/{session_id}/{name}")
async def download(session_id: str, name: str):
    _, target = _resolve_in_workspace(session_id, name)
    if not target.exists():
        raise HTTPException(404, "not found")
    return FileResponse(target, filename=target.name)


# ─── PWA + Web Share Target ────────────────────────────────────────────

@app.post("/share")
async def share_target(
    files: list[UploadFile] = File(default=[]),
    title: str = Form(default=""),  # noqa: ARG001 — reserved for future use
    text: str = Form(default=""),   # noqa: ARG001
    url: str = Form(default=""),    # noqa: ARG001
):
    """Receives shares from the OS share sheet (Android Chrome).
    Saves files to the default workspace and redirects the browser to
    the UI with ?shared=<comma-separated names> so the tray pre-loads."""
    session_dir = (WORKSPACE / "default").resolve()
    session_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for f in files:
        if not f.filename:
            continue
        safe_name = Path(f.filename).name
        target = session_dir / safe_name
        target.write_bytes(await f.read())
        saved.append(safe_name)
    if saved:
        return RedirectResponse(url=f"/?shared={','.join(saved)}", status_code=303)
    return RedirectResponse(url="/", status_code=303)


@app.get("/sw.js")
async def service_worker():
    # Served from root so its scope covers the whole site, not just /static/.
    return FileResponse(
        ROOT / "frontend" / "sw.js",
        media_type="application/javascript",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/manifest.json")
async def manifest():
    return FileResponse(
        ROOT / "frontend" / "manifest.json",
        media_type="application/manifest+json",
    )


app.mount("/static", StaticFiles(directory=ROOT / "frontend"), name="static")


@app.get("/")
async def index():
    return FileResponse(ROOT / "frontend" / "index.html")
