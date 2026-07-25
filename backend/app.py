from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import shutil
import sqlite3
import subprocess
from pathlib import Path

import httpx
import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from backend.bridges import cellc as cellc_bridge
from backend.bridges import open_palette
from backend.capabilities import CapabilityCache
from backend.skills import html_demo
from backend.converters import Registry, run_conversion
from backend.enhance import enhance_prompt
from backend.host_context import host_context_block
from backend.operations import (
    Operation,
    OperationRegistry,
    make_cellc_save_operation,
    make_converter_operation,
    run_local,
    validate_params,
)
from backend.output_copy import copy_to_output
from backend.sessions import SessionStore, SessionStoreError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("local-chatbot")

ROOT = Path(__file__).resolve().parent.parent
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())
OLLAMA_URL = CONFIG["ollama"]["url"]

# ─── Storage paths ────────────────────────────────────────────────────
STORAGE = CONFIG.get("storage") or {}
WORKSPACE = ROOT / STORAGE.get("workspace", "workspaces")
WORKSPACE.mkdir(exist_ok=True)
def _resolve_dropbox_path(raw: str | None, root: Path) -> Path:
    """Resolve the drop-folder path; empty/null falls back to the default."""
    path = Path(os.path.expanduser(raw or "dropbox"))
    if not path.is_absolute():
        path = root / path
    return path.resolve()


DROPBOX = _resolve_dropbox_path(STORAGE.get("dropbox"), ROOT)
try:
    DROPBOX.mkdir(parents=True, exist_ok=True)
    log.info("drop folder: %s", DROPBOX)
except OSError as exc:
    log.warning("drop folder unavailable (%s): %s", DROPBOX, exc)
_OUT_RAW = STORAGE.get("output_dir")
OUTPUT_DIR: Path | None = (
    Path(os.path.expanduser(_OUT_RAW)).resolve() if _OUT_RAW else None
)
if OUTPUT_DIR is not None:
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        log.info("output mirror: %s", OUTPUT_DIR)
    except OSError as exc:
        log.warning("output mirror unavailable (%s): %s", OUTPUT_DIR, exc)
        OUTPUT_DIR = None

DEMOS_DIR = ROOT / "demos"
try:
    DEMOS_DIR.mkdir(parents=True, exist_ok=True)
    html_demo.init_store(DEMOS_DIR)
    log.info("html_demo store: %s (render %s)",
             DEMOS_DIR, "available" if html_demo.available() else "unavailable")
except Exception as exc:  # never block boot on the demo feature
    log.warning("html_demo init failed: %s", exc)

# ─── Sessions store (server-side history) ────────────────────────────
_BG_TASKS: set = set()  # holds fire-and-forget asyncio.Task refs (e.g. auto-title) alive
SESSIONS: SessionStore | None = None
try:
    SESSIONS = SessionStore(ROOT / "data" / "sessions.db")
    log.info("sessions store: %s", ROOT / "data" / "sessions.db")
except Exception as exc:  # never block boot on the feature
    log.warning("sessions store init failed: %s", exc)
    SESSIONS = None

# ─── Converter registry (existing) ───────────────────────────────────
REGISTRY = Registry(CONFIG.get("converters", []))
log.info("converters: %d enabled, %d disabled", len(REGISTRY.enabled), len(REGISTRY.missing))

# ─── Bridges + operations registry (new) ─────────────────────────────
BRIDGES_CFG: dict = CONFIG.get("bridges") or {}


def _probe_bridges_sync() -> set[str]:
    """Health-probe every configured bridge at startup. Returns ids that responded."""
    available: set[str] = set()
    for bridge_id, cfg in BRIDGES_CFG.items():
        ok = open_palette.health_probe(
            cfg["url"], cfg.get("probe", "/"), timeout=10.0,
        )
        if ok:
            available.add(bridge_id)
            log.info("bridge '%s' available at %s", bridge_id, cfg["url"])
        else:
            log.warning("bridge '%s' not responding at %s — bridge ops disabled",
                        bridge_id, cfg["url"])
    return available


_BRIDGES_AVAILABLE = _probe_bridges_sync()
OPERATIONS = OperationRegistry(CONFIG.get("operations", []), _BRIDGES_AVAILABLE)

# Synthetic op: wraps the converter registry. Computed at startup so the
# target-extension enum reflects what's actually installed.
_REACHABLE_TARGETS: set[str] = set()
for _conv in REGISTRY.enabled:
    _REACHABLE_TARGETS.update(_conv.targets)
if _REACHABLE_TARGETS:
    OPERATIONS.add(make_converter_operation(tuple(sorted(_REACHABLE_TARGETS))))
if cellc_bridge.available():
    OPERATIONS.add(make_cellc_save_operation())

log.info("operations: %d enabled, %d disabled", len(OPERATIONS.enabled), len(OPERATIONS.missing))

# Capability cache — persisted across restarts. Probes run lazily on
# first model selection, never eagerly at startup (would force a
# 20-40min model thrash through every installed Ollama model).
CAPABILITIES = CapabilityCache(ROOT / "data" / "capabilities.json", OLLAMA_URL)

app = FastAPI(title="local-chatbot")


@app.on_event("startup")
async def _eager_probe_on_startup():
    """If the capability cache is empty (or just wiped), probe all
    installed Ollama models in the background. Cheap now that probing
    is metadata-based — ~50ms per model."""
    if CAPABILITIES.all():
        return  # already populated
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"{OLLAMA_URL}/api/tags")
            r.raise_for_status()
        names = [m["name"] for m in r.json().get("models", [])]
    except httpx.HTTPError as exc:
        log.warning("startup auto-probe skipped (Ollama unreachable): %s", exc)
        return
    log.info("auto-probing %d Ollama models on startup (metadata only)", len(names))
    await CAPABILITIES.probe_models(names)
    log.info("startup auto-probe complete")


# ─── System prompt assembly ──────────────────────────────────────────


def _operations_prompt_block() -> str:
    """Tells the model what operations exist + how to invoke via the structured fallback.

    Models with native tool-calling support get the same operation list as
    Ollama tools (see /api/chat). Models without it parse this block out
    of the system prompt and emit ``op:<name> {...}`` blocks themselves.
    """
    if not OPERATIONS.enabled:
        return ""
    lines = [
        "",
        "## File-processing operations (USE THESE — do not give shell instructions)",
        "",
        "When the user asks you to process, convert, edit, trim, or generate a file,",
        "you MUST invoke one of the operations below — never tell the user to run",
        "ffmpeg/imagemagick/etc. themselves. The system runs the operation locally",
        "and posts the resulting file back into the chat as a download link.",
        "",
        "To invoke, emit a fenced block exactly like this (one block per call):",
        "",
        "```op:<operation_id>",
        '{"param_name": "value", ...}',
        "```",
        "",
        "The user sees a Y/N/E confirm card before the operation runs, so a",
        "wrong call is recoverable. Do NOT invent operation ids or parameter",
        "names — only use what's listed here. If the user uploaded a file, its",
        "name appears in an earlier system message; pass that exact filename as",
        "the source param.",
        "",
        "Available operations:",
    ]
    for op in OPERATIONS.enabled:
        param_summary = ", ".join(
            f"{p.name}{'?' if not p.required else ''}" for p in op.params
        )
        lines.append(f"- {op.id}({param_summary}) — {op.description.strip()}")
    return "\n".join(lines)


def _full_system_prompt() -> str:
    """Base prompt + dynamic host facts + operations manifest."""
    base = (
        CONFIG["assistant"]["system_prompt"]
        + host_context_block()
        + _operations_prompt_block()
    )
    if cellc_bridge.available():
        base = base + _CELLC_PROMPT_HINT
    if html_demo.available():
        base = base + _HTMLDEMO_PROMPT_HINT
    return base


# ─── Existing endpoints (config / themes / models / chat / converters) ──


async def _titling_call(model: str, first_user: str, first_assistant: str) -> str:
    """Small non-streaming Ollama call that proposes a short title. Best-effort:
    any failure (timeout, model missing, bad response) returns "" so the caller
    falls back to the truncated-first-message title."""
    prompt = ("Give a 3-6 word title (no quotes, no punctuation at the end) for this chat:\n\n"
              f"User: {first_user[:500]}\nAssistant: {first_assistant[:500]}\nTitle:")
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(f"{OLLAMA_URL}/api/generate",
                             json={"model": model, "prompt": prompt, "stream": False,
                                   "options": {"num_predict": 24, "temperature": 0.3}})
            r.raise_for_status()
            return (r.json().get("response") or "").strip().strip('"')[:60]
    except Exception:
        return ""


async def _auto_title(sess_id: str, first_user: str, first_assistant: str) -> None:
    """Give a still-untitled session a short title. Uses auto_router.summary_model
    (fallback captioner_model, else a truncated first message). Never raises."""
    if SESSIONS is None:
        return
    try:
        full = SESSIONS.get_full(sess_id)
        if full is None or full.get("title"):
            return   # already titled or gone
        auto = CONFIG.get("auto_router") or {}
        model = auto.get("summary_model") or auto.get("captioner_model") or ""
        title = ""
        if model:
            title = await _titling_call(model, first_user, first_assistant)
        if not title:
            title = (first_user or "New chat").strip().splitlines()[0][:60]
        SESSIONS.rename(sess_id, title[:60])
    except Exception as exc:   # titling must never break chat
        log.warning("auto-title failed for %s: %s", sess_id, exc)


@app.get("/api/config")
async def get_config():
    auto = CONFIG.get("auto_router") or {}
    return {
        "default_theme": CONFIG["ui"]["default_theme"],
        "system_prompt": _full_system_prompt(),
        "auto_router": {
            "captioner_model": auto.get("captioner_model", ""),
        },
    }


_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff", ".tif")

# Plain-text formats that carry no dedicated converter: their bytes are
# already model-readable, so we inject their content into context directly
# instead of routing them through the (non-existent) converter pipeline.
_TEXT_EXTS = (
    ".txt", ".md", ".markdown", ".json", ".csv", ".tsv", ".log",
    ".xml", ".yaml", ".yml", ".ini", ".toml", ".rst",
)
# Per-file ceiling on injected text. Injected reference content shares the
# num_ctx budget with the system prompt, the model's reasoning, and its
# output — so a single attachment must not be allowed to crowd those out.
# ~60k chars ≈ ~15k tokens, comfortably under _CHAT_DEFAULT_NUM_CTX while
# still swallowing whole specs (the BIP-0039 mediawiki is ~5.5 KB).
_MAX_TEXT_INJECT_CHARS = 60_000


@app.post("/api/auto-caption")
async def auto_caption(payload: dict):
    """Specialised path: route an uploaded image + caption-y prompt to a
    dedicated captioner model regardless of which chat model the user
    has selected. Streams the captioner's response in the same format
    as /api/chat so the frontend treats it identically.

    Triggered by frontend heuristic (caption-intent regex + image file).
    Returns 503 if no captioner is configured."""
    auto = CONFIG.get("auto_router") or {}
    captioner = payload.get("captioner_model") or auto.get("captioner_model", "")
    if not captioner:
        raise HTTPException(503, "no captioner_model configured in auto_router")

    session_id = payload.get("session_id", "default")
    source = payload.get("source")
    prompt = (payload.get("prompt") or "Describe this image in detail.").strip()
    if not source:
        raise HTTPException(400, "missing 'source' (uploaded filename)")

    _, image_path = _resolve_in_workspace(session_id, source)
    if not image_path.exists():
        raise HTTPException(404, f"image not found: {source}")
    if image_path.suffix.lower() not in _IMAGE_EXTS:
        raise HTTPException(400, f"source {source} is not a supported image type")

    image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")

    body = {
        "model": captioner,
        "messages": [{"role": "user", "content": prompt, "images": [image_b64]}],
        "stream": True,
    }

    async def stream():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", f"{OLLAMA_URL}/api/chat", json=body) as r:
                if r.status_code != 200:
                    raw = (await r.aread()).decode("utf-8", errors="replace")[:500]
                    yield f"\n[captioner error: HTTP {r.status_code}: {raw}]"
                    return
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    chunk = (data.get("message") or {}).get("content")
                    if chunk:
                        yield chunk
                    if data.get("done"):
                        return

    return StreamingResponse(stream(), media_type="text/plain")


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


# Sentinel markers wrapping a reasoning model's "thinking" channel inside
# the text stream. Ollama returns chain-of-thought on a separate `thinking`
# field; we splice it into the same stream bracketed by these control-char
# markers so the frontend can render it distinctly and still scan it for
# code blocks. Control chars won't collide with model prose.
THINK_OPEN = "THINK"
THINK_CLOSE = "/THINK"

MAX_CELLC_ITERS = 5
MAX_HTMLDEMO_ITERS = 4


async def _stream_one(body: dict):
    """Stream one POST to Ollama. Yields:
        ('chunk', str)           — model text chunks
        ('tool_calls', list)     — final tool_calls payload, if any
        ('stats', dict)          — timing/token counters from Ollama's done frame
        ('error', dict)          — Ollama returned non-2xx; dict has 'status' + 'error'
        ('done', None)           — stream completed normally
    """
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", f"{OLLAMA_URL}/api/chat", json=body) as r:
            if r.status_code != 200:
                raw = (await r.aread()).decode("utf-8", errors="replace")[:500]
                try:
                    err = json.loads(raw).get("error", raw)
                except json.JSONDecodeError:
                    err = raw
                yield "error", {"status": r.status_code, "error": err}
                return
            async for line in r.aiter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msg = data.get("message", {}) or {}
                reasoning = msg.get("thinking")
                if reasoning:
                    yield "thinking", reasoning
                chunk = msg.get("content")
                if chunk:
                    yield "chunk", chunk
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    yield "tool_calls", tool_calls
                if data.get("done"):
                    # Surface Ollama's perf counters to the UI stats bar.
                    # All durations are nanoseconds; the frontend converts.
                    stats = {
                        "model": data.get("model") or body.get("model"),
                        "load_ns": data.get("load_duration", 0),
                        "prompt_eval_count": data.get("prompt_eval_count", 0),
                        "prompt_eval_ns": data.get("prompt_eval_duration", 0),
                        "eval_count": data.get("eval_count", 0),
                        "eval_ns": data.get("eval_duration", 0),
                        "total_ns": data.get("total_duration", 0),
                    }
                    yield "stats", stats
                    yield "done", None
                    return
            yield "done", None


def _partition_tool_calls(tool_calls):
    """Split a tool_calls list into (cellc_calls, demo_calls, op_calls)."""
    cellc_calls, demo_calls, op_calls = [], [], []
    for tc in tool_calls or []:
        name = (tc.get("function") or {}).get("name")
        if name in cellc_bridge.CELLC_TOOL_NAMES:
            cellc_calls.append(tc)
        elif name in html_demo.HTML_DEMO_TOOL_NAMES:
            demo_calls.append(tc)
        else:
            op_calls.append(tc)
    return cellc_calls, demo_calls, op_calls


def _summarize_cellc_step(name, result):
    """Return a short human-readable summary of a cellc tool result."""
    if result.get("tool_error"):
        return f"\u26a0 {name}: {result.get('stderr', 'error')[:80]}"
    if name == "cellc_check":
        if result.get("ok"):
            return "cellc_check \u2192 \u2713 passed"
        lines = ", ".join(f"L{d.get('line')}" for d in (result.get("diagnostics") or [])[:3])
        return f"cellc_check \u2192 \u2717 {result.get('error_count', 0)} error(s) ({lines})"
    if name == "cellc_explain":
        return f"cellc_explain \u2192 {result.get('ecode', '')}"
    if name == "cellc_metadata":
        res = result.get("resources_count", 0)
        act = result.get("actions_count", 0)
        return f"cellc_metadata \u2192 {res} resources, {act} actions"
    if name == "cellc_list_examples":
        n = len(result.get("examples", []))
        return f"cellc_list_examples \u2192 {n} examples"
    return f"{name} \u2192 ok"




@app.post("/api/chat")
async def chat(payload: dict):
    """Streaming chat. Forwards Ollama tool definitions when available so
    capable models can emit native tool_calls; less-capable models emit
    structured ``op:<name> {...}`` blocks parsed client-side instead.

    If ``image_files`` is supplied (list of workspace filenames), the bytes
    are base64-encoded and attached to the last user message's ``images``
    field — Ollama's multimodal-vision contract. The frontend is expected
    to only send images when the active model has the vision capability."""
    model = payload.get("model")
    incoming = payload.get("messages", [])
    image_files = payload.get("image_files") or []
    text_files = payload.get("text_files") or []
    session_id = payload.get("session_id", "default")
    if not model or not incoming:
        raise HTTPException(400, "Missing 'model' or 'messages'")

    user_and_assistant = [m for m in incoming if m.get("role") != "system"]
    messages = [{"role": "system", "content": _full_system_prompt()}, *user_and_assistant]

    sess_id = payload.get("chat_session_id")   # sessions store key; NOT the workspace session_id
    if SESSIONS is not None and sess_id:
        last_user = next((m.get("content", "") for m in reversed(incoming)
                          if m.get("role") == "user"), "")
        try:
            SESSIONS.append_message(sess_id, "user", last_user)
            history = SESSIONS.load_context(sess_id)     # [summary?] + prior turns incl. this user msg
            messages = [{"role": "system", "content": _full_system_prompt()}, *history]
        except (SessionStoreError, sqlite3.Error) as exc:
            log.warning("session %s: context load failed, falling back to legacy assembly: %s", sess_id, exc)
            sess_id = None   # unknown session or store failure -> fall back to legacy assembly below

    cellc_chat = False
    if cellc_bridge.available():
        last_user = next((m.get("content", "") for m in reversed(incoming) if m.get("role") == "user"), "")
        cellc_chat = _inject_cellc_context(messages, last_user)

    html_demo_chat = False
    if html_demo.available():
        last_user = next((m.get("content", "") for m in reversed(incoming)
                          if m.get("role") == "user"), "")
        html_demo_chat = _inject_html_demo_context(messages, last_user)

    if text_files:
        injected = _inject_text_files(messages, session_id, text_files)
        if injected:
            log.info("chat: injected %d text file(s) into context for %s", injected, model)

    if image_files:
        images_b64: list[str] = []
        for name in image_files:
            try:
                _, path = _resolve_in_workspace(session_id, name)
            except (ValueError, HTTPException) as exc:
                log.warning("chat: cannot resolve image %r: %s", name, exc)
                continue
            if not path.exists() or path.suffix.lower() not in _IMAGE_EXTS:
                log.warning("chat: skipping non-image or missing file %r", name)
                continue
            images_b64.append(base64.b64encode(path.read_bytes()).decode("ascii"))
        if images_b64:
            for m in reversed(messages):
                if m.get("role") == "user":
                    m["images"] = images_b64
                    log.info("chat: attached %d image(s) to %s", len(images_b64), model)
                    break

    tools = OPERATIONS.tool_schemas() if OPERATIONS.enabled else []
    if cellc_bridge.available():
        tools = tools + cellc_bridge.tool_schemas()
    if html_demo.available():
        tools = tools + html_demo.tool_schemas()
    # Ollama defaults num_ctx to ~4096 regardless of the model's real window,
    # which truncates multi-turn agentic loops (system prompt + tool results +
    # the model's own reasoning). Give the loop room; caller may override.
    _options = {"num_ctx": int(payload.get("num_ctx") or _CHAT_DEFAULT_NUM_CTX)}
    _caps = CAPABILITIES.get(model) or {}
    _is_vision = bool(_caps.get("vision"))
    body_with_tools: dict = {"model": model, "messages": messages, "stream": True, "options": _options}
    if tools:
        body_with_tools["tools"] = tools
    body_without_tools: dict = {"model": model, "messages": messages, "stream": True, "options": _options}
    # Reasoning models (e.g. qwen3.6) over-deliberate on hard CellScript tasks —
    # they draft the whole contract inside the <think> channel and exhaust the
    # generation budget before ever emitting a cellc_check tool call. Bounded
    # thinking ("low") keeps enough reasoning to ground on the reference while
    # forcing the model to ACT (call tools) so the write→check→fix loop runs.
    # Only for cellc chats; normal chats keep the model's default thinking.
    if cellc_chat:
        think = payload.get("think", "low")
        body_with_tools["think"] = think
        body_without_tools["think"] = think


    async def stream():
        # First attempt: with tools. If Ollama rejects (the model doesn't
        # support function-calling), retry without — the structured ``op:``
        # fence parser still works, that's the whole point of the hybrid.
        # `in_thinking` tracks whether we're inside a reasoning run so we can
        # bracket it with sentinels; it survives the tools→no-tools retry
        # because `relay` recurses with the same nonlocal state.
        attempted_fallback = False
        in_thinking = False
        assistant_parts: list[str] = []
        final_eval = {"tokens": None}

        async def relay(source, messages, iterations):
            nonlocal attempted_fallback, in_thinking
            async for kind, value in source:
                if kind == "thinking":
                    if not in_thinking:
                        in_thinking = True
                        yield THINK_OPEN
                    yield value
                elif kind == "chunk":
                    if in_thinking:
                        in_thinking = False
                        yield THINK_CLOSE
                    assistant_parts.append(value)
                    yield value
                elif kind == "tool_calls":
                    cellc_calls, demo_calls, op_calls = _partition_tool_calls(value)
                    if op_calls:
                        yield "\n" + json.dumps({"__tool_calls__": op_calls})
                    if demo_calls and iterations < MAX_HTMLDEMO_ITERS:
                        if in_thinking:
                            in_thinking = False
                            yield THINK_CLOSE
                        tool_msgs = []
                        image_msgs = []
                        for c in demo_calls:
                            name = c["function"]["name"]
                            args = dict(c["function"].get("arguments") or {})
                            args["_want_screenshot"] = _is_vision
                            try:
                                result = await asyncio.to_thread(html_demo.dispatch, name, args)
                            except Exception as exc:
                                result = {"tool_error": True, "message": str(exc)}
                            shot = result.pop("_screenshot_b64", None)
                            yield "\n" + json.dumps({"__demo_step__": {
                                "tool": name, "summary": _summarize_demo_step(name, result)}}) + "\n"
                            if name == "save_demo" and result.get("saved"):
                                yield "\n" + json.dumps({"__demo_card__": {
                                    "preview_url": result.get("preview_url"),
                                    "slug": result.get("slug")}}) + "\n"
                            tool_msgs.append({"role": "tool", "tool_name": name,
                                              "content": json.dumps(result)})
                            if _is_vision and shot:
                                image_msgs.append({"role": "user",
                                    "content": "Screenshot of the rendered demo (visual review):",
                                    "images": [shot]})
                        new_messages = messages + [
                            {"role": "assistant", "content": "", "tool_calls": demo_calls},
                            *tool_msgs, *image_msgs]
                        async for wire in relay(_stream_one(
                                {**body_with_tools, "messages": new_messages}),
                                new_messages, iterations + 1):
                            yield wire
                        return
                    if demo_calls and iterations >= MAX_HTMLDEMO_ITERS:
                        yield "\n" + json.dumps({"__demo_step__": {
                            "tool": "html_demo", "summary": "(reached demo iteration limit)"}}) + "\n"
                    if cellc_calls and iterations < MAX_CELLC_ITERS:
                        if in_thinking:
                            in_thinking = False
                            yield THINK_CLOSE
                        tool_msgs = []
                        for c in cellc_calls:
                            name = c["function"]["name"]
                            args = c["function"].get("arguments") or {}
                            try:
                                result = await asyncio.to_thread(cellc_bridge.dispatch, name, args)
                            except Exception as exc:  # dispatch must never break the stream
                                result = {"ok": False, "tool_error": True, "exit_code": -1, "stderr": str(exc)}
                            yield "\n" + json.dumps({"__cellc_step__": {"tool": name, "summary": _summarize_cellc_step(name, result)}}) + "\n"
                            tool_msgs.append({"role": "tool", "tool_name": name, "content": json.dumps(result)})
                        new_messages = messages + [{"role": "assistant", "content": "", "tool_calls": cellc_calls}, *tool_msgs]
                        async for wire in relay(_stream_one({**body_with_tools, "messages": new_messages}), new_messages, iterations + 1):
                            yield wire
                        return
                    if cellc_calls and iterations >= MAX_CELLC_ITERS:
                        yield "\n" + json.dumps({"__cellc_step__": {"tool": "cellc", "summary": "(reached cellc check limit)"}}) + "\n"
                elif kind == "stats":
                    if in_thinking:
                        in_thinking = False
                        yield THINK_CLOSE
                    final_eval["tokens"] = value.get("eval_count")
                    yield "\n" + json.dumps({"__stats__": value})
                elif kind == "error":
                    err_msg = str(value.get("error", "")).lower()
                    if (not attempted_fallback
                            and "tools" in err_msg
                            and OPERATIONS.enabled):
                        log.info("model %s rejected tools — retrying without", model)
                        attempted_fallback = True
                        async for wire in relay(_stream_one(body_without_tools), messages, iterations):
                            yield wire
                        return
                    if in_thinking:
                        in_thinking = False
                        yield THINK_CLOSE
                    yield f"\n[backend error: Ollama {value['status']}: {value['error']}]"
                    return
                elif kind == "done":
                    if in_thinking:
                        in_thinking = False
                        yield THINK_CLOSE
                    return

        async for wire in relay(_stream_one(body_with_tools), messages, 0):
            yield wire

        if SESSIONS is not None and sess_id and assistant_parts:
            try:
                SESSIONS.append_message(sess_id, "assistant", "".join(assistant_parts),
                                        tokens=final_eval["tokens"], model=model)
                full = SESSIONS.get_full(sess_id)
                if full and not full.get("title") and len(full["messages"]) <= 2:
                    _first_user = next((m.get("content", "") for m in incoming
                                        if m.get("role") == "user"), "")
                    _t = asyncio.create_task(
                        _auto_title(sess_id, _first_user, "".join(assistant_parts)))
                    _BG_TASKS.add(_t)
                    _t.add_done_callback(_BG_TASKS.discard)
            except (SessionStoreError, sqlite3.Error) as exc:
                log.warning("session %s: assistant persist failed: %s", sess_id, exc)

    return StreamingResponse(stream(), media_type="text/plain")


@app.get("/api/converters")
async def list_converters():
    return {
        "enabled": [
            {"id": c.id, "from": list(c.sources), "to": list(c.targets), "params": c.params}
            for c in REGISTRY.enabled
        ],
        "disabled": [{"id": cid, "missing": req} for cid, req in REGISTRY.missing],
    }


# ─── Operations API ──────────────────────────────────────────────────


@app.get("/api/operations")
async def list_operations():
    return {
        "enabled": [op.to_client_dict() for op in OPERATIONS.enabled],
        "disabled": [{"id": oid, "reason": reason} for oid, reason in OPERATIONS.missing],
        "bridges_available": sorted(_BRIDGES_AVAILABLE),
        "output_dir": str(OUTPUT_DIR) if OUTPUT_DIR else None,
    }


@app.get("/api/capabilities")
async def get_all_capabilities():
    """Return the full capabilities cache. Frontend uses this to render
    glyphs in the model dropdown and decide which models need a probe."""
    return CAPABILITIES.all()


@app.get("/api/capabilities/{model:path}")
async def get_one_capability(model: str):
    """Return one model's capabilities, or 404 if not yet probed."""
    caps = CAPABILITIES.get(model)
    if caps is None:
        raise HTTPException(404, f"capabilities for {model!r} not yet probed")
    return caps


@app.post("/api/capabilities/probe")
async def probe_capability(payload: dict):
    """Probe one model via Ollama's /api/show metadata. Fast (<100ms)."""
    model = payload.get("model")
    if not model:
        raise HTTPException(400, "missing 'model'")
    return await CAPABILITIES.probe_now(model)


@app.post("/api/capabilities/probe-all")
async def probe_all_known():
    """Sweep every model currently in `ollama list` and probe each.
    Wipes existing cache first so stale entries from the old inference-
    based probe don't survive. Fast — ~50ms per model."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"{OLLAMA_URL}/api/tags")
            r.raise_for_status()
        names = [m["name"] for m in r.json().get("models", [])]
    except httpx.HTTPError as e:
        raise HTTPException(503, f"Ollama unreachable: {e}")
    # Clear old (potentially-bogus) cache entries before re-populating.
    async with CAPABILITIES._lock:
        CAPABILITIES._cache = {}
        CAPABILITIES._save()
    return await CAPABILITIES.probe_models(names)


# ─── Sessions API ────────────────────────────────────────────────────

@app.post("/api/sessions")
async def create_session(payload: dict):
    if SESSIONS is None:
        raise HTTPException(503, "sessions store unavailable")
    model = payload.get("model") or ""
    sid = SESSIONS.create_session(model=model, title=payload.get("title"))
    return {"id": sid}


@app.get("/api/sessions")
async def list_sessions():
    if SESSIONS is None:
        return []
    return SESSIONS.list_sessions()


@app.get("/api/sessions/{sid}")
async def get_session(sid: str):
    if SESSIONS is None:
        raise HTTPException(503, "sessions store unavailable")
    full = SESSIONS.get_full(sid)
    if full is None:
        raise HTTPException(404, "session not found")
    return full


@app.patch("/api/sessions/{sid}")
async def rename_session(sid: str, payload: dict):
    if SESSIONS is None or SESSIONS.get_full(sid) is None:
        raise HTTPException(404, "session not found")
    SESSIONS.rename(sid, payload.get("title") or "")
    return {"ok": True}


@app.delete("/api/sessions/{sid}")
async def delete_session(sid: str):
    if SESSIONS is not None:
        SESSIONS.delete(sid)
    return {"ok": True}


@app.post("/api/operations/enhance")
async def enhance(payload: dict):
    """Strengthen a free-form image prompt using the user's chat model.

    Returns ``{enhanced_prompt, negative_prompt, changes, mode, original_prompt}``.
    Always returns 200 — on failure the original prompt comes back
    unchanged with ``mode="passthrough"``."""
    user_prompt = (payload.get("prompt") or "").strip()
    chat_model = payload.get("chat_model") or ""
    if not user_prompt:
        raise HTTPException(400, "missing 'prompt'")
    if not chat_model:
        raise HTTPException(400, "missing 'chat_model'")
    return await enhance_prompt(
        ollama_url=OLLAMA_URL,
        chat_model=chat_model,
        user_prompt=user_prompt,
        image_model=payload.get("image_model") or "",
        explicit_mode=payload.get("mode") if payload.get("mode") != "auto" else None,
    )


_LANG_CODE_RE = re.compile(r"^[a-z]{2}(?:[-_][A-Za-z]{2,4})?$")
_TRANSLATE_MAX_TEXT_CHARS = 8000
_TRANSLATE_DEFAULT_NUM_CTX = 2048
# Default context window for /api/chat. Ollama otherwise caps unspecified
# num_ctx at ~4096, which truncates agentic cellc loops. num_ctx is the *total*
# budget shared by the system prompt, any injected reference files, the model's
# reasoning (<think>) tokens, AND its output — so heavy single-shot generations
# (e.g. a self-contained page embedding the 2048-word BIP39 list) exhaust 16k
# and truncate mid-stream. 24k gives that headroom and still fits qwen3.6:27b
# Q4 on a 24GB card with VRAM free; callers can override via payload num_ctx.
_CHAT_DEFAULT_NUM_CTX = 24576


@app.post("/api/operations/translate")
async def translate(payload: dict):
    """Translate text via a model that expects the TranslateGemma T3
    structured chat-template envelope.

    Wraps user content in ``{"type":"text","source_lang_code":...,
    "target_lang_code":...,"text":...}`` and posts to Ollama directly,
    skipping the standard system-prompt assembly that would otherwise
    knock the model out of pure-translation mode.

    Streams chunks in the same text/plain format as /api/chat and
    /api/auto-caption so the frontend reuses its existing render path.

    Caps num_ctx at 2048 by default so the 27 B Q4_K_M fits on a 24 GB
    card; caller may override via 'num_ctx' at their own VRAM risk.
    """
    model = (payload.get("model") or "").strip()
    source_lang = (payload.get("source_lang") or "").strip()
    target_lang = (payload.get("target_lang") or "").strip()
    text = payload.get("text") or ""

    if not model:
        raise HTTPException(400, "missing 'model'")
    if not _LANG_CODE_RE.match(source_lang):
        raise HTTPException(
            400,
            f"invalid 'source_lang': {source_lang!r} "
            "(expected ISO 639-1 like 'en' or 'pt-BR')",
        )
    if not _LANG_CODE_RE.match(target_lang):
        raise HTTPException(
            400,
            f"invalid 'target_lang': {target_lang!r} "
            "(expected ISO 639-1 like 'es' or 'zh-Hans')",
        )
    if not text.strip():
        raise HTTPException(400, "missing 'text'")
    if len(text) > _TRANSLATE_MAX_TEXT_CHARS:
        raise HTTPException(
            413,
            f"text too long ({len(text)} chars > {_TRANSLATE_MAX_TEXT_CHARS}); "
            "split into smaller chunks",
        )

    envelope = json.dumps(
        {
            "type": "text",
            "source_lang_code": source_lang,
            "target_lang_code": target_lang,
            "text": text,
        },
        ensure_ascii=False,
    )

    try:
        num_ctx = int(payload.get("num_ctx") or _TRANSLATE_DEFAULT_NUM_CTX)
    except (TypeError, ValueError):
        raise HTTPException(400, "'num_ctx' must be an integer")

    body = {
        "model": model,
        "messages": [{"role": "user", "content": envelope}],
        "stream": True,
        "options": {"num_ctx": num_ctx, "temperature": 0.1},
        "keep_alive": payload.get("keep_alive", "5m"),
    }

    async def stream():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", f"{OLLAMA_URL}/api/chat", json=body) as r:
                if r.status_code != 200:
                    raw = (await r.aread()).decode("utf-8", errors="replace")[:500]
                    yield f"\n[translate error: HTTP {r.status_code}: {raw}]"
                    return
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    chunk = (data.get("message") or {}).get("content")
                    if chunk:
                        yield chunk
                    if data.get("done"):
                        return

    return StreamingResponse(stream(), media_type="text/plain")


@app.post("/api/operations/run")
async def run_operation(payload: dict):
    """Execute a model-requested operation. Always called *after* the
    user has confirmed the Y/N/E card in the UI — never auto-fired
    from /api/chat directly. Returns the resulting workspace file."""
    op_id = payload.get("operation")
    if not op_id:
        raise HTTPException(400, "missing 'operation'")
    op = OPERATIONS.find(op_id)
    if op is None:
        raise HTTPException(404, f"operation {op_id!r} not found or disabled")

    session_id = payload.get("session_id", "default")
    session_dir, _ = _resolve_in_workspace(session_id, "_")
    session_dir.mkdir(parents=True, exist_ok=True)

    raw_params = payload.get("params") or {}
    if not isinstance(raw_params, dict):
        raise HTTPException(400, "'params' must be an object")

    try:
        validated = validate_params(op, raw_params)
    except ValueError as e:
        raise HTTPException(400, str(e))

    if op.kind == "local":
        out = await _run_local_op(op, session_dir, payload, validated)
    elif op.kind == "bridge":
        out = await _run_bridge_op(op, session_dir, validated)
    elif op.kind == "converter":
        out = await _run_converter_op(session_dir, validated)
    elif op.kind == "cellc_save":
        out = await _run_cellc_save_op(op, session_dir, validated)
    else:
        raise HTTPException(500, f"unknown op kind {op.kind!r}")

    mirrored = copy_to_output(out, OUTPUT_DIR)
    return {
        "name": out.name,
        "session_id": session_id,
        "size": out.stat().st_size,
        "via": op.id,
        "kind": op.kind,
        "mirror": str(mirrored) if mirrored else None,
        "url": f"/api/files/{session_id}/{out.name}",
    }


_CELLC_SAVE_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _run_cellc_save_op_sync(session_dir: Path, validated: dict) -> Path:
    name = validated["name"]
    if not _CELLC_SAVE_NAME_RE.match(name):
        raise HTTPException(400, "invalid save name (letters, digits, _ or - only)")
    source = validated["source"]
    result = cellc_bridge.check(source)
    if not result.get("ok"):
        diags = result.get("diagnostics") or []
        first = diags[0].get("message") if diags else "check failed"
        raise HTTPException(400, f"refusing to save: cellc_check failed ({result.get('error_count', '?')} error(s)): {first}")
    target = (session_dir / f"{name}.cell").resolve()
    target.relative_to(session_dir.resolve())  # path-traversal guard
    target.write_text(source, encoding="utf-8")
    return target


async def _run_cellc_save_op(op, session_dir: Path, validated: dict) -> Path:
    return await asyncio.to_thread(_run_cellc_save_op_sync, session_dir, validated)


async def _run_local_op(
    op: Operation, session_dir: Path, payload: dict, validated: dict[str, str],
) -> Path:
    source_path: Path | None = None
    source_name = payload.get("source")
    if source_name:
        _, source_path = _resolve_in_workspace(payload.get("session_id", "default"), source_name)
        if not source_path.exists():
            raise HTTPException(404, f"source file not found: {source_name}")
    elif op.source_param:
        # Operations that declare a source_param need a file. Bail clearly.
        raise HTTPException(400, f"operation {op.id} needs a 'source' file")

    try:
        return await asyncio.to_thread(run_local, op, session_dir, source_path, validated)
    except subprocess.TimeoutExpired:
        raise HTTPException(504, f"{op.id} timed out")
    except (RuntimeError, ValueError) as e:
        raise HTTPException(500, str(e))


async def _run_converter_op(
    session_dir: Path, validated: dict[str, str],
) -> Path:
    """Dispatch the synthetic ``convert`` op through the existing converter
    registry. The model picks source filename + target extension; we look
    up the right converter and run it through the same code path the
    user-tray UI uses."""
    source_name = validated["source"]
    target_ext = validated["target"].lstrip(".").lower()
    _, input_path = _resolve_in_workspace(session_dir.name, source_name)
    if not input_path.exists():
        raise HTTPException(404, f"file not found in workspace: {source_name}")
    src_ext = input_path.suffix.lstrip(".").lower()
    converter = REGISTRY.find(src_ext, target_ext)
    if not converter:
        reachable = sorted(REGISTRY.reachable_from(src_ext))
        raise HTTPException(
            400,
            f"no converter for .{src_ext} → .{target_ext}. "
            f"From .{src_ext} you can reach: {reachable or '(nothing)'}",
        )
    try:
        return await asyncio.to_thread(
            run_conversion, converter, input_path, session_dir, target_ext, {},
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(504, f"{converter.id} timed out")
    except (RuntimeError, ValueError) as e:
        raise HTTPException(500, str(e))


async def _run_bridge_op(
    op: Operation, session_dir: Path, validated: dict[str, str],
) -> Path:
    if op.bridge != "open_palette":
        raise HTTPException(500, f"unknown bridge {op.bridge!r}")
    bridge_cfg = BRIDGES_CFG.get(op.bridge) or {}
    base_url = bridge_cfg.get("url")
    if not base_url:
        raise HTTPException(503, f"bridge {op.bridge} not configured")
    # Strip the chat-only ``mode`` slot before forwarding to open-palette
    # (it doesn't recognise it).
    forward = {k: v for k, v in validated.items() if k != "mode"}
    try:
        return await open_palette.generate_image(
            base_url=base_url,
            params=forward,
            workspace=session_dir,
            overall_timeout=float(bridge_cfg.get("timeout", 600)),
        )
    except open_palette.BridgeUnavailable as e:
        raise HTTPException(502, str(e))


# ─── cellc (CellScript) endpoints — read-only, no workspace files ─────
# Dedicated like /api/operations/translate (cellc returns DATA, not a
# file, so it does not fit the file-producing OPERATIONS registry).

# `re` and `asyncio` are already imported at the top of backend/app.py
# (lines 3 and 8) — use them directly; do NOT add duplicate imports.

_CELLC_INTENT_RE = re.compile(
    r"cellscript|\.cell\b|cell contract|ckb contract|nervos contract|\bcellc\b|resource\s+\w+\s+has",
    re.IGNORECASE,
)


def _is_cellc_intent(text: str) -> bool:
    return bool(_CELLC_INTENT_RE.search(text or ""))


_CELLC_PROMPT_HINT = (
    "\n\nCellScript (.cell) tooling is available. Workflow: write the contract, "
    "call cellc_check to verify it, and if it reports errors read them and fix "
    "the contract, then check again, until it passes. Once cellc_check passes, "
    "call cellc_save with the EXACT source you just verified to persist it as a "
    "file — do not retype or reformat the contract in your reply, because edits "
    "made after verification are unverified; the saved file is the source of "
    "truth. The CellScript language reference is already provided above; you do "
    "not need a tool to fetch it."
)

def _inject_cellc_context(messages: list, last_user: str) -> bool:
    """If cellc is available and the message looks CellScript-related, append the
    language reference + design notes to the system message. Returns True if so."""
    if not (cellc_bridge.available() and _is_cellc_intent(last_user)):
        return False
    messages[0]["content"] += "\n\n# CellScript language reference\n" + cellc_bridge.language_reference()
    notes = cellc_bridge.design_notes()
    if notes:
        messages[0]["content"] += "\n\n# CellScript design notes\n" + notes
    return True


_HTMLDEMO_INTENT_RE = re.compile(
    r"html\s+demo|self-contained\s+html|single[- ]file\s+html|canvas\s+game|"
    r"\bmake\s+(?:me\s+)?a\s+game\b|physics\s+demo|interactive\s+demo|"
    r"animation\s+in\s+html|webpage\s+that",
    re.IGNORECASE,
)


def _is_html_demo_intent(text: str) -> bool:
    return bool(_HTMLDEMO_INTENT_RE.search(text or ""))


_HTMLDEMO_PROMPT_HINT = (
    "\n\nYou can build and TEST self-contained HTML demos. Workflow: write the "
    "full HTML, call preview_demo (returns a demo_id + diagnostics), then fix "
    "issues with patch_demo(demo_id, edits) — small unique find/replace edits, "
    "do NOT resend the whole file. When it renders cleanly, call save_demo. "
    "Full house-style guidance is provided above when you start a demo."
)


def _inject_html_demo_context(messages: list, last_user: str) -> bool:
    """If the html_demo skill is available and the message asks for a demo,
    append the house-style guidance + one example to the system message.
    Returns True if injected. Kept lean: guidance + a single example only."""
    if not (html_demo.available() and _is_html_demo_intent(last_user)):
        return False
    messages[0]["content"] += "\n\n# Self-contained HTML demo guidance\n" + html_demo.guidance()
    ex = html_demo.examples()
    if ex:
        messages[0]["content"] += (
            f"\n\n# Example demo ({ex[0]['name']})\n```html\n{ex[0]['html']}\n```")
    return True


def _summarize_demo_step(name: str, result: dict) -> str:
    if result.get("tool_error"):
        return f"⚠ {name}: {result.get('message', 'error')[:80]}"
    if name == "save_demo":
        return f"save_demo → saved {result.get('slug', '')}"
    if result.get("applied") is False:
        return f"patch_demo → ✗ {str(result.get('reason', ''))[:70]}"
    did = result.get("demo_id", "")
    if not result.get("loaded", True):
        return f"{name} → ✗ failed to load ({did})"
    errs = len(result.get("console_errors", [])) + len(result.get("exceptions", []))
    if errs:
        return f"{name} → ✗ {errs} error(s) ({did})"
    raf = "raf✓" if result.get("raf_ran") else "static"
    return f"{name} → ✓ clean, {raf} ({did})"


def _inject_text_files(messages: list, session_id: str, text_files: list) -> int:
    """Append the contents of named workspace text files to the last user
    message so the model can use them as reference.

    The upload pipeline only stores bytes and converts *binary* documents
    (pdf/docx/…) to markdown; plain text/markdown/json have no converter, so
    without this their content never reaches the model — it only ever sees
    the filename. Mirrors the ``image_files`` contract: filenames are
    resolved + validated against the session workspace, unknown/oversized/
    non-text entries are skipped with a log line, never raising. Oversized
    files are clamped to ``_MAX_TEXT_INJECT_CHARS`` with a visible marker so
    the injected reference can't silently blow the num_ctx budget. Returns
    the number of files actually injected."""
    if not text_files:
        return 0
    blocks: list[str] = []
    for name in text_files:
        try:
            _, path = _resolve_in_workspace(session_id, name)
        except (ValueError, HTTPException) as exc:
            log.warning("chat: cannot resolve text file %r: %s", name, exc)
            continue
        if not path.exists() or path.suffix.lower() not in _TEXT_EXTS:
            log.warning("chat: skipping non-text or missing file %r", name)
            continue
        try:
            text = path.read_bytes().decode("utf-8", errors="replace")
        except OSError as exc:
            log.warning("chat: cannot read text file %r: %s", name, exc)
            continue
        marker = ""
        if len(text) > _MAX_TEXT_INJECT_CHARS:
            dropped = len(text) - _MAX_TEXT_INJECT_CHARS
            text = text[:_MAX_TEXT_INJECT_CHARS]
            marker = f"\n[… truncated {dropped} chars of {path.name}]"
        blocks.append(f"--- attached file: {path.name} ---\n{text}{marker}")
    if not blocks:
        return 0
    appendix = (
        "\n\nThe user attached the following file(s) as reference. Use their "
        "contents to answer the request above:\n\n" + "\n\n".join(blocks)
    )
    for m in reversed(messages):
        if m.get("role") == "user":
            m["content"] = (m.get("content") or "") + appendix
            return len(blocks)
    # No user message at all (degenerate) — attach to the system prompt so
    # the reference is at least present rather than silently dropped.
    messages[0]["content"] += appendix
    return len(blocks)


_CELLC_MAX_SOURCE = 200_000
_CELLC_PROFILES = {"ckb"}
_CELLC_CODE_RE = re.compile(r"^[A-Za-z0-9_]+$")
_CELLC_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _cellc_require_available() -> None:
    if not cellc_bridge.available():
        raise HTTPException(
            503, "cellc not available — build it and set CELLC_BIN "
                 "(cd ~/CellScript && cargo build --release -p cellscript --bin cellc)"
        )


def _cellc_validate_source(payload: dict) -> str:
    source = payload.get("source")
    if not isinstance(source, str) or not source.strip():
        raise HTTPException(400, "missing or empty 'source'")
    if len(source) > _CELLC_MAX_SOURCE:
        raise HTTPException(413, f"source too long ({len(source)} > {_CELLC_MAX_SOURCE})")
    return source


def _cellc_validate_profile(payload: dict) -> str:
    profile = payload.get("target_profile", "ckb")
    if profile not in _CELLC_PROFILES:
        raise HTTPException(400, f"invalid target_profile {profile!r}; allowed: {sorted(_CELLC_PROFILES)}")
    return profile


@app.get("/api/cellc/status")
async def cellc_status():
    return cellc_bridge.status()


@app.post("/api/cellc/check")
async def cellc_check_endpoint(payload: dict):
    _cellc_require_available()
    source = _cellc_validate_source(payload)
    profile = _cellc_validate_profile(payload)
    full = bool(payload.get("full"))
    return await asyncio.to_thread(cellc_bridge.check, source=source, target_profile=profile, full=full)


@app.post("/api/cellc/metadata")
async def cellc_metadata_endpoint(payload: dict):
    _cellc_require_available()
    source = _cellc_validate_source(payload)
    profile = _cellc_validate_profile(payload)
    full = bool(payload.get("full"))
    return await asyncio.to_thread(cellc_bridge.metadata, source=source, target_profile=profile, full=full)


@app.post("/api/cellc/explain")
async def cellc_explain_endpoint(payload: dict):
    _cellc_require_available()
    code = payload.get("code")
    if not isinstance(code, str) or not _CELLC_CODE_RE.match(code):
        raise HTTPException(400, "invalid 'code' (expected an error code/name like E0014)")
    return await asyncio.to_thread(cellc_bridge.explain, code=code)


@app.get("/api/cellc/reference")
async def cellc_reference_endpoint():
    _cellc_require_available()
    return {"reference": cellc_bridge.language_reference()}


@app.get("/api/cellc/examples")
async def cellc_examples_endpoint():
    _cellc_require_available()
    return {"examples": cellc_bridge.list_examples()}


@app.get("/api/cellc/example/{name}")
async def cellc_example_endpoint(name: str):
    _cellc_require_available()
    if not _CELLC_NAME_RE.match(name):
        raise HTTPException(400, "invalid example name")
    result = cellc_bridge.get_example(name)
    if result.get("tool_error"):
        raise HTTPException(404, result.get("stderr", "example not found"))
    return result


# ─── Workspace helpers + uploads + downloads ──────────────────────────


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


def _resolve_in_dropbox(name: str) -> Path:
    """Resolve a basename inside the drop root; reject anything that escapes it."""
    target = (DROPBOX / Path(name).name).resolve()
    if target.parent != DROPBOX:
        raise HTTPException(400, "invalid path")
    return target


@app.get("/api/dropbox")
async def dropbox_list():
    if not DROPBOX.exists():
        return {"files": []}
    entries = []
    for p in DROPBOX.iterdir():
        if p.name.startswith(".") or not p.is_file():
            continue
        st = p.stat()
        entries.append({"name": p.name, "size": st.st_size, "modified": st.st_mtime})
    entries.sort(key=lambda e: e["modified"], reverse=True)
    return {"files": entries}


@app.post("/api/dropbox/import")
async def dropbox_import(payload: dict):
    session_id = payload.get("session_id", "default")
    names = payload.get("names") or []
    if not isinstance(names, list):
        raise HTTPException(400, "'names' must be a list")

    session_dir, _ = _resolve_in_workspace(session_id, "_")
    session_dir.mkdir(parents=True, exist_ok=True)

    imported: list[dict] = []
    skipped: list[dict] = []
    for name in names:
        try:
            src = _resolve_in_dropbox(str(name))
        except HTTPException:
            skipped.append({"name": name, "reason": "invalid name"})
            continue
        if not src.is_file():
            skipped.append({"name": name, "reason": "not found"})
            continue
        _, dest = _resolve_in_workspace(session_id, src.name)
        try:
            shutil.copy(src, dest)
            imported.append({"name": dest.name, "size": dest.stat().st_size})
        except OSError:
            skipped.append({"name": name, "reason": "copy failed"})
            continue

    return {"imported": imported, "skipped": skipped}


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

    mirrored = copy_to_output(out, OUTPUT_DIR)
    return {
        "name": out.name,
        "session_id": session_id,
        "size": out.stat().st_size,
        "via": converter.id,
        "mirror": str(mirrored) if mirrored else None,
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
app.mount("/demos", StaticFiles(directory=DEMOS_DIR, check_dir=False), name="demos")


@app.get("/")
async def index():
    return FileResponse(ROOT / "frontend" / "index.html")
