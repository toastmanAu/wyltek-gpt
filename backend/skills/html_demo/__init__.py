"""HTML-demo skill — a bridge cut from the cellc mold.

Exposes three tools (preview_demo / patch_demo / save_demo) that let a
tool-capable model write, headless-test, patch, and save a self-contained
HTML demo. HTML lives server-side under a demo_id handle (see store.py) so
it crosses the model's context at most once. ``available()`` gates on
Playwright/Chromium so the app boots without the feature.
"""
from __future__ import annotations

import logging
from pathlib import Path

from . import render as _render
from .store import DemoStore

log = logging.getLogger(__name__)

_HERE = Path(__file__).parent
_MAX_HTML = 400_000

HTML_DEMO_TOOL_NAMES = frozenset({"preview_demo", "patch_demo", "save_demo"})

_STORE: DemoStore | None = None


def init_store(demos_dir: Path) -> DemoStore:
    global _STORE
    _STORE = DemoStore(demos_dir)
    return _STORE


def store() -> DemoStore:
    if _STORE is None:
        raise RuntimeError("html_demo store not initialised — call init_store()")
    return _STORE


def available() -> bool:
    return _render.available()


def guidance() -> str:
    return (_HERE / "guidance.md").read_text()


def examples() -> list[dict]:
    out: list[dict] = []
    exdir = _HERE / "examples"
    if exdir.is_dir():
        for f in sorted(exdir.glob("*.html")):
            out.append({"name": f.stem, "html": f.read_text()})
    return out


def tool_schemas() -> list[dict]:
    def fn(name, desc, props, required):
        return {"type": "function", "function": {
            "name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required}}}
    return [
        fn("preview_demo",
           "Render a self-contained HTML demo headless and return diagnostics "
           "(load ok, JS console errors, exceptions, DOM/canvas sanity, whether "
           "requestAnimationFrame ran). Returns a demo_id — reuse it for patches. "
           "Send the FULL html here only once; use patch_demo to iterate.",
           {"html": {"type": "string", "description": "full self-contained HTML document"},
            "name": {"type": "string", "description": "short label for the demo"},
            "demo_id": {"type": "string", "description": "reuse to replace an existing demo (rewrite fallback)"},
            "settle_ms": {"type": "integer", "description": "ms to wait after load before capture (default 700)"}},
           ["html"]),
        fn("patch_demo",
           "Apply small find/replace edits to a stored demo (by demo_id) and "
           "re-render. Each 'find' must match the stored HTML uniquely. On "
           "failure, re-send full html via preview_demo with the same demo_id.",
           {"demo_id": {"type": "string", "description": "handle from preview_demo"},
            "edits": {"type": "array", "description": "list of {find, replace} string edits",
                      "items": {"type": "object", "properties": {
                          "find": {"type": "string"}, "replace": {"type": "string"}},
                          "required": ["find", "replace"]}}},
           ["demo_id", "edits"]),
        fn("save_demo",
           "Persist a finished demo (by demo_id) to disk and surface a preview "
           "card in the chat. Call this once the demo renders cleanly.",
           {"demo_id": {"type": "string", "description": "handle to persist"},
            "name": {"type": "string", "description": "optional filename label override"}},
           ["demo_id"]),
    ]


def _err(msg: str) -> dict:
    return {"tool_error": True, "message": msg}


def dispatch(name: str, arguments: dict) -> dict:
    arguments = arguments or {}
    st = store()
    settle = int(arguments.get("settle_ms", 700) or 700)
    want_shot = bool(arguments.get("_want_screenshot"))
    interactions = arguments.get("interactions")

    if name == "preview_demo":
        html = arguments.get("html")
        if not isinstance(html, str) or not html.strip():
            return _err("missing or empty 'html'")
        if len(html) > _MAX_HTML:
            return _err(f"html too long ({len(html)} > {_MAX_HTML})")
        demo_id = arguments.get("demo_id")
        if isinstance(demo_id, str) and st.get(demo_id):
            st.replace(demo_id, html)
        else:
            demo_id = st.create(html, arguments.get("name"))
        diag = _render.render(html, settle_ms=settle, interactions=interactions,
                              want_screenshot=want_shot)
        diag["demo_id"] = demo_id
        return diag

    if name == "patch_demo":
        demo_id = arguments.get("demo_id")
        if not isinstance(demo_id, str) or not st.get(demo_id):
            return _err(f"unknown demo_id {demo_id!r} — call preview_demo first")
        ok, reason = st.apply_patch(demo_id, arguments.get("edits") or [])
        if not ok:
            return {"applied": False, "reason": reason, "demo_id": demo_id}
        html = st.get(demo_id)["html"]
        diag = _render.render(html, settle_ms=settle, interactions=interactions,
                              want_screenshot=want_shot)
        diag["demo_id"] = demo_id
        diag["applied"] = True
        return diag

    if name == "save_demo":
        demo_id = arguments.get("demo_id")
        if not isinstance(demo_id, str) or not st.get(demo_id):
            return _err(f"unknown demo_id {demo_id!r}")
        try:
            res = st.save(demo_id, arguments.get("name"))
        except KeyError:
            return _err("demo not found")
        res["saved"] = True
        return res

    return _err(f"unknown html_demo tool {name!r}")
