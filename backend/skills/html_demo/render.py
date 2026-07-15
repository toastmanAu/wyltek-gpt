"""Headless Chromium render + diagnostics for HTML demos.

Runs the given self-contained HTML in a mobile-viewport headless page,
captures JS console errors / uncaught exceptions, a DOM/canvas sanity
snapshot, whether requestAnimationFrame fired (physics proof), and — only
when asked — a downscaled PNG screenshot. Never raises: on any failure it
returns ``{"loaded": False, "error": ...}``.
"""
from __future__ import annotations

import base64
import io
import logging

log = logging.getLogger(__name__)

try:
    from playwright.sync_api import sync_playwright
    _PW_OK = True
except Exception as exc:  # pragma: no cover - only without playwright
    log.warning("playwright not importable — html_demo render disabled: %s", exc)
    sync_playwright = None  # type: ignore[assignment]
    _PW_OK = False

try:
    from PIL import Image
    _PIL_OK = True
except Exception:  # pragma: no cover
    Image = None  # type: ignore[assignment]
    _PIL_OK = False

_VIEWPORT = {"width": 390, "height": 844}
_MAX_ITEMS = 20
_SHOT_WIDTH = 300

# Runs before any page script on every document: shim rAF to record that it ran.
_RAF_PROBE = (
    "window.__rafRan = false;"
    "const _raf = window.requestAnimationFrame;"
    "window.requestAnimationFrame = function(cb){ window.__rafRan = true; return _raf(cb); };"
)


def available() -> bool:
    if not _PW_OK:
        return False
    try:
        with sync_playwright() as p:
            b = p.chromium.launch(headless=True)
            b.close()
        return True
    except Exception as exc:  # pragma: no cover
        log.warning("chromium not launchable — html_demo render disabled: %s", exc)
        return False


def _downscale_b64(png: bytes) -> str:
    if not _PIL_OK:
        return base64.b64encode(png).decode("ascii")
    img = Image.open(io.BytesIO(png))
    if img.width > _SHOT_WIDTH:
        h = round(img.height * _SHOT_WIDTH / img.width)
        img = img.resize((_SHOT_WIDTH, h))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _do_interaction(page, act: dict) -> None:
    t = act.get("type")
    x = float(act.get("x", 0) or 0)
    y = float(act.get("y", 0) or 0)
    try:
        if t in ("tap", "click", "pointerdown"):
            page.mouse.click(x, y)
        elif t == "move":
            page.mouse.move(x, y)
    except Exception as exc:  # interactions are best-effort
        log.debug("interaction %s failed: %s", t, exc)


def render(html: str, settle_ms: int = 700, interactions: list | None = None,
           want_screenshot: bool = False) -> dict:
    if not _PW_OK:
        return {"loaded": False, "error": "playwright unavailable"}
    console_errors: list[str] = []
    exceptions: list[str] = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                page = browser.new_page(viewport=_VIEWPORT)
                page.add_init_script(_RAF_PROBE)
                page.on("console", lambda m: (
                    console_errors.append(f"{m.type}: {m.text}")
                    if m.type in ("error", "warning") else None))
                page.on("pageerror", lambda e: exceptions.append(str(e)))
                # NOTE: page.set_content() does NOT trigger add_init_script (no
                # real navigation occurs), so the rAF probe would silently never
                # run. Navigating to a base64 data: URL is a real navigation and
                # correctly fires init scripts before page scripts execute.
                data_url = "data:text/html;base64," + base64.b64encode(html.encode("utf-8")).decode("ascii")
                page.goto(data_url, wait_until="load")
                for act in (interactions or [])[:_MAX_ITEMS]:
                    _do_interaction(page, act)
                page.wait_for_timeout(max(0, min(int(settle_ms), 5000)))
                sanity = page.evaluate(
                    "() => ({"
                    " elements: document.getElementsByTagName('*').length,"
                    " text_len: (document.body ? document.body.innerText.length : 0),"
                    " has_canvas: !!document.querySelector('canvas'),"
                    " raf_ran: !!window.__rafRan })")
                shot = None
                if want_screenshot:
                    shot = _downscale_b64(page.screenshot(type="png"))
                return {
                    "loaded": True,
                    "console_errors": console_errors[:_MAX_ITEMS],
                    "exceptions": exceptions[:_MAX_ITEMS],
                    "render_sanity": sanity,
                    "raf_ran": bool(sanity.get("raf_ran")),
                    "viewport": _VIEWPORT,
                    "_screenshot_b64": shot,
                }
            finally:
                browser.close()
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("render failed: %s", exc)
        return {"loaded": False, "error": str(exc)}
