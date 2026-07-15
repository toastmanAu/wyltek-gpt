# HTML-Demo Skill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the local Ollama fleet a context-lean, house-styled, agentic build-test-fix loop for self-contained HTML demos, rendered headless and previewable in the PWA.

**Architecture:** A new skill package `backend/skills/html_demo/` cut from the proven `cellc` bridge mold: `available()`/`tool_schemas()`/`dispatch()`/`guidance()`/`examples()`. HTML lives server-side under a `demo_id` handle so the full document crosses the model's context once; iterations are find/replace patches with a full-rewrite fallback. Three tools — `preview_demo`, `patch_demo`, `save_demo` — loop through the existing `relay()` machinery in `/api/chat`, mirroring the cellc write→check→fix loop. Screenshots are captured only for vision-capable chat models.

**Tech Stack:** Python 3 / FastAPI (backend), Playwright sync API + headless Chromium (render), Pillow (screenshot downscale), vanilla JS (frontend, no framework), pytest (tests).

## Global Constraints

- Self-contained single-file HTML only — no CDNs, bundlers, or external assets.
- Skill must **degrade gracefully**: `available()` returns False and the skill silently does not register if Playwright/Chromium is absent. App must boot cleanly either way (mirror `cellc.available()`).
- **Screenshot to the model ONLY if that model is vision-capable** (`CAPABILITIES.get(model)["vision"]`). Never attach a screenshot for non-vision models.
- **Context-lean, in priority order:** (1) HTML lives under a `demo_id` handle, re-sent to the model at most once; (2) edits are patches with rewrite fallback; (3) `guidance()`+`examples()` inject only on demo intent, never in every chat; (4) tool results carry only actionable diagnostics, never the HTML or a DOM dump; (5) iteration cap `MAX_HTMLDEMO_ITERS = 4`.
- Tool names are exactly `preview_demo`, `patch_demo`, `save_demo`.
- Follow existing patterns: `dispatch()` never raises (returns an error dict); UI sentinels are single-line JSON like `{"__demo_step__": {...}}`; run blocking work via `asyncio.to_thread`.
- Commit style: `type: description`, no attribution trailer (attribution disabled globally). Work on branch `feat/html-demo-skill` (already created).

---

### Task 1: Demo handle store (`store.py`)

**Files:**
- Create: `backend/skills/__init__.py` (empty)
- Create: `backend/skills/html_demo/__init__.py` (empty for now — filled in Task 3)
- Create: `backend/skills/html_demo/store.py`
- Test: `tests/test_html_demo_store.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `DemoStore(demos_dir: Path)` with:
    - `create(html: str, name: str | None = None) -> str` (returns `demo_id`)
    - `get(demo_id: str) -> dict | None` (dict has `"html"`, `"name"`)
    - `replace(demo_id: str, html: str) -> bool`
    - `apply_patch(demo_id: str, edits: list[dict]) -> tuple[bool, str]` (edits are `{"find","replace"}`; `find` must match uniquely)
    - `save(demo_id: str, name: str | None = None) -> dict` (returns `{"path","slug","preview_url"}`)
  - `_slugify(name: str) -> str`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_html_demo_store.py
from pathlib import Path
import pytest
from backend.skills.html_demo.store import DemoStore, _slugify


def test_create_get_roundtrip(tmp_path):
    s = DemoStore(tmp_path)
    did = s.create("<html>A</html>", "My Demo")
    assert isinstance(did, str) and did
    rec = s.get(did)
    assert rec["html"] == "<html>A</html>"
    assert rec["name"] == "My Demo"


def test_replace_updates_html(tmp_path):
    s = DemoStore(tmp_path)
    did = s.create("<html>A</html>")
    assert s.replace(did, "<html>B</html>") is True
    assert s.get(did)["html"] == "<html>B</html>"
    assert s.replace("nope", "x") is False


def test_apply_patch_unique_find(tmp_path):
    s = DemoStore(tmp_path)
    did = s.create("<h1>hello</h1><p>world</p>")
    ok, reason = s.apply_patch(did, [{"find": "world", "replace": "there"}])
    assert ok is True and reason == "ok"
    assert s.get(did)["html"] == "<h1>hello</h1><p>there</p>"


def test_apply_patch_missing_find_fails(tmp_path):
    s = DemoStore(tmp_path)
    did = s.create("<h1>hello</h1>")
    ok, reason = s.apply_patch(did, [{"find": "absent", "replace": "x"}])
    assert ok is False
    assert "not found" in reason


def test_apply_patch_ambiguous_find_fails(tmp_path):
    s = DemoStore(tmp_path)
    did = s.create("<b>x</b><b>x</b>")
    ok, reason = s.apply_patch(did, [{"find": "<b>x</b>", "replace": "y"}])
    assert ok is False
    assert "unique" in reason


def test_save_writes_file_and_urls(tmp_path):
    s = DemoStore(tmp_path)
    did = s.create("<html>Z</html>", "Cool Thing!")
    res = s.save(did)
    assert Path(res["path"]).read_text() == "<html>Z</html>"
    assert res["slug"] == "cool-thing"
    assert res["preview_url"] == "/demos/cool-thing.html"


def test_save_collision_gets_suffix(tmp_path):
    s = DemoStore(tmp_path)
    a = s.create("<html>ONE</html>", "dupe")
    b = s.create("<html>TWO</html>", "dupe")
    ra = s.save(a)
    rb = s.save(b)
    assert ra["preview_url"] != rb["preview_url"]
    assert Path(ra["path"]).read_text() == "<html>ONE</html>"
    assert Path(rb["path"]).read_text() == "<html>TWO</html>"


def test_slugify():
    assert _slugify("Hello World") == "hello-world"
    assert _slugify("  ") == "demo"
    assert _slugify("A/B:C") == "a-b-c"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/local-chatbot && .venv/bin/pytest tests/test_html_demo_store.py -v`
Expected: FAIL — `ModuleNotFoundError: backend.skills.html_demo.store`

- [ ] **Step 3: Write minimal implementation**

Create `backend/skills/__init__.py` and `backend/skills/html_demo/__init__.py` as empty files. Then create `backend/skills/html_demo/store.py`:

```python
"""In-process handle store for HTML demos.

The full HTML for a demo lives here keyed by a short ``demo_id`` so it
crosses the model's context at most once — subsequent iterations reference
the handle and send only find/replace patches. Bounded LRU so a long
session can't leak memory; the store is process-local (single uvicorn
worker), never persisted except via ``save()``.
"""
from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path

_MAX_HTML = 400_000
_STORE_CAP = 64
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(name: str) -> str:
    s = _SLUG_RE.sub("-", (name or "demo").lower()).strip("-")
    return s or "demo"


class DemoStore:
    def __init__(self, demos_dir: Path):
        self.demos_dir = demos_dir
        self._store: "OrderedDict[str, dict]" = OrderedDict()
        self._counter = 0

    def create(self, html: str, name: str | None = None) -> str:
        self._counter += 1
        demo_id = f"demo{self._counter}"
        self._store[demo_id] = {"html": html, "name": name or demo_id}
        self._store.move_to_end(demo_id)
        while len(self._store) > _STORE_CAP:
            self._store.popitem(last=False)
        return demo_id

    def get(self, demo_id: str) -> dict | None:
        rec = self._store.get(demo_id)
        if rec is not None:
            self._store.move_to_end(demo_id)
        return rec

    def replace(self, demo_id: str, html: str) -> bool:
        if demo_id not in self._store:
            return False
        self._store[demo_id]["html"] = html
        self._store.move_to_end(demo_id)
        return True

    def apply_patch(self, demo_id: str, edits: list[dict]) -> tuple[bool, str]:
        rec = self._store.get(demo_id)
        if rec is None:
            return False, f"unknown demo_id {demo_id!r}"
        html = rec["html"]
        for i, e in enumerate(edits or []):
            find = e.get("find")
            repl = e.get("replace")
            if not isinstance(find, str) or not isinstance(repl, str):
                return False, f"edit {i}: 'find' and 'replace' must both be strings"
            count = html.count(find)
            if count == 0:
                return False, (f"edit {i}: find-string not found — re-send the full "
                               "HTML via preview_demo with this demo_id")
            if count > 1:
                return False, (f"edit {i}: find-string matches {count} times (must be "
                               "unique) — add surrounding context to disambiguate")
            html = html.replace(find, repl, 1)
        if len(html) > _MAX_HTML:
            return False, f"patched HTML too large ({len(html)} > {_MAX_HTML})"
        rec["html"] = html
        self._store.move_to_end(demo_id)
        return True, "ok"

    def save(self, demo_id: str, name: str | None = None) -> dict:
        rec = self._store.get(demo_id)
        if rec is None:
            raise KeyError(demo_id)
        slug = _slugify(name or rec["name"])
        self.demos_dir.mkdir(parents=True, exist_ok=True)
        path = self.demos_dir / f"{slug}.html"
        n = 1
        while path.exists() and path.read_text(errors="replace") != rec["html"]:
            path = self.demos_dir / f"{slug}-{n}.html"
            n += 1
        path.write_text(rec["html"])
        return {"path": str(path), "slug": path.stem, "preview_url": f"/demos/{path.name}"}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/local-chatbot && .venv/bin/pytest tests/test_html_demo_store.py -v`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git -C ~/local-chatbot add backend/skills/__init__.py backend/skills/html_demo/__init__.py backend/skills/html_demo/store.py tests/test_html_demo_store.py
git -C ~/local-chatbot commit -m "feat: html-demo handle store with patch + save"
```

---

### Task 2: Headless render + diagnostics (`render.py`)

**Files:**
- Create: `backend/skills/html_demo/render.py`
- Test: `tests/test_html_demo_render.py`

**Interfaces:**
- Consumes: nothing from prior tasks.
- Produces:
  - `available() -> bool` (Playwright importable AND Chromium launchable)
  - `render(html: str, settle_ms: int = 700, interactions: list | None = None, want_screenshot: bool = False) -> dict`
    - returns keys: `loaded: bool`, `console_errors: list[str]`, `exceptions: list[str]`, `render_sanity: dict`, `raf_ran: bool`, `viewport: dict`, `_screenshot_b64: str | None`

**Note:** The render tests are integration tests gated on Chromium being installed; they `pytest.skip` when `available()` is False, so the suite stays green on machines without Playwright.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_html_demo_render.py
import pytest
from backend.skills.html_demo import render


pytestmark = pytest.mark.skipif(not render.available(), reason="Playwright/Chromium not installed")


def test_clean_page_loads_no_errors():
    html = "<!doctype html><html><body><h1 id='t'>hi</h1></body></html>"
    r = render.render(html, settle_ms=100)
    assert r["loaded"] is True
    assert r["console_errors"] == []
    assert r["exceptions"] == []
    assert r["render_sanity"]["text_len"] >= 2
    assert r["_screenshot_b64"] is None  # not requested


def test_js_exception_is_captured():
    html = "<!doctype html><html><body><script>throw new Error('BOOM')</script></body></html>"
    r = render.render(html, settle_ms=100)
    assert r["loaded"] is True
    assert any("BOOM" in e for e in r["exceptions"])


def test_raf_detected():
    html = ("<!doctype html><html><body><canvas></canvas>"
            "<script>requestAnimationFrame(()=>{})</script></body></html>")
    r = render.render(html, settle_ms=150)
    assert r["raf_ran"] is True
    assert r["render_sanity"]["has_canvas"] is True


def test_screenshot_returned_when_requested():
    html = "<!doctype html><html><body style='background:#0af'>x</body></html>"
    r = render.render(html, settle_ms=100, want_screenshot=True)
    assert isinstance(r["_screenshot_b64"], str) and len(r["_screenshot_b64"]) > 100
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/local-chatbot && .venv/bin/pytest tests/test_html_demo_render.py -v`
Expected: FAIL — `ModuleNotFoundError: backend.skills.html_demo.render` (or all-skipped if Playwright absent; install it first — see Task 6 / run `.venv/bin/pip install playwright pillow && .venv/bin/playwright install chromium`)

- [ ] **Step 3: Write minimal implementation**

```python
# backend/skills/html_demo/render.py
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
                page.set_content(html, wait_until="load")
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/local-chatbot && .venv/bin/pytest tests/test_html_demo_render.py -v`
Expected: PASS (4 passed) — or 4 skipped if Chromium not installed. If skipped, install per Step 2 note and re-run to confirm PASS.

- [ ] **Step 5: Commit**

```bash
git -C ~/local-chatbot add backend/skills/html_demo/render.py tests/test_html_demo_render.py
git -C ~/local-chatbot commit -m "feat: html-demo headless render + diagnostics"
```

---

### Task 3: Skill module — schemas, dispatch, guidance (`__init__.py`)

**Files:**
- Modify: `backend/skills/html_demo/__init__.py`
- Create: `backend/skills/html_demo/guidance.md`
- Create: `backend/skills/html_demo/examples/bouncing-ball.html`
- Test: `tests/test_html_demo_skill.py`

**Interfaces:**
- Consumes: `DemoStore` (Task 1), `render.available`/`render.render` (Task 2).
- Produces:
  - `HTML_DEMO_TOOL_NAMES: frozenset[str]` = `{"preview_demo","patch_demo","save_demo"}`
  - `available() -> bool`
  - `init_store(demos_dir: Path) -> DemoStore` and `store() -> DemoStore`
  - `tool_schemas() -> list[dict]`
  - `dispatch(name: str, arguments: dict) -> dict`
  - `guidance() -> str`, `examples() -> list[dict]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_html_demo_skill.py
import pytest
from backend.skills import html_demo


@pytest.fixture(autouse=True)
def _store(tmp_path):
    html_demo.init_store(tmp_path)
    yield


def test_tool_schemas_shape():
    names = {t["function"]["name"] for t in html_demo.tool_schemas()}
    assert names == {"preview_demo", "patch_demo", "save_demo"}
    assert names == set(html_demo.HTML_DEMO_TOOL_NAMES)


def test_guidance_is_lean_text():
    g = html_demo.guidance()
    assert "self-contained" in g.lower()
    assert len(g) < 6000  # lean: shares the loop's context budget


def test_examples_present():
    ex = html_demo.examples()
    assert ex and all("html" in e and "name" in e for e in ex)


def test_dispatch_patch_unknown_id_errors():
    r = html_demo.dispatch("patch_demo", {"demo_id": "nope", "edits": []})
    assert r["tool_error"] is True


def test_dispatch_save_roundtrip(tmp_path):
    # create via store directly, then save through dispatch
    did = html_demo.store().create("<html>S</html>", "demo")
    r = html_demo.dispatch("save_demo", {"demo_id": did})
    assert r["saved"] is True
    assert r["preview_url"].startswith("/demos/")


def test_dispatch_unknown_tool_errors():
    r = html_demo.dispatch("nonsense", {})
    assert r["tool_error"] is True


@pytest.mark.skipif(not html_demo.available(), reason="Playwright/Chromium not installed")
def test_dispatch_preview_creates_and_renders():
    r = html_demo.dispatch("preview_demo", {"html": "<h1>hi</h1>", "settle_ms": 100})
    assert r["loaded"] is True
    assert isinstance(r["demo_id"], str)
    # screenshot suppressed unless caller opts in
    assert r.get("_screenshot_b64") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/local-chatbot && .venv/bin/pytest tests/test_html_demo_skill.py -v`
Expected: FAIL — `AttributeError: module 'backend.skills.html_demo' has no attribute 'tool_schemas'`

- [ ] **Step 3: Write minimal implementation**

Create `backend/skills/html_demo/guidance.md` (keep under ~6000 chars — it shares the loop budget):

```markdown
# Self-contained HTML demo — house style

You are building a SINGLE self-contained `.html` file. Follow these rules,
then call `preview_demo` to render and test it. Fix using `patch_demo`.

## Self-contained (non-negotiable)
- One file. Inline ALL CSS in `<style>` and ALL JS in `<script>`.
- No CDNs, no external `<script src>`, no web-fonts, no remote images.
- Embed any asset as a `data:` URI. Must work fully offline.
- Start with `<!doctype html>` and a `<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">`.

## Physics & animation
- Use one `requestAnimationFrame` loop. Track delta-time:
  `let last=performance.now(); function frame(t){ const dt=(t-last)/1000; last=t; /*update(dt); draw();*/ requestAnimationFrame(frame);} requestAnimationFrame(frame);`
- Integrate with dt (position += velocity*dt). Clamp dt (~0.05) to survive tab stalls.
- Simple collisions: AABB overlap or circle distance; resolve by reflecting velocity.
- Ease with `t*t*(3-2*t)` (smoothstep) rather than linear when it reads better.

## Layout
- Prefer `100dvh` over `100vh` (mobile browser chrome). Use `min-height:100dvh`.
- Respect notches: `padding: env(safe-area-inset-top) env(safe-area-inset-right) ...`.
- Use flexbox/grid; avoid fixed pixel widths that overflow small screens.
- Size a full-bleed `<canvas>` to `devicePixelRatio` for crispness.

## Touch (assume a phone)
- Use Pointer Events, not mouse-only: `pointerdown/pointermove/pointerup`.
- Set `touch-action: none` on interactive canvases to stop scroll-hijack.
- `e.preventDefault()` on gesture handlers; test taps and drags.

## Design
- Dark-mode aware: honour `@media (prefers-color-scheme: dark)`.
- Consistent spacing scale (4/8/16/24) and a small type scale.
- Motion with purpose — animate state changes, keep durations 150–300ms.

## Loop discipline (SAVES YOUR CONTEXT)
- Send the FULL html to `preview_demo` ONCE. It returns a `demo_id`.
- To fix, call `patch_demo(demo_id, edits=[{find, replace}])` with SMALL unique
  find-strings — do NOT resend the whole file. If a patch fails, only then
  resend full html via `preview_demo` reusing the same demo_id.
- Read `console_errors`, `exceptions`, and `raf_ran` before editing.
- When it works, call `save_demo(demo_id)`.
```

Create `backend/skills/html_demo/examples/bouncing-ball.html`:

```html
<!doctype html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<style>
  html,body{margin:0;height:100dvh;background:#0b1021;overflow:hidden}
  canvas{display:block;touch-action:none}
</style>
</head>
<body>
<canvas id="c"></canvas>
<script>
const cv = document.getElementById('c'), ctx = cv.getContext('2d');
let W,H,dpr; function resize(){ dpr=devicePixelRatio||1; W=cv.width=innerWidth*dpr; H=cv.height=innerHeight*dpr; }
addEventListener('resize',resize); resize();
let x=W/2,y=H/2,vx=180*dpr,vy=140*dpr,r=24*dpr;
addEventListener('pointerdown',e=>{ x=e.clientX*dpr; y=e.clientY*dpr; vx=(Math.random()*2-1)*300*dpr; vy=(Math.random()*2-1)*300*dpr; });
let last=performance.now();
function frame(t){ const dt=Math.min((t-last)/1000,0.05); last=t;
  x+=vx*dt; y+=vy*dt;
  if(x<r){x=r;vx=-vx} if(x>W-r){x=W-r;vx=-vx}
  if(y<r){y=r;vy=-vy} if(y>H-r){y=H-r;vy=-vy}
  ctx.fillStyle='#0b1021'; ctx.fillRect(0,0,W,H);
  ctx.beginPath(); ctx.arc(x,y,r,0,7); ctx.fillStyle='#4fd1c5'; ctx.fill();
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
</script>
</body>
</html>
```

Fill `backend/skills/html_demo/__init__.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/local-chatbot && .venv/bin/pytest tests/test_html_demo_skill.py -v`
Expected: PASS (6 passed, 1 passed-or-skipped depending on Chromium)

- [ ] **Step 5: Commit**

```bash
git -C ~/local-chatbot add backend/skills/html_demo/__init__.py backend/skills/html_demo/guidance.md backend/skills/html_demo/examples/bouncing-ball.html tests/test_html_demo_skill.py
git -C ~/local-chatbot commit -m "feat: html-demo skill module (schemas, dispatch, guidance)"
```

---

### Task 4: Wire the skill into `/api/chat` + intent injection + `/demos` mount

**Files:**
- Modify: `backend/app.py`
  - imports (near line 19–20)
  - store init + demos dir + mount (near lines 61–70 and 1252)
  - `_full_system_prompt()` (line 182)
  - three-way partition replacing `_partition_cellc_calls` (line 363) + its call site (line 492)
  - the `tool_calls` branch in `relay()` (lines 491–514) — add a demo branch
  - new helpers `_is_html_demo_intent`, `_HTMLDEMO_PROMPT_HINT`, `_inject_html_demo_context`, `_summarize_demo_step` (near the cellc equivalents ~line 899)
  - `MAX_HTMLDEMO_ITERS` constant (near line 307)
  - call `_inject_html_demo_context` in `chat()` (near line 419)
- Test: `tests/test_html_demo_wiring.py`

**Interfaces:**
- Consumes: `html_demo.available/tool_schemas/dispatch/guidance/examples/HTML_DEMO_TOOL_NAMES/init_store` (Task 3); `CAPABILITIES.get(model)` for vision; existing `relay()` machinery.
- Produces:
  - `_is_html_demo_intent(text: str) -> bool`
  - `_inject_html_demo_context(messages: list, last_user: str) -> bool`
  - `_partition_tool_calls(tool_calls) -> tuple[list, list, list]` (cellc, demo, op) — **replaces** `_partition_cellc_calls`
  - `_summarize_demo_step(name: str, result: dict) -> str`
  - UI sentinels: `{"__demo_step__": {"tool","summary"}}` and `{"__demo_card__": {"preview_url","slug"}}`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_html_demo_wiring.py
from backend import app as app_module


def test_is_html_demo_intent():
    assert app_module._is_html_demo_intent("make a self-contained html demo of pong")
    assert app_module._is_html_demo_intent("build me a canvas game")
    assert app_module._is_html_demo_intent("a physics demo with bouncing balls")
    assert not app_module._is_html_demo_intent("what OS are you on?")


def test_partition_tool_calls_three_way():
    calls = [
        {"function": {"name": "cellc_check", "arguments": {}}},
        {"function": {"name": "preview_demo", "arguments": {}}},
        {"function": {"name": "translate", "arguments": {}}},
    ]
    cellc, demo, op = app_module._partition_tool_calls(calls)
    assert [c["function"]["name"] for c in cellc] == ["cellc_check"]
    assert [c["function"]["name"] for c in demo] == ["preview_demo"]
    assert [c["function"]["name"] for c in op] == ["translate"]


def test_summarize_demo_step():
    ok = app_module._summarize_demo_step("preview_demo",
        {"loaded": True, "console_errors": [], "exceptions": [], "raf_ran": True, "demo_id": "demo1"})
    assert "demo1" in ok and "✓" in ok
    bad = app_module._summarize_demo_step("preview_demo",
        {"loaded": True, "console_errors": ["error: x"], "exceptions": ["Boom"], "raf_ran": False, "demo_id": "demo1"})
    assert "1 err" in bad or "error" in bad.lower()
    saved = app_module._summarize_demo_step("save_demo", {"saved": True, "slug": "pong"})
    assert "pong" in saved


def test_inject_html_demo_context_on_intent(monkeypatch):
    monkeypatch.setattr(app_module.html_demo, "available", lambda: True)
    monkeypatch.setattr(app_module.html_demo, "guidance", lambda: "GUIDANCE_TOKEN")
    monkeypatch.setattr(app_module.html_demo, "examples",
                        lambda: [{"name": "x", "html": "EXAMPLE_TOKEN"}])
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "make an html demo"}]
    hit = app_module._inject_html_demo_context(messages, "make an html demo")
    assert hit is True
    assert "GUIDANCE_TOKEN" in messages[0]["content"]
    assert "EXAMPLE_TOKEN" in messages[0]["content"]


def test_inject_html_demo_context_skips_non_intent(monkeypatch):
    monkeypatch.setattr(app_module.html_demo, "available", lambda: True)
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hello"}]
    assert app_module._inject_html_demo_context(messages, "hello") is False
    assert messages[0]["content"] == "sys"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/local-chatbot && .venv/bin/pytest tests/test_html_demo_wiring.py -v`
Expected: FAIL — `AttributeError: module 'backend.app' has no attribute '_is_html_demo_intent'`

- [ ] **Step 3: Write minimal implementation**

**(3a)** Add the import after line 20 (`from backend.bridges import open_palette`):

```python
from backend.skills import html_demo
```

**(3b)** After the `OUTPUT_DIR` block (after line 70), add the demos dir + store init:

```python
DEMOS_DIR = ROOT / "demos"
try:
    DEMOS_DIR.mkdir(parents=True, exist_ok=True)
    html_demo.init_store(DEMOS_DIR)
    log.info("html_demo store: %s (render %s)",
             DEMOS_DIR, "available" if html_demo.available() else "unavailable")
except Exception as exc:  # never block boot on the demo feature
    log.warning("html_demo init failed: %s", exc)
```

**(3c)** Mount `/demos` right after the `/static` mount (line 1252):

```python
app.mount("/demos", StaticFiles(directory=DEMOS_DIR), name="demos")
```

**(3d)** Add the iteration cap near `MAX_CELLC_ITERS = 5` (line 307):

```python
MAX_HTMLDEMO_ITERS = 4
```

**(3e)** In `_full_system_prompt()` (line 182), add the stub hint. Change the body to:

```python
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
```

**(3f)** Replace `_partition_cellc_calls` (lines 363–369) with a three-way partition:

```python
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
```

**(3g)** Add the demo helpers next to the cellc ones (after `_inject_cellc_context`, ~line 928):

```python
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
        return f"{name} → ✗ {errs} err ({did})"
    raf = "raf✓" if result.get("raf_ran") else "static"
    return f"{name} → ✓ clean, {raf} ({did})"
```

**(3h)** In `chat()` after the cellc injection block (after line 419), add:

```python
    html_demo_chat = False
    if html_demo.available():
        last_user = next((m.get("content", "") for m in reversed(incoming)
                          if m.get("role") == "user"), "")
        html_demo_chat = _inject_html_demo_context(messages, last_user)
```

Also compute the vision flag once (near where `model` is used for the body, after line 451):

```python
    _caps = CAPABILITIES.get(model) or {}
    _is_vision = bool(_caps.get("vision"))
```

**(3i)** In `relay()`, update the partition call and add the demo branch. Change line 492 from `cellc_calls, op_calls = _partition_cellc_calls(value)` to:

```python
                    cellc_calls, demo_calls, op_calls = _partition_tool_calls(value)
```

Then, immediately after the `if op_calls:` emit (after line 494) and BEFORE the cellc block, add the demo branch:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/local-chatbot && .venv/bin/pytest tests/test_html_demo_wiring.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Run the full suite + boot smoke check**

Run: `cd ~/local-chatbot && .venv/bin/pytest -q && .venv/bin/python -c "from backend import app; print('boot ok'); print('/demos' in [r.path for r in app.app.routes])"`
Expected: whole suite green; prints `boot ok` and `True`.

- [ ] **Step 6: Commit**

```bash
git -C ~/local-chatbot add backend/app.py tests/test_html_demo_wiring.py
git -C ~/local-chatbot commit -m "feat: wire html-demo skill into chat loop + /demos mount"
```

---

### Task 5: Frontend — demo step chips + preview card + full-screen iframe

**Files:**
- Modify: `frontend/app.js` (add sentinel parsers near `parseToolCallSentinel` ~line 788 and the cellc-step parser ~line 824; wire into `renderAssistant` ~line 896)
- Modify: `frontend/style.css` (card + overlay styles)
- Modify: `frontend/index.html` (a `<div id="demo-overlay">` host for the full-screen iframe)
- Test: manual (browser) — no JS unit harness exists in this repo.

**Interfaces:**
- Consumes: backend sentinels `{"__demo_step__": {tool,summary}}` and `{"__demo_card__": {preview_url,slug}}` (Task 4).
- Produces: `parseDemoSteps(text)`, `parseDemoCard(text)`, `openDemoOverlay(url)` in `frontend/app.js`.

- [ ] **Step 1: Add the parsers** (in `frontend/app.js`, after the `parseCellcSteps`/`renderCellcSteps` block ~line 841)

```javascript
// Extract ALL `{"__demo_step__": {...}}` sentinels (interleave across iterations).
function parseDemoSteps(text) {
  const steps = [];
  let cleaned = [];
  for (const line of text.split("\n")) {
    const t = line.trim();
    if (t.startsWith('{"__demo_step__"')) {
      try { steps.push(JSON.parse(t).__demo_step__); continue; } catch {}
    }
    if (t.startsWith('{"__demo_card__"')) { continue; } // handled separately
    cleaned.push(line);
  }
  return { steps, cleanedText: cleaned.join("\n") };
}

// The final save emits one `{"__demo_card__": {preview_url, slug}}` line.
function parseDemoCard(text) {
  const idx = text.lastIndexOf('{"__demo_card__"');
  if (idx === -1) return { card: null, cleanedText: text };
  const end = text.indexOf("\n", idx);
  const line = (end === -1 ? text.slice(idx) : text.slice(idx, end)).trim();
  try {
    const card = JSON.parse(line).__demo_card__ || null;
    return { card, cleanedText: (text.slice(0, idx) + (end === -1 ? "" : text.slice(end))).trim() };
  } catch { return { card: null, cleanedText: text }; }
}

function renderDemoSteps(steps) {
  if (!steps || !steps.length) return "";
  const items = steps.map((s) => `<li class="demo-step">${escapeHtml(s.summary || "")}</li>`).join("");
  return `<ul class="demo-steps">${items}</ul>`;
}

function renderDemoCard(card) {
  if (!card || !card.preview_url) return "";
  const url = escapeHtml(card.preview_url);
  const slug = escapeHtml(card.slug || "demo");
  return `<button class="demo-card" data-demo-url="${url}">▶ Preview demo: ${slug}</button>`;
}

function openDemoOverlay(url) {
  const host = document.getElementById("demo-overlay");
  if (!host) return;
  host.innerHTML = `<div class="demo-overlay-bar">
      <button id="demo-overlay-close">✕ Close</button>
      <a href="${url}" target="_blank" rel="noopener">Open in new tab ↗</a>
    </div>
    <iframe class="demo-overlay-frame" src="${url}"></iframe>`;
  host.style.display = "flex";
  document.getElementById("demo-overlay-close").onclick = () => {
    host.style.display = "none";
    host.innerHTML = "";
  };
}
```

**Note:** reuse the existing `escapeHtml` helper in app.js. If it is named differently, match the existing name used by `renderCellcSteps`.

- [ ] **Step 2: Wire into `renderAssistant`** (in `frontend/app.js` ~line 896). Where cellc steps are already parsed and stripped, add demo parsing in the same place, before the final markdown render. Insert:

```javascript
  // html-demo skill: pull step chips + the final preview card out of the stream.
  const demo = parseDemoSteps(acc);
  acc = demo.cleanedText;
  const demoCardParsed = parseDemoCard(acc);
  acc = demoCardParsed.cleanedText;
  const demoStepsHtml = renderDemoSteps(demo.steps);
  const demoCardHtml = renderDemoCard(demoCardParsed.card);
```

Then append `demoStepsHtml` and `demoCardHtml` into the assistant bubble's HTML alongside the existing cellc-steps output (mirror exactly how `renderCellcSteps(...)` output is concatenated into `out.innerHTML` in this function).

- [ ] **Step 3: Delegate the card click** (near the other global click handlers in app.js; search for an existing `addEventListener("click"` delegation block and add a branch, or add this once at init):

```javascript
document.addEventListener("click", (e) => {
  const btn = e.target.closest(".demo-card");
  if (btn && btn.dataset.demoUrl) openDemoOverlay(btn.dataset.demoUrl);
});
```

- [ ] **Step 4: Add the overlay host** (in `frontend/index.html`, just before the closing `</body>`):

```html
<div id="demo-overlay"></div>
```

- [ ] **Step 5: Add styles** (in `frontend/style.css`, end of file):

```css
.demo-steps { list-style: none; margin: 8px 0; padding: 0; font-family: monospace; font-size: 12px; opacity: 0.85; }
.demo-step { padding: 2px 0; }
.demo-card { display: inline-block; margin: 8px 0; padding: 10px 16px; border: 1px solid currentColor;
  border-radius: 8px; background: transparent; color: inherit; cursor: pointer; font-size: 14px; }
.demo-card:hover { opacity: 0.8; }
#demo-overlay { display: none; position: fixed; inset: 0; z-index: 1000; flex-direction: column;
  background: #000; padding-top: env(safe-area-inset-top); }
.demo-overlay-bar { display: flex; justify-content: space-between; align-items: center;
  padding: 10px 16px; background: #111; color: #fff; }
.demo-overlay-bar a, .demo-overlay-bar button { color: #fff; background: transparent; border: none;
  font-size: 15px; cursor: pointer; }
.demo-overlay-frame { flex: 1; width: 100%; border: none; background: #fff; }
```

- [ ] **Step 6: Manual verification**

```bash
cd ~/local-chatbot && .venv/bin/uvicorn backend.app:app --host 127.0.0.1 --port 8000
```
Then, in a browser at `http://127.0.0.1:8000`:
1. Select a native-tool model (e.g. `devstral-small-2:24b`).
2. Prompt: "make a self-contained html demo: a ball that bounces and follows my taps."
3. Expect: `demo-step` chips appear during the loop (`preview_demo → ✓ clean, raf✓ (demo1)`), then a `▶ Preview demo:` card.
4. Tap the card → full-screen iframe renders the demo; Close returns to chat.
5. Confirm the file serves: `curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/demos/<slug>.html` → `200`.
6. Confirm a saved file exists: `ls ~/local-chatbot/demos/`.

- [ ] **Step 7: Commit**

```bash
git -C ~/local-chatbot add frontend/app.js frontend/style.css frontend/index.html
git -C ~/local-chatbot commit -m "feat: html-demo preview card + full-screen iframe in PWA"
```

---

### Task 6: Installer audit + gitignore + docs

**Files:**
- Modify: `install.sh` (add a Playwright/Chromium + Pillow audit alongside the existing dependency checks)
- Modify: `.gitignore` (ignore the generated `demos/` output)
- Modify: `README.md` (one paragraph documenting the html-demo skill)

**Interfaces:**
- Consumes: nothing. Pure ops/docs.
- Produces: nothing importable.

- [ ] **Step 1: Add render deps to the installer**

In `install.sh`, alongside the other "audit what's installed vs needed" checks, add a check that mirrors the existing style (colored output, prompt-before-install, honor an override env var). The concrete commands it should offer:

```bash
# HTML-demo skill (headless render): Playwright + Chromium + Pillow
if ! "$VENV_PY" -c "import playwright" 2>/dev/null; then
  echo "  html-demo render needs: pip install playwright pillow && playwright install chromium"
  # (follow the script's existing prompt-then-install pattern; skip if user declines)
fi
"$VENV_PY" -m playwright install chromium  # no-op if already present
```

Match the surrounding functions' naming and the `$VENV_PY` / venv-path variable the script already uses. The skill must remain optional: if the user declines, the app still boots (render `available()` returns False).

- [ ] **Step 2: Ignore generated demos**

Add to `.gitignore`:

```
demos/
```

- [ ] **Step 3: Document the skill in the README**

Add under "What it does" in `README.md`:

```markdown
- **Build & test self-contained HTML demos** — ask a tool-capable model for a
  demo (game, physics, animation) and it writes the HTML, renders it in a
  headless browser, reads back diagnostics (console errors, whether the
  animation loop ran) and — if the model has vision — a screenshot, iterates
  via small patches to stay within context, then saves it. Tap the preview
  card to see it full-screen on your phone. Requires Playwright + Chromium
  (the installer offers to set these up); degrades gracefully without them.
```

- [ ] **Step 4: Verify boot still clean**

Run: `cd ~/local-chatbot && .venv/bin/pytest -q && .venv/bin/python -c "from backend import app; print('ok')"`
Expected: suite green, prints `ok`.

- [ ] **Step 5: Commit**

```bash
git -C ~/local-chatbot add install.sh .gitignore README.md
git -C ~/local-chatbot commit -m "chore: html-demo installer audit, gitignore, docs"
```

---

## Self-Review

**Spec coverage:**
- Full agentic loop → Tasks 2–4 (render, dispatch, relay loop). ✓
- Layered feedback (text always, screenshot if vision) → render `want_screenshot` + Task 4 `_is_vision` gating. ✓
- Vision critic = chat model only if vision → Task 4 `_is_vision = CAPABILITIES.get(model)["vision"]`, screenshot attached only then. ✓
- Save + inline preview → Task 1 `save`, Task 4 `__demo_card__`, Task 5 overlay iframe. ✓
- Bridge cut from cellc → Task 3 interface parity; Task 4 mirrors relay/partition/inject. ✓
- Context strategy (handle store, patch+fallback, progressive disclosure, lean results, iteration cap) → Task 1 store, Task 3 dispatch, Task 4 `_inject_html_demo_context` + `MAX_HTMLDEMO_ITERS` + `_screenshot_b64` stripped from tool text. ✓
- Three tools with exact names → Task 3 `HTML_DEMO_TOOL_NAMES`. ✓
- Non-tool models via `op:` path → unaffected: three-way partition keeps demo names out of `op_calls`; non-tool models simply never emit them and still get guidance via injection. ✓
- Graceful degrade without Playwright → `render.available()` gate, Task 4 boot guard, Task 6 optional install. ✓
- Diagnostics fields (`loaded/console_errors/exceptions/render_sanity/raf_ran/viewport`) → Task 2 render return. ✓
- Time-based capture (settle_ms + interactions + raf_ran) → Task 2. ✓
- Module layout → Tasks 1–3 match spec's `backend/skills/html_demo/` tree. ✓

**Placeholder scan:** No TBD/TODO. Task 5 (frontend) and Task 6 (installer) intentionally say "mirror the existing pattern" for concatenation/prompt style because they must match unread local conventions; the concrete code/commands to add are given in full. Acceptable — the actual content is present, only the insertion styling defers to surroundings.

**Type consistency:** `demo_id` (str) consistent across store/dispatch/relay. `_screenshot_b64` produced in render, popped in relay. `HTML_DEMO_TOOL_NAMES` defined in Task 3, consumed in Task 4. `_partition_tool_calls` returns 3-tuple, call site updated. Sentinel keys `__demo_step__`/`__demo_card__` consistent between Task 4 (emit) and Task 5 (parse). ✓

## Open verification items (run during execution, not blockers)
- Confirm `escapeHtml` helper's exact name in app.js (Task 5 Step 1 note).
- Confirm the venv python path variable name in `install.sh` (Task 6 Step 1).
- Confirm the exact concatenation site for step/card HTML in `renderAssistant` (Task 5 Step 2).
