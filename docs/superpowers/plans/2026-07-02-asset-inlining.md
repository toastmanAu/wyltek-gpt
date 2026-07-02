# Asset Inlining (Self-Contained HTML) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a chat model produce a genuinely self-contained HTML page that embeds uploaded/workspace images as base64 data URIs, with zero base64 ever entering the model's context.

**Architecture:** The model writes ordinary HTML referencing images as `src="asset:<filename>"`. A pure backend function `inline_assets(html, workspace)` swaps each `asset:` reference for a `data:<mime>;base64,…` URI, guarded to the session workspace. It is exposed as `POST /api/inline-assets` and called (a) by the web UI's code-block download handler before it builds the download Blob, and (b) directly by headless agents. Substitution happens at the artifact boundary — after the model is done — so base64 never costs context.

**Tech Stack:** Python 3 / FastAPI (backend), vanilla JS (frontend), pytest + FastAPI `TestClient` (tests). No new dependencies.

## Global Constraints

- **Images only in v1** — inline image references; do not inline `<script src>` or `<link>`.
- **Workspace-scoped reads only** — a referenced path that resolves outside the caller's session workspace is rejected, never read. Reuse the resolve-then-`relative_to` guard already used by `_resolve_in_workspace`.
- **Fail loud, never silent** — a missing / oversize / out-of-workspace reference leaves the literal `asset:` token in place and is reported in `missing[]`. Never drop a reference silently or emit a broken page without signalling.
- **Per-asset size cap: 5 MB** (`MAX_ASSET_BYTES = 5 * 1024 * 1024`). **Total input HTML cap at the endpoint: 5 MB.**
- **No base64 in model context** — substitution is server-side only; the model never receives encoded bytes.
- **MIME by extension**, unknown extension → `application/octet-stream` (still valid, still self-contained).
- Match existing import style: siblings imported as `from backend.<module> import …` (see `backend/app.py:19-33`).

---

### Task 1: `inline_assets` core function

**Files:**
- Create: `backend/inlining.py`
- Test: `tests/test_inline_assets.py`

**Interfaces:**
- Consumes: nothing (leaf module; stdlib only).
- Produces:
  - `MAX_ASSET_BYTES: int` (= 5 MB)
  - `@dataclass(frozen=True) class InlineResult: html: str; replaced: list[str]; missing: list[dict]`
  - `def inline_assets(html: str, workspace: Path, max_bytes: int = MAX_ASSET_BYTES) -> InlineResult`
  - Each `missing` entry is `{"name": <raw ref>, "reason": <"not found"|"outside workspace"|"exceeds size cap">}`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_inline_assets.py`:

```python
import base64
from pathlib import Path

import pytest

from backend.inlining import InlineResult, inline_assets, MAX_ASSET_BYTES

# A 1x1 transparent PNG (68 bytes decoded).
_PNG_1x1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def _write(ws: Path, name: str, data: bytes) -> None:
    p = ws / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


def test_inlines_src_reference_as_png_data_uri(tmp_path):
    _write(tmp_path, "photo.png", _PNG_1x1)
    html = '<img src="asset:photo.png">'
    result = inline_assets(html, tmp_path)
    expected_b64 = base64.b64encode(_PNG_1x1).decode("ascii")
    assert f"data:image/png;base64,{expected_b64}" in result.html
    assert "asset:photo.png" not in result.html
    assert result.replaced == ["photo.png"]
    assert result.missing == []


def test_href_and_css_url_forms_are_substituted(tmp_path):
    _write(tmp_path, "a.gif", _PNG_1x1)
    _write(tmp_path, "b.webp", _PNG_1x1)
    html = "<a href=\"asset:a.gif\">x</a><style>div{background:url(asset:b.webp)}</style>"
    result = inline_assets(html, tmp_path)
    assert "data:image/gif;base64," in result.html
    assert "data:image/webp;base64," in result.html
    assert "asset:" not in result.html
    assert sorted(result.replaced) == ["a.gif", "b.webp"]


@pytest.mark.parametrize(
    "name,mime",
    [
        ("x.png", "image/png"),
        ("x.jpg", "image/jpeg"),
        ("x.jpeg", "image/jpeg"),
        ("x.gif", "image/gif"),
        ("x.webp", "image/webp"),
        ("x.svg", "image/svg+xml"),
    ],
)
def test_mime_by_extension(tmp_path, name, mime):
    _write(tmp_path, name, _PNG_1x1)
    result = inline_assets(f'<img src="asset:{name}">', tmp_path)
    assert f"data:{mime};base64," in result.html


def test_unknown_extension_is_octet_stream(tmp_path):
    _write(tmp_path, "blob.xyz", _PNG_1x1)
    result = inline_assets('<img src="asset:blob.xyz">', tmp_path)
    assert "data:application/octet-stream;base64," in result.html
    assert result.replaced == ["blob.xyz"]


def test_missing_file_leaves_token_and_reports(tmp_path):
    html = '<img src="asset:nope.png">'
    result = inline_assets(html, tmp_path)
    assert result.html == html  # untouched
    assert result.replaced == []
    assert result.missing == [{"name": "nope.png", "reason": "not found"}]


def test_path_traversal_is_rejected(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (tmp_path / "secret.txt").write_bytes(b"topsecret")
    html = '<img src="asset:../secret.txt">'
    result = inline_assets(html, ws)
    assert result.html == html  # untouched, secret not embedded
    assert "topsecret" not in result.html
    assert result.missing == [{"name": "../secret.txt", "reason": "outside workspace"}]


def test_oversize_asset_is_rejected(tmp_path):
    _write(tmp_path, "big.png", b"\x00" * (MAX_ASSET_BYTES + 1))
    html = '<img src="asset:big.png">'
    result = inline_assets(html, tmp_path)
    assert result.html == html
    assert result.missing == [{"name": "big.png", "reason": "exceeds size cap"}]


def test_same_reference_twice_reads_once_and_lists_once(tmp_path):
    _write(tmp_path, "dup.png", _PNG_1x1)
    html = '<img src="asset:dup.png"><img src="asset:dup.png">'
    result = inline_assets(html, tmp_path)
    assert result.html.count("data:image/png;base64,") == 2
    assert result.replaced == ["dup.png"]  # deduped in the report


def test_returns_inline_result_type(tmp_path):
    result = inline_assets("<p>no assets</p>", tmp_path)
    assert isinstance(result, InlineResult)
    assert result.html == "<p>no assets</p>"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/phill/local-chatbot && .venv/bin/pytest tests/test_inline_assets.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.inlining'`.

- [ ] **Step 3: Write the implementation**

Create `backend/inlining.py`:

```python
"""Inline workspace image references into self-contained HTML.

The chat model authors HTML referencing uploaded images as
``src="asset:<filename>"``. This module rewrites each such reference to a
``data:<mime>;base64,...`` URI, reading only files that resolve *inside* the
caller's session workspace. Substitution happens at the artifact boundary
(download / API response), so the base64 never enters the model's context.

Fail-loud contract: a reference that is missing, oversize, or escapes the
workspace is left as the literal ``asset:`` token and reported in ``missing``.
"""
from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from pathlib import Path

MAX_ASSET_BYTES = 5 * 1024 * 1024  # 5 MB per asset

_MIME_BY_EXT = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
    "svg": "image/svg+xml",
    "bmp": "image/bmp",
    "ico": "image/x-icon",
    "avif": "image/avif",
}

# Capture the reference after ``asset:`` up to the closing quote, paren,
# whitespace, or ``>``. Works inside src="...", href='...', and url(...).
_ASSET_RE = re.compile(r"""asset:([^"'\)\s>]+)""")


@dataclass(frozen=True)
class InlineResult:
    html: str
    replaced: list[str]
    missing: list[dict]


def _mime_for(suffix: str) -> str:
    ext = suffix.lstrip(".").lower()
    return _MIME_BY_EXT.get(ext, "application/octet-stream")


def inline_assets(
    html: str, workspace: Path, max_bytes: int = MAX_ASSET_BYTES
) -> InlineResult:
    workspace = workspace.resolve()
    cache: dict[str, str | None] = {}
    replaced: list[str] = []
    missing: list[dict] = []

    def _fail(raw: str, reason: str) -> None:
        cache[raw] = None
        missing.append({"name": raw, "reason": reason})

    def _repl(m: re.Match) -> str:
        raw = m.group(1)
        if raw in cache:
            val = cache[raw]
            return val if val is not None else m.group(0)

        try:
            path = (workspace / raw).resolve()
            path.relative_to(workspace)
        except ValueError:
            _fail(raw, "outside workspace")
            return m.group(0)

        if not path.is_file():
            _fail(raw, "not found")
            return m.group(0)
        if path.stat().st_size > max_bytes:
            _fail(raw, "exceeds size cap")
            return m.group(0)

        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        data_uri = f"data:{_mime_for(path.suffix)};base64,{b64}"
        cache[raw] = data_uri
        replaced.append(raw)
        return data_uri

    out = _ASSET_RE.sub(_repl, html)
    return InlineResult(html=out, replaced=replaced, missing=missing)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/phill/local-chatbot && .venv/bin/pytest tests/test_inline_assets.py -q`
Expected: PASS — all tests green.

- [ ] **Step 5: Commit**

```bash
cd /home/phill/local-chatbot
git add backend/inlining.py tests/test_inline_assets.py
git commit -m "feat: inline_assets — asset: refs to base64 data URIs, workspace-guarded"
```

---

### Task 2: `POST /api/inline-assets` endpoint

**Files:**
- Modify: `backend/app.py` (add import near line 19-33; add endpoint near the other `/api/…` routes, e.g. after `run_operation` which ends at line 785)
- Test: `tests/test_inline_assets_endpoint.py`

**Interfaces:**
- Consumes: `inline_assets`, `InlineResult` from `backend.inlining` (Task 1); `_resolve_in_workspace(session_id, name)` from `backend/app.py:1066` (returns `(session_dir, file_path)`; raises `HTTPException(400)` on a bad `session_id`).
- Produces: route `POST /api/inline-assets`, body `{session_id: str, html: str}` → `200 {html, replaced, missing}`; `400` on missing/oversize `html`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_inline_assets_endpoint.py`:

```python
import base64

from fastapi.testclient import TestClient

from backend import app as app_module

client = TestClient(app_module.app)

_PNG_1x1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def _seed_workspace(tmp_path, monkeypatch, session_id, name, data):
    ws = tmp_path / "ws"
    (ws / session_id).mkdir(parents=True, exist_ok=True)
    (ws / session_id / name).write_bytes(data)
    monkeypatch.setattr(app_module, "WORKSPACE", ws.resolve())
    return ws


def test_endpoint_inlines_uploaded_image(tmp_path, monkeypatch):
    _seed_workspace(tmp_path, monkeypatch, "s1", "photo.png", _PNG_1x1)
    r = client.post(
        "/api/inline-assets",
        json={"session_id": "s1", "html": '<img src="asset:photo.png">'},
    )
    assert r.status_code == 200
    body = r.json()
    assert "data:image/png;base64," in body["html"]
    assert body["replaced"] == ["photo.png"]
    assert body["missing"] == []


def test_endpoint_missing_asset_does_not_500(tmp_path, monkeypatch):
    _seed_workspace(tmp_path, monkeypatch, "s1", "photo.png", _PNG_1x1)
    r = client.post(
        "/api/inline-assets",
        json={"session_id": "s1", "html": '<img src="asset:ghost.png">'},
    )
    assert r.status_code == 200
    assert r.json()["missing"] == [{"name": "ghost.png", "reason": "not found"}]


def test_endpoint_rejects_missing_html():
    r = client.post("/api/inline-assets", json={"session_id": "s1"})
    assert r.status_code == 400


def test_endpoint_rejects_oversize_html():
    r = client.post(
        "/api/inline-assets",
        json={"session_id": "s1", "html": "x" * (5_000_001)},
    )
    assert r.status_code == 400
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/phill/local-chatbot && .venv/bin/pytest tests/test_inline_assets_endpoint.py -q`
Expected: FAIL — `404 Not Found` for the route (assertions on 200 fail).

- [ ] **Step 3: Add the import**

In `backend/app.py`, add alongside the other `from backend.…` imports (after line 24, before the `from backend.operations import (` block at line 25):

```python
from backend.inlining import inline_assets
```

- [ ] **Step 4: Add the endpoint**

In `backend/app.py`, immediately after the `run_operation` function (it ends at line 785, before `_CELLC_SAVE_NAME_RE` at line 788), insert:

```python
_MAX_INLINE_HTML_BYTES = 5_000_000  # bound work; also the per-asset cap in inlining.py


@app.post("/api/inline-assets")
async def inline_assets_endpoint(payload: dict):
    """Inline ``asset:<filename>`` references in the given HTML to base64 data
    URIs, reading only files inside the caller's session workspace. Shared by
    the web UI's download button and by headless agents. base64 never touches
    the model's context — this runs on an already-authored artifact."""
    html = payload.get("html")
    if not isinstance(html, str) or not html:
        raise HTTPException(400, "missing 'html'")
    if len(html) > _MAX_INLINE_HTML_BYTES:
        raise HTTPException(400, "html too large")

    session_id = payload.get("session_id", "default")
    session_dir, _ = _resolve_in_workspace(session_id, "_")

    result = await asyncio.to_thread(inline_assets, html, session_dir)
    return {
        "html": result.html,
        "replaced": result.replaced,
        "missing": result.missing,
    }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /home/phill/local-chatbot && .venv/bin/pytest tests/test_inline_assets_endpoint.py -q`
Expected: PASS — all four tests green.

- [ ] **Step 6: Commit**

```bash
cd /home/phill/local-chatbot
git add backend/app.py tests/test_inline_assets_endpoint.py
git commit -m "feat: POST /api/inline-assets — shared asset-inlining endpoint"
```

---

### Task 3: Web UI download wiring

**Files:**
- Modify: `frontend/app.js` — `appendCodeDownloads` (line 1083) and its one caller (line 1591).

**Interfaces:**
- Consumes: `POST /api/inline-assets` (Task 2); the existing module-level `SESSION` constant (see `frontend/app.js:574`); `extractCodeBlocks(text)` returning `[{filename, body, lang}]` (line 1031).
- Produces: no new exports. Behaviour change: `html`/`svg` download blocks are inlined server-side before their Blob is built.

- [ ] **Step 1: Make `appendCodeDownloads` inline html/svg blocks before building the Blob**

Replace the body of `appendCodeDownloads` (currently `frontend/app.js:1083-1103`) with:

```javascript
async function appendCodeDownloads(msgWrap, blocks) {
  if (!msgWrap || !blocks.length) return;
  const row = document.createElement("div");
  row.className = "code-downloads";
  for (const b of blocks) {
    let body = b.body;
    // Self-contained HTML/SVG: swap asset:<file> refs for base64 data URIs
    // server-side (base64 never enters the model's context). Best-effort —
    // on any failure we fall back to the raw text so the download never breaks.
    if ((b.lang === "html" || b.lang === "svg") && /asset:/.test(body)) {
      try {
        const r = await fetch("/api/inline-assets", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_id: SESSION, html: body }),
        });
        if (r.ok) {
          const data = await r.json();
          body = data.html;
          if (data.missing && data.missing.length) {
            const names = data.missing.map((m) => m.name).join(", ");
            const warn = document.createElement("div");
            warn.className = "code-download-warn";
            warn.textContent = `⚠ could not embed: ${names}`;
            row.appendChild(warn);
          }
        }
      } catch {
        // keep raw body; download still works, just not self-contained
      }
    }
    const blob = new Blob([body], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.className = "code-download";
    a.href = url;
    a.download = b.filename;
    a.textContent = `⬇ ${b.filename}`;
    a.title = `${body.length} bytes · ${b.lang || "text"}`;
    if (IS_IOS_PWA) {
      a.addEventListener("click", (e) => shareInsteadOfDownload(e, url, b.filename));
    }
    row.appendChild(a);
  }
  msgWrap.appendChild(row);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}
```

- [ ] **Step 2: Update the caller so the now-async function isn't lost**

At `frontend/app.js:1591`, the call is currently:

```javascript
    appendCodeDownloads(out.parentElement, extractCodeBlocks(scanText));
```

Change it to explicitly ignore the returned promise (behaviour is fire-and-forget; the download row fills in when inlining resolves):

```javascript
    void appendCodeDownloads(out.parentElement, extractCodeBlocks(scanText));
```

- [ ] **Step 3: Manual verification (no JS unit harness in this repo)**

Start the app and exercise the real flow:

```bash
cd /home/phill/local-chatbot && .venv/bin/uvicorn backend.app:app --port 8000
```

Then in the browser at `http://localhost:8000`:
1. Upload a small PNG (e.g. `photo.png`).
2. Ask the model: *"Make a self-contained HTML page that shows photo.png. Reference it as `asset:photo.png`."*
3. Click the `⬇ …html` download button; open the saved file.

Expected: the saved HTML contains `data:image/png;base64,` and **no** literal `asset:photo.png`; the image renders with the file opened directly from disk (no server).

Sanity-check the endpoint without the browser:

```bash
# with the app running and photo.png uploaded to session "default":
curl -s localhost:8000/api/inline-assets \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"default","html":"<img src=\"asset:photo.png\">"}' | head -c 120
```

Expected: JSON whose `html` starts to show `<img src="data:image/png;base64,`.

- [ ] **Step 4: Commit**

```bash
cd /home/phill/local-chatbot
git add frontend/app.js
git commit -m "feat: inline asset: image refs into html/svg downloads (self-contained pages)"
```

---

### Task 4: Model guidance for the `asset:` convention

**Files:**
- Modify: `backend/app.py` — `_operations_prompt_block` (line 142).
- Modify: `frontend/app.js` — upload-injection system message (line 593).
- Test: `tests/test_inline_assets_prompt.py`

**Interfaces:**
- Consumes: `_operations_prompt_block() -> str` (existing, `backend/app.py:142`).
- Produces: no new symbols; asserts the guidance text is present in the prompt block.

- [ ] **Step 1: Write the failing test**

Create `tests/test_inline_assets_prompt.py`:

```python
from backend import app as app_module


def test_prompt_block_teaches_asset_convention():
    block = app_module._operations_prompt_block()
    # Only asserted when operations are enabled in this environment; the
    # convention text must be present whenever the block is non-empty.
    if block:
        assert "asset:" in block
        assert "self-contained" in block.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/phill/local-chatbot && .venv/bin/pytest tests/test_inline_assets_prompt.py -q`
Expected: FAIL — `assert "asset:" in block` (the block does not yet mention it), *when operations are enabled*. (If the env has no operations enabled the test is a no-op; enable at least one op, or trust CI where ops are present.)

- [ ] **Step 3: Add the guidance to the prompt block**

In `backend/app.py`, inside `_operations_prompt_block`, extend the `lines` list. After the existing line `"        source param.",` (the closing sentence of the invoke instructions, near line 168) and before the blank line preceding `"Available operations:"`, insert:

```python
        "",
        "### Embedding images in self-contained HTML",
        "When the user asks for a single self-contained HTML file that shows an",
        "uploaded image, reference the image as src=\"asset:<filename>\" (using the",
        "exact uploaded filename). The system replaces each asset: reference with an",
        "inline base64 data URI when the page is downloaded, so the resulting file",
        "needs no external images. Do NOT paste base64 yourself — just write",
        "asset:<filename>.",
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/phill/local-chatbot && .venv/bin/pytest tests/test_inline_assets_prompt.py -q`
Expected: PASS.

- [ ] **Step 5: Extend the upload-injection note for images**

In `frontend/app.js`, the upload-injection message at line 593 currently reads:

```javascript
      content: `User uploaded file "${meta.name}" (${formatBytes(meta.size)}) — available in the workspace as "${meta.name}". When the user asks you to process, convert, edit, or modify this file, use one of the available operations with "${meta.name}" as the source.`,
```

Replace it with a version that appends the `asset:` hint for image uploads:

```javascript
      content: `User uploaded file "${meta.name}" (${formatBytes(meta.size)}) — available in the workspace as "${meta.name}". When the user asks you to process, convert, edit, or modify this file, use one of the available operations with "${meta.name}" as the source.${
        /\.(png|jpe?g|gif|webp|svg|bmp|avif)$/i.test(meta.name)
          ? ` To embed this image in a self-contained HTML page, reference it as src="asset:${meta.name}".`
          : ""
      }`,
```

- [ ] **Step 6: Run the full suite**

Run: `cd /home/phill/local-chatbot && .venv/bin/pytest -q`
Expected: PASS — new tests plus the existing suite stay green.

- [ ] **Step 7: Commit**

```bash
cd /home/phill/local-chatbot
git add backend/app.py frontend/app.js tests/test_inline_assets_prompt.py
git commit -m "feat: teach models the asset: convention for self-contained HTML"
```

---

## Notes for the implementer

- **Run tests with the project venv:** `.venv/bin/pytest` (the repo ships a `.venv`; `requirements-dev.txt` pins `pytest>=8.0`).
- **`asset:` regex scope:** `asset:([^"'\)\s>]+)` captures a bare filename ending at the first quote, `)`, whitespace, or `>`. This is deliberately simple; subfolder refs like `asset:img/logo.png` resolve fine and stay inside the workspace guard.
- **Traversal guard is resolve-then-`relative_to`, not basename.** Do not `Path(name).name` the reference — that would silently rewrite `../secret` to `secret`. The test `test_path_traversal_is_rejected` locks this behaviour in.
- **Frontend has no unit-test harness** in this repo; Task 3 uses manual + `curl` verification, matching how the drop-folder work was verified (Playwright smoke, manual HTTP).
