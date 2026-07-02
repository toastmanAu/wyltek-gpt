# Self-Contained HTML — asset inlining design spec

**Date:** 2026-07-02
**Status:** Approved, ready for implementation plan

## Problem

When a chat model (e.g. qwen3.6 27b) is asked to build a *self-contained* HTML
page that displays an uploaded image, the image must be embedded as a base64
data URI (`data:image/png;base64,…`) so the single `.html` file works with no
external files. But the model has no way to produce that: it authors HTML as a
fenced ` ```html ` code block that reaches the user as a **download button**
(`frontend/app.js:902+`), and it cannot emit a real base64 encoding of a binary
image.

The naive fix — a tool that returns the image's data URI as text — is a trap.
base64 inflates bytes ~33%, so a modest 150 KB PNG becomes ~200 KB of text
(≈50–70k tokens). The chat models here run at a **16–24k context**. A single
such tool result would overflow the context window before the model could
finish the page.

The real requirement: **get the data URI into the final HTML artifact without it
ever passing through the model's context.** We do this by substituting at the
*artifact boundary* (download / API response), after the model is done writing.

## Goals

- Let a model produce a genuinely self-contained HTML page that embeds
  uploaded/workspace images, with **zero base64 in the model's context**.
- Serve both the **web UI** (download button) and **headless agents** (Kernel,
  Argus, any direct API caller) from **one** server-side implementation.
- Reuse existing workspace-resolution and path-traversal guards; no new trust
  boundary.

## Non-goals (v1)

- Inlining JavaScript or CSS (`<script src>`, `<link rel=stylesheet>`). v1
  inlines **images only**. Easy to extend later.
- Inlining remote URLs (`https://…`). Only workspace-local `asset:` references
  are resolved; everything else is left untouched.
- Fetching or bundling files outside the session workspace. Out-of-workspace
  references are rejected, not resolved.
- A model-invoked *operation* (Y/N/E confirm card). This is a transform applied
  to an artifact the model already produced, not a discrete op the model calls.

## Design decisions (resolved during brainstorming)

1. **`asset:` sentinel convention.** The model writes ordinary HTML in its usual
   fenced ` ```html ` block, but references a workspace image as
   `src="asset:<filename>"` (also honoured in `href="…"` and CSS `url(asset:…)`).
   The model authors normally and reliably; only the sentinel is special.
2. **Substitution at the artifact boundary, server-side.** A single backend
   function performs the base64 inlining. It is exposed as an HTTP endpoint so
   the substitution logic lives **once** and is shared by every consumer. The
   model context never sees base64.
3. **Endpoint, not operation.** Operations return a *workspace file* and route
   through the confirm card — wrong ergonomics for "transform this text I'm about
   to download." A plain endpoint fits both the browser download path and
   headless callers.
4. **Fail loud, never silent.** A referenced file that is missing, too large, or
   outside the workspace is **left as the literal `asset:` token** and reported
   in a `missing[]` list, so the caller can surface it. We never silently drop a
   reference or emit a broken page without signalling.

## Architecture & data flow

```
model writes ```html … src="asset:photo.png" … ```      (no base64, in context)
        │
        ├── web UI: download handler (app.js) ──► POST /api/inline-assets
        │                                              {session_id, html}
        │                                                    │
        └── headless agent ─────────────────────► POST /api/inline-assets
                                                             │
                                          backend inline_assets(html, workspace)
                                          • scan asset:<name> in src/href/url()
                                          • resolve inside workspace (guarded)
                                          • base64 + data:<mime>;base64,…
                                          • size cap, mime-by-extension
                                                             │
                                        ◄── {html, replaced[], missing[]} ──
        │
   web UI: build Blob from returned html ──► self-contained .html download
   headless agent: write returned html to a file
```

## Components

### 1. `inline_assets(html: str, workspace: Path) -> InlineResult`
New pure function (proposed: `backend/inlining.py`, a small focused module).

- **Scan.** Regex-match `asset:<name>` appearing inside `src="…"` / `src='…'`,
  `href="…"`, and CSS `url(…)`. `<name>` is a bare workspace filename
  (no slashes beyond the resolved workspace; `../` is rejected at resolve time).
- **Resolve.** `path = (workspace / name).resolve()`, then `_assert_inside(path,
  workspace)` (the existing guard). Non-resolving or escaping paths → recorded in
  `missing`, token left intact.
- **Encode.** `data:<mime>;base64,<b64>` where `<mime>` comes from an
  extension→MIME map (png, jpg/jpeg, gif, webp, svg, bmp, ico, avif). Unknown
  extension → `application/octet-stream` (still valid, still self-contained).
- **Size cap.** Reject any single asset whose raw size exceeds a configured cap
  (default **5 MB**); recorded in `missing` with reason, token left intact.
- **Return** `InlineResult(html, replaced: list[str], missing: list[dict])`
  where each `missing` entry is `{name, reason}`.

The function does no I/O beyond reading the resolved files; it is deterministic
and unit-testable without the web layer.

### 2. `POST /api/inline-assets`
New FastAPI endpoint in `backend/app.py`.

- Body: `{session_id: str, html: str}`.
- Resolves the session workspace via the existing `_resolve_in_workspace`
  helper; `session_dir` is the `workspace` passed to `inline_assets`.
- Response: `{html, replaced, missing}` (200). Bad body → 400. The endpoint
  itself never 500s on a missing asset — that is a normal `missing[]` outcome.
- A request-level guard caps total input `html` length (e.g. 5 MB) to bound work.

### 3. Web UI wiring (`frontend/app.js`)
The code-block download path (`frontend/app.js:902+`) currently builds a Blob
straight from the code text. For blocks classified as `html` (and `svg`), it
will first `POST /api/inline-assets` with the block text + current `session_id`,
then build the Blob from the returned `html`. If any `missing[]` come back, show
a non-blocking warning listing the unresolved names. Non-html blocks are
unchanged. A fetch failure falls back to downloading the raw (un-inlined) text
with a warning — never blocks the download.

### 4. Model guidance
- Add one short paragraph to `_operations_prompt_block` (`backend/app.py:142`)
  teaching the convention: to embed an uploaded image in a self-contained HTML
  page, reference it as `src="asset:<filename>"`; it is inlined automatically
  when the page is downloaded.
- Extend the upload-injection note (`frontend/app.js:593`) so that when a file is
  an image, the model is reminded it can embed it via `asset:<name>`.

## Error handling

- **Missing / unreadable file:** token left intact, `{name, reason:"not found"}`
  in `missing`. Page still downloads; user sees which refs failed.
- **Path traversal (`asset:../../etc/passwd`):** rejected at `_assert_inside`;
  `reason:"outside workspace"`; token left intact. (Covered by a test.)
- **Oversize asset (> cap):** `reason:"exceeds size cap"`; token left intact.
- **Endpoint unreachable from UI:** raw text downloaded with a warning banner;
  the user still gets their (non-self-contained) file.
- Every failure is surfaced; none is swallowed.

## Security

- Only files that resolve **inside the caller's session workspace** are ever
  read; reuses `_assert_inside` / `_resolve_in_workspace` — the same seam used by
  operations and converters. No new attack surface beyond "read a file you
  already uploaded to your own session."
- Size caps on both per-asset bytes and total input HTML bound resource use.
- No shell, no subprocess; pure in-process read + base64.

## Testing

**pytest (`tests/test_inline_assets.py`):**
- Happy path: `asset:photo.png` in `src` → correct `data:image/png;base64,…`.
- Each MIME: png, jpg, gif, webp, svg → correct MIME label.
- `href` and CSS `url(asset:…)` forms both substituted.
- Missing file → token intact, listed in `missing`.
- Path traversal `asset:../../etc/passwd` → rejected, token intact, listed.
- Oversize asset → rejected, token intact, listed.
- Unknown extension → `application/octet-stream`.
- Endpoint: 200 shape, 400 on bad body, missing asset does not 500.

**Playwright smoke (manual/gated, matches existing practice):**
- Upload an image → prompt the model for a self-contained page using
  `asset:<name>` → click download → assert the saved HTML contains
  `data:image/…;base64,` and no literal `asset:` token.

## Open questions / future work

- Extend to `<script src="asset:…">` and `<link>` for fully-bundled pages.
- Optional: honour bare `src="photo.png"` (no sentinel) when the name matches a
  workspace file — more magical, but risks inlining things the author meant to
  stay external. Deferred; the explicit `asset:` sentinel is the v1 contract.
