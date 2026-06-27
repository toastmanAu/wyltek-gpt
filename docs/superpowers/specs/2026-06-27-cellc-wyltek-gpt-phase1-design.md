# cellc ↔ wyltek-gpt — Phase 1 (human-driven endpoints)

**Date:** 2026-06-27
**Status:** Approved design, pre-implementation
**Repo:** `~/local-chatbot` (wyltek-gpt), runs on driveThree
**Depends on:** `~/cellc-mcp` (the MCP server; its `cellc_mcp.server` tool functions are imported in-process)

## Purpose

Give wyltek-gpt a way to compile/inspect CellScript `.cell` source through the
already-built `cellc` tooling, surfaced as dedicated read-only HTTP endpoints
plus a minimal frontend affordance the **user** triggers. Phase 1 is the safe,
immediately-useful slice: a human writes/pastes a contract (or has the model
emit one) and clicks "check" to get structured diagnostics back.

Phase 2 (separate spec) adds the agentic tool-result loop so the model
self-iterates write → check → fix. Phase 1 deliberately does NOT build that loop.

## Why dedicated endpoints (not the OPERATIONS registry)

Investigation of `backend/app.py` + `frontend/app.js` established:

- The `OPERATIONS` registry + `tool_schemas()` + `__tool_calls__` path is
  **single-shot file production**: every op kind (`local`/`bridge`/`converter`)
  runs via `/api/operations/run`, returns a **workspace file** (`{url, name,
  size}`), and is gated by the Y/N/E confirm card.
- There is **no tool-result feedback loop**: an operation's output becomes a
  file shown to the user, never a result fed back into the model's context.
- `translate` (the other text-returning model-op) is a **dedicated endpoint**
  (`/api/operations/translate`) *outside* the registry. That is the precedent
  cellc follows: cellc returns DATA, not a file, so it does not fit
  `/api/operations/run`.

## Non-Goals (Phase 1)

- No agentic tool-result loop / model self-iteration (that is Phase 2).
- No registration of cellc in the `OPERATIONS` registry or `tool_schemas()`
  (no model-emitted `tool_calls` path; no Y/N/E confirm card involvement).
- No workspace file output. cellc endpoints are pure read-only query/response.
- No generalization of the existing open_palette bridge dispatch (the earlier
  bridge-op design is abandoned for cellc; open_palette code is untouched).

## Architecture

```
frontend (app.js)  ──HTTP──▶  /api/cellc/*  (app.py, dedicated endpoints)
                                   │
                                   ▼
                        backend/bridges/cellc.py   (in-process)
                                   │  import cellc_mcp.server
                                   ▼
                        cellc_mcp.server.cellc_*   →  cellc binary
```

### Component 1 — `backend/bridges/cellc.py` (new, in-process bridge)

Imports `cellc_mcp.server` and exposes thin, JSON-able wrappers. The heavy
lifting (package synthesis, tiering, module-scan isolation, never-raises
contract) already lives in `cellc_mcp` and must not be duplicated.

```python
def available() -> bool
    # True iff cellc_mcp imports AND the cellc binary resolves
    # (CELLC_BIN or PATH). Used to disable the feature cleanly when the
    # binary isn't built — same spirit as a missing local-op binary.

def status() -> dict          # {"available": bool, "binary": str | None}
def check(source: str, target_profile: str = "ckb", full: bool = False) -> dict
def explain(code: str) -> dict
def metadata(source: str, target_profile: str = "ckb", full: bool = False) -> dict
def language_reference() -> str
def list_examples() -> list[dict]
def get_example(name: str) -> dict
```

Each wrapper calls the corresponding `cellc_mcp.server.cellc_*` function and
returns its (already tiered, already error-safe) result. If `cellc_mcp` cannot
be imported, the module still loads and `available()` returns False (import is
attempted lazily / guarded) so wyltek-gpt boots even without cellc installed.

### Component 2 — dedicated endpoints in `backend/app.py`

All read-only, no confirm card, no workspace files. Validate every input at the
boundary (mirror `translate`'s discipline), then call the bridge:

| Method + path | Body / params | Returns |
| --- | --- | --- |
| `GET /api/cellc/status` | — | `{"available": bool, "binary": str\|null}` |
| `POST /api/cellc/check` | `{source, target_profile?, full?}` | tiered check dict |
| `POST /api/cellc/explain` | `{code}` | explain dict |
| `POST /api/cellc/metadata` | `{source, target_profile?, full?}` | metadata summary dict |
| `GET /api/cellc/reference` | — | `{"reference": str}` |
| `GET /api/cellc/examples` | — | `{"examples": [...]}` |
| `GET /api/cellc/example/{name}` | path `name` | `{"name", "source"}` or 404 |

Validation rules (reject with HTTP 400 before calling the bridge):
- `source`: required non-empty string; reject if larger than `_CELLC_MAX_SOURCE`
  (default 200_000 chars).
- `target_profile`: optional; if present must be in `{"ckb"}`.
- `full`: optional bool.
- `code`: required; must match `^[A-Za-z0-9_]+$` (E-codes/names; the bridge/cellc
  re-validates).
- `name`: must match `^[A-Za-z0-9_-]+$` (path-traversal guard; cellc_mcp also
  guards, this is defence-in-depth at the boundary).

If `bridges.cellc.available()` is False, the POST/inspect endpoints return
HTTP 503 with a clear "cellc not available — build it and set CELLC_BIN" message.

### Component 3 — frontend affordance (`frontend/app.js`, minimal)

- On load, call `GET /api/cellc/status`; only enable the affordance when
  `available` is true.
- A `/check` slash command (and/or a small "check" button on rendered `.cell` /
  `cellscript` code fences) that POSTs the fence's source to `/api/cellc/check`
  and renders the result inline: `✓ ok` or a list of `Ln:Cn [code] message`,
  capped to the returned diagnostics (already ≤5). Reuse the existing op-result
  rendering styles where practical; keep the JS addition small and self-contained.
- This is human-triggered. No model `tool_calls`, no confirm card.

### Component 4 — wiring / environment (driveThree)

- Install the server into wyltek-gpt's venv: `~/local-chatbot/.venv/bin/pip
  install -e ~/cellc-mcp`.
- The uvicorn process needs:
  `CELLC_BIN=~/CellScript/target/release/cellc` and
  `CELLSCRIPT_REPO=~/CellScript`.
  Provide via a documented `.env`/run-script note (README section) and a
  `config.yaml` `cellc:` block is optional (binary path + default profile) but
  env is authoritative.
- `bridges/cellc.py.available()` reads the same env, so a missing/unbuilt binary
  degrades to a disabled feature rather than a 500.

## Testing

wyltek-gpt currently has NO test setup. Phase 1 introduces a minimal one:

- Add `pytest` + `httpx` (FastAPI `TestClient` needs httpx, already a dep) to a
  `requirements-dev.txt`.
- `tests/test_cellc_bridge.py` — unit tests for `backend/bridges/cellc.py`
  with `cellc_mcp.server` mocked: each wrapper forwards correctly and returns
  the mocked dict; `available()` False when import/binary missing.
- `tests/test_cellc_endpoints.py` — FastAPI `TestClient` tests with the bridge
  mocked: validation (empty source → 400, bad profile → 400, bad code → 400,
  traversal name → 400/404, oversized source → 400), happy-path shapes,
  503 when `available()` is False.
- `tests/test_cellc_integration.py` (`@pytest.mark.needs_cellc`, skips if no
  binary) — real `/api/cellc/check` with a good contract (`ok: true`) and a bad
  one (`ok: false`, diagnostics present); `/api/cellc/explain` `E0014` →
  `ecode == "E0014"`; `/api/cellc/status` → `available: true`.

Run: `~/local-chatbot/.venv/bin/python -m pytest -v`
Integration: `CELLC_BIN=~/CellScript/target/release/cellc
CELLSCRIPT_REPO=~/CellScript .venv/bin/python -m pytest -m needs_cellc -v`.

## Git / safety

- `~/local-chatbot` is on `main` with pre-existing uncommitted WIP
  (`backend/app.py`, `frontend/app.js`, `data/capabilities.json`). Implementation
  happens on a feature branch `feat/cellc-phase1`; commits stage ONLY the
  Phase-1 files. Never `git add -A` / never sweep the user's WIP into a commit.

## Phase 2 carry-forward (not built here)

The agentic tool-result loop: when a capable model emits a read-only cellc
`tool_call`, the backend auto-runs it (read-only → no confirm card, per the
earlier decision), appends a `role: "tool"` result message, and re-invokes
Ollama to continue — looping until the model stops calling tools. This reuses
`backend/bridges/cellc.py` unchanged; it adds the loop to the `/api/chat`
streaming path.
