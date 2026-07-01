# Drop Folder — design spec

**Date:** 2026-07-02
**Status:** Approved, ready for implementation plan

## Problem

wyltek-gpt can only bring files into a chat via the browser upload button
(`POST /api/upload`), which stores bytes per-session in
`workspaces/<session_id>/`. There is no way to reference files that were placed
on the host outside the browser (scp, file manager, a synced/network folder,
another machine). This spec adds a **drop folder**: a host directory the user
fills externally, browsable from the UI, whose files can be copied into a chat.

## Goals

- Let the user populate a folder on the host outside the app, then pull files
  from it into a conversation.
- Reuse the existing upload contract so ops, converters, and chat file
  resolution need **no changes** downstream of import.
- Support selecting and importing **multiple** files at once; they are then
  referenced by name in chat to disambiguate.

## Non-goals (v1)

- Nested subfolder browsing. Listing is **flat** — only files directly in the
  drop folder. (Easy to extend later.)
- The app writing to or deleting from the drop folder. It is **read-only** from
  the app's side (list + read only).
- Auto-exposing every drop-folder file to the model. Import is explicit,
  user-driven (browse panel), not automatic.

## Design decisions (resolved during brainstorming)

1. **Discovery = browse panel.** The UI lists what's in the drop folder; the
   user clicks to attach. Mirrors the upload flow; most discoverable.
2. **Import = copy into the session workspace.** On attach, each file is copied
   from the drop folder into `workspaces/<session_id>/`. After that it behaves
   exactly like an uploaded file. The drop folder stays a pristine, append-only
   source. This is the single trust/ownership boundary crossing.
3. **Multi-select.** The panel has checkboxes; one "Add to chat" action imports
   all checked files. Files are disambiguated by name in chat.
4. **Flat listing.** Subdirectories and dotfiles are ignored in v1.
5. **Location.** Default `dropbox/` at project root, configurable via the
   existing `storage:` block in `config.yaml` (supports `~` and absolute paths),
   so it can point at a synced/network folder.

## Architecture & data flow

```
host drop folder ──list──▶ panel (checkboxes) ──Add──▶ POST /api/dropbox/import
                                                             │  copy each → workspaces/<session>/
                                                             ▼
                                    same as upload: history system-msg + download row per file
```

The drop folder is read-only from the app's side — the app lists and reads,
never writes or deletes there. Import converges on the existing upload contract
as fast as possible: after the copy, every existing code path (ops, converters,
chat resolution, the model's "files in my workspace" model) works unchanged.

## Backend (`backend/app.py`)

### Config resolution

- Add `storage.dropbox` (default `dropbox`) to `config.yaml`; resolve like
  `workspace` — support `~` expansion and absolute paths. Define a module-level
  `DROPBOX = ...` alongside `WORKSPACE`.
- Create the folder on startup if absent, so it is discoverable.

### Security helper

A helper mirroring `_resolve_in_workspace`, scoped to the drop root:

- Reduce any incoming name to `Path(name).name` (strip path components).
- Resolve against `DROPBOX`; reject if the resolved parent is not the drop root.
- Only regular files qualify (no directories, no symlinked escapes, no dotfiles).

### Endpoints

- `GET /api/dropbox` → `{ "files": [{ "name", "size", "modified" }, ...] }`
  - Files directly in the drop folder only. Skip subdirectories and dotfiles.
  - Sorted by mtime descending (newest first).
  - Missing or empty folder → `{ "files": [] }`. Never errors.

- `POST /api/dropbox/import` → body `{ "session_id", "names": [...] }`
  - For each name: validate via the security helper, confirm it is a regular
    file directly in the drop folder, copy into the session workspace using
    `_resolve_in_workspace` for the destination.
  - Use `shutil.copy` (streams; no whole-file read into memory).
  - Name collision in the session workspace → **overwrite**, consistent with
    the current `/api/upload` (`target.write_bytes`).
  - Returns `{ "imported": [{name,size}], "skipped": [{name,reason}] }` so
    partial failures are explicit, never silent.

## Frontend (`frontend/app.js`, `frontend/index.html`, `frontend/style.css`)

- A **📂 Drop folder** button next to the upload control opens a lightweight
  modal/panel.
- On open, fetch `/api/dropbox` and render a checkbox row per file:
  name · size · relative mtime.
- **Add to chat** posts the checked names to `/api/dropbox/import`. For **each**
  imported file, run the same three steps the upload handler uses today:
  1. `appendDownload(...)` — a download row.
  2. `history.push({ role: "system", content: ... })` naming the file so the
     model knows it exists and can reference it in operation calls.
  3. Tray park (`showTray`) — **only when exactly one** file is imported,
     matching current single-file UX. For multi-file imports, register all in
     history + download rows, no tray.
- A **Refresh** action re-fetches `/api/dropbox` (the user may have dropped more
  files while the panel is open).
- Empty state: "Drop folder is empty — add files to `<path>` and refresh."

## Edge cases

- **Traversal / symlinks:** names reduced to basename; resolved parent must
  equal the drop root; only regular files served.
- **Big files:** streamed via `shutil.copy`.
- **Collision:** overwrite (matches upload semantics).
- **Missing folder at list time:** return empty list, not an error.
- **Import of a name that vanished / is a dir / is a dotfile:** appears in
  `skipped` with a reason, not a 500.

## Testing (pytest, mirrors `tests/`)

- **List:**
  - Populated folder returns names sorted by mtime descending.
  - Subdirectories and dotfiles excluded.
  - Missing folder → `{ "files": [] }`.
- **Import:**
  - Copies into the correct session workspace.
  - Multi-name import copies all named files.
  - Traversal name (`../../etc/passwd`) rejected (lands in `skipped`, nothing
    written outside the workspace).
  - Nonexistent name → `skipped`, not a 500.
  - Collision with an existing workspace file overwrites.

## Files touched

- `backend/app.py` — config resolution, security helper, two endpoints.
- `config.yaml` — `storage.dropbox` default.
- `frontend/index.html` — button + panel markup.
- `frontend/app.js` — panel open/list/import/refresh wiring, reusing the upload
  registration path.
- `frontend/style.css` — panel styling.
- `tests/test_dropbox.py` — list + import coverage.
