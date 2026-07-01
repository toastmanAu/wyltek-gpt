# Drop Folder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a host "drop folder" the user fills outside the browser, browsable from the UI, whose files can be multi-selected and copied into the current chat's session workspace.

**Architecture:** A single global host folder (default `dropbox/` at project root, configurable via `config.yaml`). Two read-mostly FastAPI endpoints — list and import — plus a browse panel in the frontend. Import *copies* each selected file into `workspaces/<session_id>/`, after which it is identical to an uploaded file, so ops/converters/chat resolution need no changes.

**Tech Stack:** Python 3 / FastAPI (backend), pytest + `fastapi.testclient.TestClient` (tests), vanilla JS + HTML + CSS (frontend, no build step, no JS test harness).

## Global Constraints

- Drop folder is **read-only from the app's side**: list and read only, never write or delete there.
- Listing is **flat**: only regular files directly in the drop folder; skip subdirectories and dotfiles.
- File names crossing into the workspace are reduced to `Path(name).name`; the resolved source path's parent must equal the resolved drop root.
- Name collision in the session workspace **overwrites** (matches existing `/api/upload` semantics).
- Missing/empty drop folder at list time returns `{"files": []}` — never an error.
- Import returns explicit `{"imported": [...], "skipped": [...]}` — partial failures are never silent.
- Follow existing code patterns: config resolution mirrors `WORKSPACE`/`OUTPUT_DIR` (`backend/app.py:41-55`); the security helper mirrors `_resolve_in_workspace` (`backend/app.py:1051-1061`); tests mirror `tests/test_chat_text_files.py` (`TestClient` + `monkeypatch.setattr(app_module, "WORKSPACE"/"DROPBOX", tmp_path)`).

---

### Task 1: Backend — config resolution, security helper, list endpoint

**Files:**
- Modify: `config.yaml:43-45` (add `dropbox` under `storage:`)
- Modify: `backend/app.py:41-44` (add `DROPBOX` resolution after `WORKSPACE`)
- Modify: `backend/app.py` (add `_resolve_in_dropbox` helper + `GET /api/dropbox`, in the "Workspace helpers + uploads + downloads" section near line 1048)
- Test: `tests/test_dropbox.py` (new)

**Interfaces:**
- Consumes: `ROOT`, `STORAGE`, `log` (`backend/app.py:37,42`, module logger); `os`, `Path` (already imported).
- Produces:
  - Module global `DROPBOX: Path` (resolved absolute drop root).
  - `_resolve_in_dropbox(name: str) -> Path` — returns resolved path of a basename inside `DROPBOX`; raises `HTTPException(400)` if the resolved parent is not `DROPBOX`.
  - `GET /api/dropbox` → `{"files": [{"name": str, "size": int, "modified": float}, ...]}` sorted by `modified` descending.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_dropbox.py`:

```python
"""Drop-folder list + import tests.

The drop folder is a host directory the user fills outside the browser. These
tests pin: (1) flat listing that excludes subdirs/dotfiles and sorts newest
first; (2) copy-into-workspace import with explicit skipped reasons and no
path-traversal escape.
"""

import os

from fastapi.testclient import TestClient

from backend import app as app_module

client = TestClient(app_module.app)


def _point_dropbox(tmp_path, monkeypatch):
    drop = tmp_path / "drop"
    drop.mkdir()
    monkeypatch.setattr(app_module, "DROPBOX", drop.resolve())
    return drop


# ── GET /api/dropbox ─────────────────────────────────────────────────


def test_list_returns_files_newest_first(tmp_path, monkeypatch):
    drop = _point_dropbox(tmp_path, monkeypatch)
    old = drop / "old.txt"
    old.write_text("a")
    new = drop / "new.txt"
    new.write_text("bb")
    os.utime(old, (1000, 1000))
    os.utime(new, (2000, 2000))

    r = client.get("/api/dropbox")
    assert r.status_code == 200
    files = r.json()["files"]
    assert [f["name"] for f in files] == ["new.txt", "old.txt"]
    assert files[1]["size"] == 1


def test_list_excludes_subdirs_and_dotfiles(tmp_path, monkeypatch):
    drop = _point_dropbox(tmp_path, monkeypatch)
    (drop / "keep.txt").write_text("x")
    (drop / ".hidden").write_text("x")
    (drop / "sub").mkdir()

    names = [f["name"] for f in client.get("/api/dropbox").json()["files"]]
    assert names == ["keep.txt"]


def test_list_missing_folder_is_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "DROPBOX", (tmp_path / "nope").resolve())
    r = client.get("/api/dropbox")
    assert r.status_code == 200
    assert r.json() == {"files": []}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /home/phill/local-chatbot && .venv/bin/pytest tests/test_dropbox.py -v`
Expected: FAIL — `AttributeError` on `app_module.DROPBOX` (attribute does not exist yet) / 404 on the route.

- [ ] **Step 3: Add the config default**

In `config.yaml`, extend the `storage:` block (currently lines 43-45) so it reads:

```yaml
storage:
  workspace: workspaces
  output_dir: ~/Downloads/wyltek-gpt-output
  dropbox: dropbox
```

- [ ] **Step 4: Resolve `DROPBOX` at module load**

In `backend/app.py`, immediately after line 44 (`WORKSPACE.mkdir(exist_ok=True)`) and before `_OUT_RAW = ...`, insert:

```python
_DROP_RAW = STORAGE.get("dropbox", "dropbox")
DROPBOX = Path(os.path.expanduser(_DROP_RAW))
if not DROPBOX.is_absolute():
    DROPBOX = ROOT / DROPBOX
DROPBOX = DROPBOX.resolve()
try:
    DROPBOX.mkdir(parents=True, exist_ok=True)
    log.info("drop folder: %s", DROPBOX)
except OSError as exc:
    log.warning("drop folder unavailable (%s): %s", DROPBOX, exc)
```

- [ ] **Step 5: Add the security helper + list endpoint**

In `backend/app.py`, in the "Workspace helpers + uploads + downloads" section, directly after `_resolve_in_workspace` (ends at line 1061), insert:

```python
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
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cd /home/phill/local-chatbot && .venv/bin/pytest tests/test_dropbox.py -v`
Expected: PASS (3 passed).

- [ ] **Step 7: Commit**

```bash
git add config.yaml backend/app.py tests/test_dropbox.py
git commit -m "feat: drop-folder config + list endpoint"
```

---

### Task 2: Backend — import endpoint (copy into workspace)

**Files:**
- Modify: `backend/app.py:7` area (add `import shutil`)
- Modify: `backend/app.py` (add `POST /api/dropbox/import` after `dropbox_list`)
- Test: `tests/test_dropbox.py` (extend)

**Interfaces:**
- Consumes: `_resolve_in_dropbox(name)` and `DROPBOX` from Task 1; `_resolve_in_workspace(session_id, name) -> tuple[Path, Path]` (`backend/app.py:1051`); `HTTPException`.
- Produces: `POST /api/dropbox/import`, body `{"session_id": str, "names": list[str]}` → `{"imported": [{"name": str, "size": int}], "skipped": [{"name": str, "reason": str}]}`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dropbox.py`:

```python
# ── POST /api/dropbox/import ─────────────────────────────────────────


def _point_both(tmp_path, monkeypatch):
    drop = tmp_path / "drop"
    drop.mkdir()
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.setattr(app_module, "DROPBOX", drop.resolve())
    monkeypatch.setattr(app_module, "WORKSPACE", ws.resolve())
    return drop, ws


def test_import_copies_multiple_into_session_workspace(tmp_path, monkeypatch):
    drop, ws = _point_both(tmp_path, monkeypatch)
    (drop / "a.txt").write_text("AAA")
    (drop / "b.txt").write_text("BB")

    r = client.post(
        "/api/dropbox/import",
        json={"session_id": "s1", "names": ["a.txt", "b.txt"]},
    )
    assert r.status_code == 200
    body = r.json()
    assert {f["name"] for f in body["imported"]} == {"a.txt", "b.txt"}
    assert body["skipped"] == []
    assert (ws / "s1" / "a.txt").read_text() == "AAA"
    assert (ws / "s1" / "b.txt").read_text() == "BB"


def test_import_traversal_name_is_skipped_not_written(tmp_path, monkeypatch):
    drop, ws = _point_both(tmp_path, monkeypatch)
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP")

    r = client.post(
        "/api/dropbox/import",
        json={"session_id": "s1", "names": ["../secret.txt"]},
    )
    assert r.status_code == 200
    # basename "secret.txt" does not exist in the drop folder -> skipped
    assert r.json()["imported"] == []
    assert r.json()["skipped"][0]["name"] == "../secret.txt"
    assert not (ws / "s1" / "secret.txt").exists()


def test_import_nonexistent_name_is_skipped_not_500(tmp_path, monkeypatch):
    _point_both(tmp_path, monkeypatch)
    r = client.post(
        "/api/dropbox/import",
        json={"session_id": "s1", "names": ["ghost.txt"]},
    )
    assert r.status_code == 200
    assert r.json()["imported"] == []
    assert r.json()["skipped"] == [{"name": "ghost.txt", "reason": "not found"}]


def test_import_collision_overwrites(tmp_path, monkeypatch):
    drop, ws = _point_both(tmp_path, monkeypatch)
    (drop / "dup.txt").write_text("NEW")
    sess = ws / "s1"
    sess.mkdir()
    (sess / "dup.txt").write_text("OLD")

    r = client.post(
        "/api/dropbox/import",
        json={"session_id": "s1", "names": ["dup.txt"]},
    )
    assert r.status_code == 200
    assert (sess / "dup.txt").read_text() == "NEW"
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `cd /home/phill/local-chatbot && .venv/bin/pytest tests/test_dropbox.py -k import -v`
Expected: FAIL — 404 (route `/api/dropbox/import` not defined).

- [ ] **Step 3: Add the `shutil` import**

In `backend/app.py`, add `import shutil` to the stdlib import block (alongside `import os` at line 7). Keep imports alphabetically grouped if the file already is; otherwise place it next to `import os`.

- [ ] **Step 4: Add the import endpoint**

In `backend/app.py`, directly after the `dropbox_list` function from Task 1, insert:

```python
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
        shutil.copy(src, dest)
        imported.append({"name": dest.name, "size": dest.stat().st_size})

    return {"imported": imported, "skipped": skipped}
```

- [ ] **Step 5: Run the full drop-folder suite to verify it passes**

Run: `cd /home/phill/local-chatbot && .venv/bin/pytest tests/test_dropbox.py -v`
Expected: PASS (7 passed).

- [ ] **Step 6: Commit**

```bash
git add backend/app.py tests/test_dropbox.py
git commit -m "feat: drop-folder import endpoint (copy into session workspace)"
```

---

### Task 3: Frontend — browse panel (button, markup, wiring, styling)

**Files:**
- Modify: `frontend/index.html:29-30` (add drop-folder button) and after `frontend/index.html:41` (add panel + backdrop markup)
- Modify: `frontend/app.js` (add panel wiring near the upload handler around line 602)
- Modify: `frontend/style.css` (add panel styles)

**Interfaces:**
- Consumes (existing, `frontend/app.js`): `SESSION` (line 22), `history` (line 85), `appendDownload(role, prefix, name, href)` (line 426), `showTray(file)` (line 498), `formatBytes(n)` (line 560), the `$` selector helper (line 5), and endpoints `GET /api/dropbox` / `POST /api/dropbox/import` from Tasks 1-2.
- Produces: user-facing panel; no exported JS API. Verified by manual smoke test (repo has no JS test harness).

- [ ] **Step 1: Add the button and panel markup**

In `frontend/index.html`, change the upload bar (lines 29-30) to add a drop-folder button after the file input:

```html
      <label class="upload-btn" for="file" title="Upload file">+ file</label>
      <input type="file" id="file" hidden />
      <button id="drop-btn" class="upload-btn" type="button" title="Drop folder">📂 drop</button>
```

Then, directly after the `tray` section (after line 41, `</section>`), add the panel and its backdrop:

```html
    <div id="drop-backdrop" class="settings-backdrop hidden"></div>
    <aside id="drop-panel" class="settings hidden" aria-hidden="true" aria-label="Drop folder">
      <header class="settings-head">
        <span>drop folder</span>
        <div class="drop-head-actions">
          <button id="drop-refresh" type="button" class="ghost" title="Refresh">⟳</button>
          <button id="drop-close" type="button" class="ghost" aria-label="Close">×</button>
        </div>
      </header>
      <div class="settings-body">
        <ul id="drop-list" class="drop-list"></ul>
      </div>
      <footer class="drop-foot">
        <button id="drop-add" type="button" class="settings-btn" disabled>Add to chat</button>
      </footer>
    </aside>
```

- [ ] **Step 2: Add the panel styling**

Append to `frontend/style.css`:

```css
.drop-head-actions { display: flex; gap: 0.25rem; }
.drop-list { list-style: none; margin: 0; padding: 0; }
.drop-list li {
  display: flex; align-items: center; gap: 0.5rem;
  padding: 0.4rem 0.2rem; border-bottom: 1px solid var(--border, #333);
}
.drop-list li label { display: flex; align-items: center; gap: 0.5rem; flex: 1; cursor: pointer; }
.drop-name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.drop-meta { opacity: 0.6; font-size: 0.85em; white-space: nowrap; }
.drop-empty { opacity: 0.6; padding: 1rem 0.2rem; }
.drop-foot { padding: 0.6rem; border-top: 1px solid var(--border, #333); text-align: right; }
```

- [ ] **Step 3: Add the panel wiring**

In `frontend/app.js`, after the upload `fileInput` change handler (ends at line 602) and before `trayClear.addEventListener(...)`, add:

```javascript
// ─── Drop folder panel ────────────────────────────────────────────
const dropBtn = $("#drop-btn");
const dropPanel = $("#drop-panel");
const dropBackdrop = $("#drop-backdrop");
const dropList = $("#drop-list");
const dropAdd = $("#drop-add");

function dropRelTime(epochSeconds) {
  const secs = Math.max(0, Date.now() / 1000 - epochSeconds);
  const units = [[86400, "d"], [3600, "h"], [60, "m"]];
  for (const [size, label] of units) {
    if (secs >= size) return `${Math.floor(secs / size)}${label} ago`;
  }
  return "just now";
}

function dropCheckedNames() {
  return [...dropList.querySelectorAll("input[type=checkbox]:checked")].map(
    (c) => c.value,
  );
}

function dropSyncAddState() {
  dropAdd.disabled = dropCheckedNames().length === 0;
}

async function dropRefresh() {
  dropList.innerHTML = "<li class='drop-empty'>loading...</li>";
  try {
    const r = await fetch("/api/dropbox");
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const files = (await r.json()).files || [];
    if (files.length === 0) {
      dropList.innerHTML =
        "<li class='drop-empty'>Drop folder is empty — add files to it and refresh.</li>";
      dropSyncAddState();
      return;
    }
    dropList.innerHTML = "";
    for (const f of files) {
      const li = document.createElement("li");
      const label = document.createElement("label");
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.value = f.name;
      cb.addEventListener("change", dropSyncAddState);
      const name = document.createElement("span");
      name.className = "drop-name";
      name.textContent = f.name;
      const meta = document.createElement("span");
      meta.className = "drop-meta";
      meta.textContent = `${formatBytes(f.size)} · ${dropRelTime(f.modified)}`;
      label.append(cb, name, meta);
      li.append(label);
      dropList.append(li);
    }
  } catch (e) {
    dropList.innerHTML = `<li class='drop-empty'>failed to load: ${e.message}</li>`;
  }
  dropSyncAddState();
}

function openDropPanel() {
  dropPanel.classList.remove("hidden");
  dropBackdrop.classList.remove("hidden");
  dropPanel.setAttribute("aria-hidden", "false");
  dropRefresh();
}

function closeDropPanel() {
  dropPanel.classList.add("hidden");
  dropBackdrop.classList.add("hidden");
  dropPanel.setAttribute("aria-hidden", "true");
}

// Reuse the upload registration path so the model learns each imported file.
function registerImportedFile(meta) {
  appendDownload(
    "system",
    `added from drop folder (${formatBytes(meta.size)}):`,
    meta.name,
    `/api/files/${SESSION}/${encodeURIComponent(meta.name)}`,
  );
  history.push({
    role: "system",
    content: `User added file "${meta.name}" (${formatBytes(meta.size)}) from the drop folder — available in the workspace as "${meta.name}". When the user asks you to process, convert, edit, or modify this file, use one of the available operations with "${meta.name}" as the source.`,
  });
}

dropBtn.addEventListener("click", openDropPanel);
$("#drop-close").addEventListener("click", closeDropPanel);
dropBackdrop.addEventListener("click", closeDropPanel);
$("#drop-refresh").addEventListener("click", dropRefresh);

dropAdd.addEventListener("click", async () => {
  const names = dropCheckedNames();
  if (names.length === 0) return;
  dropAdd.disabled = true;
  try {
    const r = await fetch("/api/dropbox/import", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: SESSION, names }),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const body = await r.json();
    for (const meta of body.imported) registerImportedFile(meta);
    if (body.imported.length === 1) {
      showTray({ ...body.imported[0], session_id: SESSION });
    }
    for (const s of body.skipped) {
      appendMsg("system", `could not add "${s.name}": ${s.reason}`);
    }
    closeDropPanel();
  } catch (e) {
    appendMsg("system", `drop import failed: ${e.message}`);
    dropAdd.disabled = false;
  }
});
```

- [ ] **Step 4: Manual smoke test**

Run the app and exercise the panel end to end:

```bash
cd /home/phill/local-chatbot && mkdir -p dropbox && printf 'hello from dropbox' > dropbox/note.txt
# start the server the usual way, e.g.:
.venv/bin/uvicorn backend.app:app --reload
```

In the browser:
1. Click **📂 drop** — panel opens and lists `note.txt` with size + relative time.
2. Tick `note.txt` — **Add to chat** enables.
3. Click **Add to chat** — panel closes; a "added from drop folder" download row appears; the tray shows `note.txt`.
4. Confirm the copy landed: `ls workspaces/default/note.txt` exists with the same bytes.
5. Ask the model "summarise note.txt" — it references the file by name (proves the `history` system message registered).

Expected: all five succeed. If the panel button does nothing, check the browser console for a selector typo; if the list 500s, re-check `DROPBOX` resolves (server log line "drop folder: ...").

- [ ] **Step 5: Commit**

```bash
git add frontend/index.html frontend/app.js frontend/style.css
git commit -m "feat: drop-folder browse panel (multi-select import into chat)"
```

---

## Post-implementation

- [ ] Run the whole suite to confirm nothing regressed: `cd /home/phill/local-chatbot && .venv/bin/pytest -q`
- [ ] Add `dropbox/` to `.gitignore` if the default location is used and you don't want dropped files tracked (check the existing `.gitignore` first — `workspaces/` handling is the precedent to mirror).

## Self-review notes

- **Spec coverage:** list endpoint (Task 1), import + copy + collision + traversal + skipped reasons (Task 2), browse panel + multi-select + refresh + empty state + tray-on-single + reuse of upload registration (Task 3), config default + startup mkdir (Task 1), security helper (Task 1). All spec sections map to a task.
- **Naming consistency:** `DROPBOX`, `_resolve_in_dropbox`, `/api/dropbox`, `/api/dropbox/import`, `dropbox_list`, `dropbox_import`, `registerImportedFile`, `dropRefresh` used identically across tasks.
- **`.gitignore`:** flagged as a post-step judgement call rather than a hard task, since the drop folder may be an external absolute path (nothing to ignore) — mirror the repo's existing `workspaces/` decision.
