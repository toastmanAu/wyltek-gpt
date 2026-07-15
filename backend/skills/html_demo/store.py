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
