"""Capability cache — persisted record of what each Ollama model can do.

File format: JSON map of ``model_name -> capability_record``. Each
capability record carries the four axes plus a probe timestamp and
counter, so callers can decide whether to re-probe stale entries.

The cache is the single source of truth for the UI dropdown glyphs and
for the warn-banner gate. It's persisted under ``data/capabilities.json``
(in the project root, sister to the SQLite gitignored data files).

Concurrency: an in-process asyncio Lock guards the dict + file, so two
simultaneous lazy-probe requests for the same fresh-install backend
don't double-write the cache. Cross-process writes aren't a concern —
this app runs as a single uvicorn worker.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from backend.probe import probe_all

log = logging.getLogger(__name__)


class CapabilityCache:
    def __init__(self, store_path: Path, ollama_url: str):
        self.store_path = store_path
        self.ollama_url = ollama_url
        self._cache: dict[str, dict] = {}
        self._lock = asyncio.Lock()
        self._inflight: dict[str, asyncio.Task] = {}
        self._load()

    def _load(self) -> None:
        if self.store_path.exists():
            try:
                self._cache = json.loads(self.store_path.read_text())
                log.info("capabilities cache loaded: %d entries", len(self._cache))
            except json.JSONDecodeError as e:
                log.warning("capabilities cache corrupt (%s) — starting empty", e)
                self._cache = {}

    def _save(self) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.store_path.write_text(json.dumps(self._cache, indent=2, sort_keys=True))

    def get(self, model: str) -> dict | None:
        return self._cache.get(model)

    def all(self) -> dict[str, dict]:
        # Defensive copy so callers can't accidentally mutate cache state.
        return {k: dict(v) for k, v in self._cache.items()}

    async def probe_now(self, model: str) -> dict:
        """Run probe and persist. If a probe for this model is already in
        flight, await the existing task instead of starting a duplicate.

        Now backed by Ollama's metadata API (`/api/show`) — fast (<100ms),
        accurate. Was previously inference-based (30-180s, unreliable)."""
        if model in self._inflight:
            return await self._inflight[model]

        async def _run():
            try:
                caps = await probe_all(self.ollama_url, model)
            except Exception as exc:
                log.warning("probe failed for %s: %s", model, exc)
                caps = {
                    "tool_calling": "error", "vision": False,
                    "audio": False, "reasoning": False, "probe_error": str(exc),
                }
            async with self._lock:
                existing = self._cache.get(model, {})
                caps["last_probed"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                caps["probe_count"] = int(existing.get("probe_count", 0)) + 1
                self._cache[model] = caps
                self._save()
            return caps

        task = asyncio.create_task(_run())
        self._inflight[model] = task
        try:
            return await task
        finally:
            self._inflight.pop(model, None)

    async def probe_models(self, models: list[str]) -> dict[str, dict]:
        """Probe a batch of models in sequence. Cheap with metadata-based
        probes — full sweep of 26 models takes a couple of seconds."""
        out: dict[str, dict] = {}
        for m in models:
            out[m] = await self.probe_now(m)
        return out
