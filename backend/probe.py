"""Capability detection for Ollama-served models.

Queries Ollama's ``POST /api/show`` for the model's declared capabilities
list — Ollama's own metadata, populated when the model was created from
its GGUF + Modelfile. No inference, no model loading, no heuristics.

Why this replaced inference-based probing:
- Inference probes were slow (30-180s of cold model load per probe)
- Inference probes were unreliable (false positives where text-only
  models hallucinated colour answers; false negatives where actual
  vision models triggered the reject-phrase regex by accident)
- Ollama's metadata is the authoritative source — `ollama show` exposes
  the exact same fields that drive the model's runtime behaviour

The capability strings Ollama emits, mapped to our cache shape:
- "completion" → text (universal — every chat model is "completion")
- "tools"      → tool_calling = "native"
- "vision"     → vision = True
- "thinking"   → reasoning = True
- "embedding"  → not used yet (embedding-only models excluded from chat dropdown)
- "insert"     → not used
"""
from __future__ import annotations

import logging
from typing import Literal

import httpx

log = logging.getLogger(__name__)


ToolCalling = Literal["native", "rejected", "error"]


async def probe_all(ollama_url: str, model: str, timeout: float = 10.0) -> dict:
    """Return the capability dict for one model. Fast — single metadata call."""
    body = {"name": model}
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(f"{ollama_url}/api/show", json=body)
    except httpx.HTTPError as exc:
        log.warning("show probe network error for %s: %s", model, exc)
        return _error_record(f"network: {exc}")

    if r.status_code != 200:
        return _error_record(f"HTTP {r.status_code}")

    try:
        data = r.json()
    except ValueError as exc:
        return _error_record(f"non-JSON response: {exc}")

    caps_list = data.get("capabilities") or []
    caps = {c.lower() for c in caps_list if isinstance(c, str)}

    return {
        "tool_calling": "native" if "tools" in caps else "rejected",
        "vision": "vision" in caps,
        "audio": "audio" in caps,
        "reasoning": "thinking" in caps,
    }


def _error_record(reason: str) -> dict:
    return {
        "tool_calling": "error",
        "vision": False,
        "audio": False,
        "reasoning": False,
        "probe_error": reason,
    }
