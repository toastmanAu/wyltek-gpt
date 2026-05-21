"""Open-palette bridge — relays image-gen jobs to a locally-running
open-palette instance (default: http://localhost:7860).

Open-palette is the canonical name; ``Wyltek Studio`` is the brand
alias. Don't rename the module without renaming the upstream folder.

Wire protocol (from open-palette/server.py):
    POST /api/generate     multipart form    →  {"job_id": "abc12345"}
    GET  /api/job/{id}     poll              →  {"status": "queued|running|complete|error",
                                                 "progress": 0-100, "output_url": "/storage/{id}.png", ...}
    GET  /api/backends     health probe      →  list of configured backends + installed models
    GET  /storage/<file>                     →  the rendered image bytes

This module never speaks shell, never lets the caller pick the bridge URL
from chat input, and treats every poll/download timeout as a bridge
failure rather than a partial success.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import httpx

log = logging.getLogger(__name__)


class BridgeUnavailable(RuntimeError):
    """Raised when the bridge can't be reached at startup or call time."""


def health_probe(base_url: str, probe_path: str = "/api/backends",
                 timeout: float = 10.0) -> bool:
    """Return True iff open-palette responds 2xx within ``timeout`` seconds.

    Sync on purpose — runs at module-import time, before uvicorn's event
    loop spins up. Calling ``asyncio.run()`` here would conflict with
    uvicorn's loop on import-time-after-startup paths.
    """
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.get(f"{base_url.rstrip('/')}{probe_path}")
        return 200 <= r.status_code < 300
    except httpx.HTTPError as exc:
        log.info("open-palette probe to %s failed: %s", base_url, exc)
        return False


async def submit_image_job(
    *,
    base_url: str,
    params: dict[str, str],
    timeout: float = 30.0,
) -> str:
    """POST to /api/generate as multipart form. Returns the job_id."""
    base = base_url.rstrip("/")
    # Use multipart form so the wire matches what open-palette's UI sends.
    form_data: list[tuple[str, tuple[None, str]]] = [
        (k, (None, str(v))) for k, v in params.items() if v not in (None, "")
    ]
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(f"{base}/api/generate", files=form_data)
    except httpx.HTTPError as exc:
        raise BridgeUnavailable(f"open-palette /api/generate failed: {exc}") from exc

    if r.status_code != 200:
        raise BridgeUnavailable(
            f"open-palette /api/generate returned {r.status_code}: {r.text[:200]}"
        )
    job_id = r.json().get("job_id")
    if not job_id:
        raise BridgeUnavailable(f"open-palette /api/generate gave no job_id: {r.text[:200]}")
    return str(job_id)


async def poll_job(
    *,
    base_url: str,
    job_id: str,
    poll_interval: float = 1.5,
    overall_timeout: float = 600.0,
) -> dict:
    """Poll ``/api/job/{id}`` until status is complete or error.

    Returns the final job dict. Raises BridgeUnavailable on timeout or
    if open-palette reports ``status: "error"``.
    """
    base = base_url.rstrip("/")
    deadline = asyncio.get_event_loop().time() + overall_timeout
    async with httpx.AsyncClient(timeout=15.0) as client:
        while True:
            try:
                r = await client.get(f"{base}/api/job/{job_id}")
            except httpx.HTTPError as exc:
                raise BridgeUnavailable(f"poll failed: {exc}") from exc
            if r.status_code != 200:
                raise BridgeUnavailable(
                    f"poll {job_id} got HTTP {r.status_code}: {r.text[:200]}"
                )
            data = r.json()
            status = data.get("status")
            if status == "complete":
                return data
            if status == "error":
                raise BridgeUnavailable(f"open-palette job failed: {data.get('error', 'unknown')}")
            if asyncio.get_event_loop().time() >= deadline:
                raise BridgeUnavailable(f"open-palette job {job_id} did not complete in {overall_timeout}s")
            await asyncio.sleep(poll_interval)


async def download_result(
    *,
    base_url: str,
    output_url: str,
    dest_path: Path,
    timeout: float = 60.0,
) -> Path:
    """Download the bytes at ``output_url`` (relative to bridge base) into dest_path."""
    base = base_url.rstrip("/")
    full_url = output_url if output_url.startswith("http") else f"{base}{output_url}"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(full_url)
    except httpx.HTTPError as exc:
        raise BridgeUnavailable(f"download {full_url} failed: {exc}") from exc
    if r.status_code != 200:
        raise BridgeUnavailable(f"download {full_url} got HTTP {r.status_code}")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_bytes(r.content)
    return dest_path


async def generate_image(
    *,
    base_url: str,
    params: dict[str, str],
    workspace: Path,
    overall_timeout: float = 600.0,
) -> Path:
    """End-to-end: submit, poll, download into workspace. Returns saved path."""
    job_id = await submit_image_job(base_url=base_url, params=params)
    log.info("open-palette job submitted: %s", job_id)
    final = await poll_job(
        base_url=base_url, job_id=job_id, overall_timeout=overall_timeout,
    )
    output_url = final.get("output_url")
    if not output_url:
        raise BridgeUnavailable(f"job {job_id} complete but no output_url: {final}")
    ext = Path(output_url.split("?")[0]).suffix or ".png"
    dest = workspace / f"generated_{job_id}{ext}"
    return await download_result(
        base_url=base_url, output_url=output_url, dest_path=dest,
    )
