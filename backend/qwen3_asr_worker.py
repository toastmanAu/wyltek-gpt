#!/usr/bin/env python3
"""Qwen3-ASR transcription worker — HTTP daemon on port 9096.

Loads the model once at startup and serves transcription requests.
Uses the qwen-asr package (system Python) with ROCm GPU support.

Environment:
    QWEN3_ASR_MODEL_SIZE — "0.6B" or "1.7B" (default: "1.7B")
    QWEN3_ASR_DEVICE     — e.g. "cuda:0" (default: auto)
    QWEN3_ASR_PORT       — listen port (default: 9096)
    QWEN3_ASR_HOST       — listen host (default: 127.0.0.1)
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

log = logging.getLogger("qwen3-asr-worker")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

# ── Configuration ──────────────────────────────────────────────────────
MODEL_SIZE = os.getenv("QWEN3_ASR_MODEL_SIZE", "1.7B").strip()
DEVICE = os.getenv("QWEN3_ASR_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
HOST = os.getenv("QWEN3_ASR_HOST", "127.0.0.1")
PORT = int(os.getenv("QWEN3_ASR_PORT", "9096"))

# Resolve cached model path from HF hub cache
HF_HUB_CACHE = os.getenv("HF_HUB_CACHE", "/bulk/huggingface-cache/hub")
MODEL_ID = f"Qwen/Qwen3-ASR-{MODEL_SIZE}"


def _resolve_cached_path(model_id: str) -> str | None:
    """Find the snapshot directory for a cached model."""
    safe_name = model_id.replace("/", "--")
    # Try HF_HUB_CACHE first, then HF_HOME, then common locations
    candidates = [
        Path(HF_HUB_CACHE) / f"models--{safe_name}",
        Path(HF_HUB_CACHE).parent / f"models--{safe_name}",
        Path(os.getenv("HF_HOME", "/bulk/huggingface-cache")) / f"models--{safe_name}",
    ]
    for model_dir in candidates:
        if not model_dir.exists():
            continue
        snap_dir = model_dir / "snapshots"
        if not snap_dir.exists():
            continue
        for entry in snap_dir.iterdir():
            if entry.is_dir():
                return str(entry)
    return None


CACHED_PATH = _resolve_cached_path(MODEL_ID)
if CACHED_PATH is None:
    log.error("Model %s not found in cache at %s", MODEL_ID, HF_HUB_CACHE)
    raise SystemExit(1)

log.info("Loading Qwen3-ASR %s from %s on %s", MODEL_SIZE, CACHED_PATH, DEVICE)

# ── Model load ─────────────────────────────────────────────────────────
from qwen_asr import Qwen3ASRModel  # noqa: E402

_model = Qwen3ASRModel.from_pretrained(
    CACHED_PATH,
    dtype=torch.bfloat16,
    device_map=DEVICE,
    max_new_tokens=256,
)
log.info("Model loaded — ready on %s", DEVICE)

# ── Helpers ────────────────────────────────────────────────────────────
VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v", ".flv"}


def _extract_audio(input_path: Path) -> Path:
    """Extract audio from video to a temporary 16 kHz mono WAV."""
    suffix = input_path.suffix.lower()
    if suffix not in VIDEO_EXTS:
        return input_path
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg required for video audio extraction")
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-vn", "-acodec", "pcm_s16le",
        "-ac", "1", "-ar", "16000",
        tmp.name,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        os.unlink(tmp.name)
        tail = (result.stderr or "")[-500:]
        raise RuntimeError(f"ffmpeg exit {result.returncode}: {tail}")
    return Path(tmp.name)


# ── FastAPI app ────────────────────────────────────────────────────────
app = FastAPI(title="qwen3-asr-worker")


class TranscribeRequest(BaseModel):
    file_path: str
    language: str | None = None  # None = auto-detect


class TranscribeResponse(BaseModel):
    text: str
    language: str
    elapsed_s: float


@app.get("/status")
async def status():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "model_size": MODEL_SIZE,
        "device": DEVICE,
        "cached_path": CACHED_PATH,
    }


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(req: TranscribeRequest):
    t0 = time.perf_counter()
    path = Path(req.file_path)
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"file not found: {req.file_path}")

    audio_path = path
    tmp_audio: Path | None = None
    try:
        audio_path = _extract_audio(path)
        if audio_path != path:
            tmp_audio = audio_path
            log.info("extracted audio from %s → %s", path.name, audio_path.name)

        results = _model.transcribe(audio=str(audio_path), language=req.language)
    except Exception as exc:
        log.exception("transcription failed for %s", req.file_path)
        raise HTTPException(status_code=500, detail=f"transcription failed: {exc}") from exc
    finally:
        if tmp_audio is not None:
            try:
                os.unlink(tmp_audio)
            except OSError:
                pass

    elapsed = time.perf_counter() - t0
    result = results[0]
    log.info("transcribed %s in %.2fs (lang=%r)", path.name, elapsed, result.language)
    return TranscribeResponse(
        text=result.text or "",
        language=result.language or "",
        elapsed_s=round(elapsed, 2),
    )


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
