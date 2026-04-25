"""Host-environment facts injected into the system prompt at request time.

This is the cheapest, most effective fix for "the model hallucinated facts
about my system" — give it the actual facts so the truthfulness rule has
something to be truthful about.

Static facts (OS, hostname, machine) are gathered once and cached.
Dynamic facts (current date/time) refresh per call so they never go stale.
"""
from __future__ import annotations

import os
import platform
import socket
from datetime import datetime, timezone

_STATIC_LINES: list[str] | None = None


def _read_distro() -> str:
    try:
        with open("/etc/os-release") as f:
            for line in f:
                if line.startswith("PRETTY_NAME="):
                    return line.split("=", 1)[1].strip().strip('"')
    except FileNotFoundError:
        pass
    return "unknown"


def _gather_static() -> list[str]:
    return [
        f"- OS: {platform.system()} {platform.release()}",
        f"- Distribution: {_read_distro()}",
        f"- Architecture: {platform.machine()}",
        f"- Hostname: {socket.gethostname()}",
        f"- User: {os.getenv('USER', 'unknown')}",
        f"- Python: {platform.python_version()}",
    ]


def host_context_block() -> str:
    """Concise factual block to append to the system prompt."""
    global _STATIC_LINES
    if _STATIC_LINES is None:
        _STATIC_LINES = _gather_static()
    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="minutes")
    return "\n".join([
        "",
        "## Host environment (these are real facts — do not contradict or guess otherwise)",
        f"- Current local date/time: {now}",
        *_STATIC_LINES,
    ])
