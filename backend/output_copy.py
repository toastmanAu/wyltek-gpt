"""Copy model-produced files into a host-visible output directory.

The workspace stays the source of truth (chat links resolve through
``/api/files/{session}/{name}``). This module just mirrors model-produced
outputs into a folder the user can browse from their desktop file
manager — defaults to ``~/Downloads/wyltek-gpt-output/``.

Copies, not symlinks: copies survive workspace cleanup, and symlinks
into a session-suffixed workspace path would be confusing in a file
manager. Idempotent — overwrites existing files of the same name.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

log = logging.getLogger(__name__)


def copy_to_output(source: Path, output_dir: Path | None) -> Path | None:
    """Copy ``source`` into ``output_dir``. Returns the destination path,
    or ``None`` if ``output_dir`` is unset.

    Failures are logged but not raised — the chat link still points at
    the workspace copy, so the user is never blocked by an unreachable
    output directory (e.g. permissions, disk full)."""
    if output_dir is None:
        return None
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        dest = output_dir / source.name
        shutil.copy2(source, dest)
        log.info("output copy: %s → %s", source.name, dest)
        return dest
    except OSError as exc:
        log.warning("output copy failed for %s: %s", source, exc)
        return None
