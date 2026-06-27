"""In-process bridge to the cellc CellScript compiler tooling.

Imports the cellc-mcp server's tool functions directly (both are local
Python on driveThree) and exposes thin JSON-able wrappers. All heavy
lifting — package synthesis, output tiering, module-scan isolation, the
never-raises contract — lives in ``cellc_mcp`` and must not be duplicated
here. If ``cellc_mcp`` is not installed, this module still imports and
``available()`` returns False so wyltek-gpt boots without the feature.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

try:
    from cellc_mcp import server as _server
    from cellc_mcp.runner import CellcNotFound as _CellcNotFound
    from cellc_mcp.runner import find_cellc as _find_cellc
    _IMPORT_OK = True
except Exception as exc:  # pragma: no cover - exercised only without cellc_mcp
    log.warning("cellc_mcp not importable — cellc bridge disabled: %s", exc)
    _server = None  # type: ignore[assignment]
    _CellcNotFound = Exception  # type: ignore[assignment,misc]
    _find_cellc = None  # type: ignore[assignment]
    _IMPORT_OK = False


def _binary() -> str | None:
    if not _IMPORT_OK or _find_cellc is None:
        return None
    try:
        return _find_cellc()
    except _CellcNotFound:
        return None


def available() -> bool:
    return _binary() is not None


def status() -> dict:
    binary = _binary()
    return {"available": binary is not None, "binary": binary}


def check(source: str, target_profile: str = "ckb", full: bool = False) -> dict:
    return _server.cellc_check(source=source, target_profile=target_profile, full=full)


def explain(code: str) -> dict:
    return _server.cellc_explain(code=code)


def metadata(source: str, target_profile: str = "ckb", full: bool = False) -> dict:
    return _server.cellc_metadata(source=source, target_profile=target_profile, full=full)


def language_reference() -> str:
    return _server.cellc_language_reference()


def list_examples() -> list[dict]:
    return _server.cellc_list_examples()


def get_example(name: str) -> dict:
    return _server.cellc_get_example(name=name)
