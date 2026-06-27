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


import re

MAX_SOURCE = 200_000
PROFILES = {"ckb"}
CODE_RE = re.compile(r"^[A-Za-z0-9_]+$")
NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")

CELLC_TOOL_NAMES = frozenset({
    "cellc_check", "cellc_explain", "cellc_get_example",
    "cellc_language_reference", "cellc_metadata", "cellc_list_examples",
})


def tool_schemas() -> list[dict]:
    def fn(name, desc, props, required):
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": desc,
                "parameters": {"type": "object", "properties": props, "required": required},
            },
        }
    return [
        fn("cellc_check",
           "Type-check a CellScript .cell contract. Returns ok + diagnostics (line/column/code/message).",
           {"source": {"type": "string", "description": "full .cell source"},
            "target_profile": {"type": "string", "enum": ["ckb"], "description": "target profile (default ckb)"}},
           ["source"]),
        fn("cellc_explain",
           "Explain a CellScript error code (e.g. E0014) with description and fix hint.",
           {"code": {"type": "string", "description": "error code or name, e.g. E0014"}},
           ["code"]),
        fn("cellc_get_example",
           "Return a bundled example .cell contract's source by name (e.g. token, nft).",
           {"name": {"type": "string", "description": "example name"}},
           ["name"]),
        fn("cellc_language_reference",
           "Return the full CellScript language surface (keywords, effects, a worked example).",
           {}, []),
        fn("cellc_metadata",
           "Compiler metadata for a contract: resources, actions, effects, obligations (summary).",
           {"source": {"type": "string", "description": "full .cell source"},
            "target_profile": {"type": "string", "enum": ["ckb"], "description": "target profile (default ckb)"}},
           ["source"]),
        fn("cellc_list_examples",
           "List bundled example .cell contracts (names + one-line summaries).",
           {}, []),
    ]


def _tool_err(msg: str) -> dict:
    return {"ok": False, "tool_error": True, "exit_code": -1, "stderr": msg}


def dispatch(name: str, arguments: dict) -> dict:
    arguments = arguments or {}
    if name == "cellc_check" or name == "cellc_metadata":
        source = arguments.get("source")
        if not isinstance(source, str) or not source.strip():
            return _tool_err("missing or empty 'source'")
        if len(source) > MAX_SOURCE:
            return _tool_err(f"source too long ({len(source)} > {MAX_SOURCE})")
        profile = arguments.get("target_profile", "ckb")
        if profile not in PROFILES:
            return _tool_err(f"invalid target_profile {profile!r}")
        full = bool(arguments.get("full"))
        if name == "cellc_check":
            return check(source, target_profile=profile, full=full)
        return metadata(source, target_profile=profile, full=full)
    if name == "cellc_explain":
        code = arguments.get("code")
        if not isinstance(code, str) or not CODE_RE.match(code):
            return _tool_err("invalid 'code' (expected an error code/name like E0014)")
        return explain(code)
    if name == "cellc_get_example":
        ex = arguments.get("name")
        if not isinstance(ex, str) or not NAME_RE.match(ex):
            return _tool_err("invalid example 'name'")
        return get_example(ex)
    if name == "cellc_language_reference":
        return {"reference": language_reference()}
    if name == "cellc_list_examples":
        return {"examples": list_examples()}
    return _tool_err(f"unknown cellc tool {name!r}")
