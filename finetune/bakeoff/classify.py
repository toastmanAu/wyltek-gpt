"""Classify a model response as native tool call, op: fallback, or neither.

Mirrors what backend/app.py + frontend/app.js actually accept. Model families
differ in surface syntax (Qwen emits <tool_call> tags, others emit bare JSON),
so both count as native. The fallback path is the fenced ```op:<name>\\n{...}
form backend/app.py's operations prompt block instructs non-tool-calling
models to emit (see `_operations_prompt_block`), which frontend/app.js parses
with `OP_FENCE_RE`. The older unfenced `op:<name> {...}` form is also accepted
for back-compat with earlier prompt revisions.
"""
import json
import re

_TOOL_CALL_TAG = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_OP_FENCED = re.compile(r"```op:[A-Za-z_][A-Za-z0-9_]*\s*\n")
_OP_UNFENCED = re.compile(r"^\s*op:([A-Za-z_][A-Za-z0-9_]*)\s*\{", re.MULTILINE)

NATIVE = "native"
FALLBACK = "fallback"
NONE = "none"


def _is_tool_call_payload(obj: object) -> bool:
    return isinstance(obj, dict) and "name" in obj and "arguments" in obj


def _payload_from_string(raw: str) -> bool:
    try:
        obj = json.loads(raw)
    except (ValueError, TypeError):
        return False
    return _is_tool_call_payload(obj)


def _has_bare_json_tool_call(text: str) -> bool:
    """Scan every ``{`` position for a standalone decodable JSON object.

    A single greedy regex spanning the first ``{`` to the last ``}`` in the
    whole response does not isolate one JSON object — any stray brace
    elsewhere (trailing prose, a second tool call) breaks the parse or
    silently merges unrelated objects. Trying at each ``{`` with
    ``json.JSONDecoder.raw_decode`` lets the real JSON grammar find each
    object's true extent instead of guessing with a regex.
    """
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _end = decoder.raw_decode(text, idx)
        except ValueError:
            continue
        if _is_tool_call_payload(obj):
            return True
    return False


def classify_response(text: str) -> str:
    if not text:
        return NONE

    for match in _TOOL_CALL_TAG.findall(text):
        if _payload_from_string(match):
            return NATIVE

    if _has_bare_json_tool_call(text):
        return NATIVE

    if _OP_FENCED.search(text) or _OP_UNFENCED.search(text):
        return FALLBACK

    return NONE
