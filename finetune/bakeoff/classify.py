"""Classify a model response as native tool call, op: fallback, or neither.

Mirrors what backend/app.py accepts. Model families differ in surface syntax
(Qwen emits <tool_call> tags, others emit bare JSON), so both count as native.
The `op:<name> {...}` form is the degraded client-side-parsed path we are
trying to make unnecessary.
"""
import json
import re

_TOOL_CALL_TAG = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_OP_BLOCK = re.compile(r"^\s*op:([A-Za-z_][A-Za-z0-9_]*)\s*\{", re.MULTILINE)
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)

NATIVE = "native"
FALLBACK = "fallback"
NONE = "none"


def _is_tool_call_payload(raw: str) -> bool:
    try:
        obj = json.loads(raw)
    except (ValueError, TypeError):
        return False
    return isinstance(obj, dict) and "name" in obj and "arguments" in obj


def classify_response(text: str) -> str:
    if not text:
        return NONE

    for match in _TOOL_CALL_TAG.findall(text):
        if _is_tool_call_payload(match):
            return NATIVE

    candidate = _JSON_OBJECT.search(text)
    if candidate and _is_tool_call_payload(candidate.group(0)):
        return NATIVE

    if _OP_BLOCK.search(text):
        return FALLBACK

    return NONE
