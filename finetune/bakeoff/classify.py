"""Classify a model response as native tool call, op: fallback, or neither.

Mirrors what backend/app.py + frontend/app.js actually accept. Model families
differ in surface syntax (Qwen emits <tool_call> tags, others emit bare JSON),
so both count as native. The fallback path is the fenced ```op:<name>\\n{...}
form backend/app.py's operations prompt block instructs non-tool-calling
models to emit (see `_operations_prompt_block`), which frontend/app.js parses
with `OP_FENCE_RE`. The older unfenced `op:<name> {...}` form is also accepted
for back-compat with earlier prompt revisions.

This module's inputs are model-generated text -- including brace-dense
HTML/CSS/JS demo output and, occasionally, degenerate/repetitive output from
a candidate that is misbehaving. `classify_response` is therefore total: for
*any* `str` input it returns one of "native", "fallback", "none" and never
raises. The probe driver (Task 5) calls it once per response in a bake-off
loop, and a single bad sample must degrade to "none" rather than aborting
the whole run.
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
    except (ValueError, TypeError, RecursionError):
        # RecursionError (a RuntimeError subclass, NOT a ValueError) is
        # what CPython's json scanner raises on deeply-nested input past
        # its internal depth guard. A `<tool_call>` payload can be
        # attacker/model-controlled degenerate JSON just as easily as a
        # bare one, so this needs the same guard as `_has_bare_json_tool_call`.
        return False
    return _is_tool_call_payload(obj)


def _looks_like_json_object_start(text: str, idx: int) -> bool:
    """Cheap O(1) check: could a JSON object plausibly open at ``idx``?

    JSON's grammar for an object is ``'{' ws* ( '"' ... | '}' )`` — after the
    brace and any whitespace, the next character must be a quote (the first
    key) or the closing brace (an empty object). Brace-dense non-JSON text
    (CSS rule blocks, JS control-flow blocks) almost always has an
    identifier, keyword, or another brace right after the opening brace, so
    this rejects those positions before ever calling ``raw_decode``.

    That matters because a *failed* ``raw_decode`` constructs a
    ``json.JSONDecodeError``, whose line/column computation
    (``doc.count("\\n", 0, pos)``) is O(position-in-document). Summed over
    every ``{`` in a large brace-dense document, that made the scan O(n^2).
    Skipping the call entirely for positions that cannot possibly start a
    JSON object keeps the scan fast on the documents that actually matter
    for this probe: model-generated HTML/CSS/JS demos, which are brace-dense
    but rarely have a quote or closing brace immediately after most of
    their ``{`` characters.
    """
    j = idx + 1
    n = len(text)
    while j < n and text[j] in " \t\r\n":
        j += 1
    return j < n and text[j] in ('"', "}")


def _has_bare_json_tool_call(text: str) -> bool:
    """Scan every ``{`` position for a standalone decodable JSON object.

    A single greedy regex spanning the first ``{`` to the last ``}`` in the
    whole response does not isolate one JSON object — any stray brace
    elsewhere (trailing prose, a second tool call) breaks the parse or
    silently merges unrelated objects. Trying at each ``{`` with
    ``json.JSONDecoder.raw_decode`` lets the real JSON grammar find each
    object's true extent instead of guessing with a regex.

    Three things can make this scan blow up on adversarial or merely
    brace-dense input, and all three are handled here:

    - Deeply-nested JSON makes ``raw_decode`` raise ``RecursionError``, not
      ``ValueError`` (`RecursionError` is a `RuntimeError` subclass). Left
      uncaught this would abort the whole scan for one degenerate response.
      Caught per-position (not at the top of `classify_response`), a
      RecursionError at one `{` still lets the scan continue and find a
      genuine tool call elsewhere in the same text.
    - Brace-dense non-JSON text whose ``{`` is not even shaped like an
      object opening (CSS rule blocks, JS control-flow blocks) is pruned by
      `_looks_like_json_object_start` before `raw_decode` is ever called.
    - Brace-dense text that DOES look like an object opening (``{"`` or
      ``{}``) but fails to decode (trailing comma, single-quoted keys, an
      unterminated string cut off mid-generation) still reaches
      `raw_decode`, and a *failed* `raw_decode` constructs a
      `json.JSONDecodeError`, whose line/column computation
      (``doc.count("\\n", 0, pos)``) is O(position-in-document). Summed over
      every such position that's just object-shaped-but-invalid, that's
      O(n^2) again — this is the round-3 finding, reproduced by an ordinary
      trailing-comma JS object literal repeated across a large document.

      The fix: a real tool call payload is a dict with both a "name" key
      and an "arguments" key (`_is_tool_call_payload`), and in every case
      this module has to detect, those keys are written literally as the
      substrings ``"name"`` and ``"arguments"`` in the source text (JSON
      does not require escaping ordinary ASCII letters, and nothing this
      probe measures does so). That means a genuine tool call's own
      substring necessarily contains both literal substrings somewhere at
      or after its opening `{`. So: if the literal substring ``"name"``
      does not occur anywhere in `text` at or after `idx`, no object
      starting at `idx` can be a tool call payload, and `raw_decode` is
      skipped — same for ``"arguments"``. The last occurrence of each
      substring in the whole text (`str.rfind`, computed once, O(n)) is
      enough for the "at or after idx" check: `idx <= last_occurrence` is
      a valid over-approximation (it may still attempt a decode that
      turns out to fail, but it can never skip a position that could
      genuinely succeed, since a genuine tool call's key literally lives
      at some position >= idx, so the LAST such position is also >= idx).

      This is sound but not a complete defense against every conceivable
      adversarial input: a document that densely repeats BOTH literal
      substrings ``"name"`` and ``"arguments"`` alongside many
      invalid-JSON-shaped braces could still accumulate failed-decode
      cost. That shape does not match any of the round-3 finding's
      reproducers (trailing comma, single-quoted key, truncated
      data-URI) or this probe's real input distribution (model-generated
      HTML/CSS/JS demos), so it's out of scope here.
    """
    last_name_pos = text.rfind('"name"')
    if last_name_pos == -1:
        return False
    last_arguments_pos = text.rfind('"arguments"')
    if last_arguments_pos == -1:
        return False

    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch != "{" or not _looks_like_json_object_start(text, idx):
            continue
        if idx > last_name_pos or idx > last_arguments_pos:
            continue
        try:
            obj, _end = decoder.raw_decode(text, idx)
        except (ValueError, RecursionError):
            continue
        if _is_tool_call_payload(obj):
            return True
    return False


def classify_response(text: str) -> str:
    """Classify ``text`` as "native", "fallback", or "none".

    Total: never raises for any ``str`` input, including empty strings and
    deeply-nested/degenerate JSON-like text (see module docstring). A
    RecursionError surfacing from one candidate match is treated as "that
    match doesn't count", not as an error that aborts classification of the
    rest of the text — the same input can still classify as native or
    fallback via a different match elsewhere.
    """
    if not text:
        return NONE

    try:
        tag_matches = _TOOL_CALL_TAG.findall(text)
    except RecursionError:
        tag_matches = []
    for match in tag_matches:
        if _payload_from_string(match):
            return NATIVE

    if _has_bare_json_tool_call(text):
        return NATIVE

    try:
        is_fallback = bool(_OP_FENCED.search(text) or _OP_UNFENCED.search(text))
    except RecursionError:
        is_fallback = False
    if is_fallback:
        return FALLBACK

    return NONE
