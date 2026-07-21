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

_OPEN_TAG = "<tool_call>"
_CLOSE_TAG = "</tool_call>"
_OP_FENCED = re.compile(r"```op:[A-Za-z_][A-Za-z0-9_]*\s*\n")
_OP_UNFENCED = re.compile(r"^\s*op:([A-Za-z_][A-Za-z0-9_]*)\s*\{", re.MULTILINE)

NATIVE = "native"
FALLBACK = "fallback"
NONE = "none"


def _is_tool_call_payload(obj: object) -> bool:
    return isinstance(obj, dict) and "name" in obj and "arguments" in obj


def _iter_tool_call_payloads(text: str):
    """Yield the raw text between each ``<tool_call>``/``</tool_call>`` pair.

    Round 4 finding: this used to be a regex --
    ``r"<tool_call>\\s*(\\{.*?\\})\\s*</tool_call>"`` with ``re.DOTALL`` -- with
    its OWN O(n^2) blowup, entirely independent of the bare-JSON scan
    hardened in rounds 1-3. `classify_response` runs this scan FIRST,
    unconditionally, before the bare-JSON scan ever gets a chance to run.
    When a `{` follows `<tool_call>` but no `}` (or no closing tag at all)
    exists anywhere later in the text, the non-greedy `.*?` must scan
    forward to end-of-string trying to complete the match, and it repeats
    that scan-to-end at EVERY one of the n `<tool_call>` occurrences in an
    input like `'<tool_call>{' * n` -- O(n) wasted work at each of O(n)
    start positions, i.e. O(n^2) overall. Measured: n=16000 (188KB) took
    6.349s; a control input with the same number of tags but no `{` (so the
    non-greedy scan never has anything to chase) ran in 0.0004s, confirming
    the `{` is what triggers it, not the tag repetition by itself.

    The fix removes the possibility structurally rather than patching this
    instance: `str.find` performs a linear-time, non-backtracking substring
    search with no notion of "try to extend the match, fail, try a longer
    extension" -- there is nothing here that can rescan the same span of
    text a second time chasing a match that doesn't exist. `pos` only ever
    advances forward (never resets backward) between iterations, so the
    `text.find` calls across the whole generator collectively perform a
    small constant number of linear passes over `text` -- O(n) total, for
    any input whatsoever, including the adversarial repetition above.

    Pairing is nearest-open-to-nearest-close, matching the original regex's
    behavior for non-overlapping, non-nested tags (both find the closest
    valid closing point rather than the outermost or farthest one). If an
    open tag has no closing tag anywhere later in `text`, the generator
    stops rather than continuing to search past it -- correct because nothing
    after an unmatched open tag can retroactively close it, and this is
    exactly what keeps an unterminated `<tool_call>` from making the scan
    consume (or misattribute) the remainder of the document.
    """
    pos = 0
    n = len(text)
    while pos <= n:
        start = text.find(_OPEN_TAG, pos)
        if start == -1:
            return
        content_start = start + len(_OPEN_TAG)
        end = text.find(_CLOSE_TAG, content_start)
        if end == -1:
            return
        yield text[content_start:end]
        pos = end + len(_CLOSE_TAG)


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
      and an "arguments" key (`_is_tool_call_payload`). This pre-filter
      ASSUMES those keys are written literally as the substrings ``"name"``
      and ``"arguments"`` in the source text — true whenever a model emits
      ordinary ASCII key names un-escaped, which is how every model this
      probe has observed writes JSON, but NOT a property JSON's grammar
      guarantees. A key can legally be written with a `\\u` escape (e.g.
      ``"na\\u006de"`` decodes to ``"name"``), in which case the literal
      substring ``"name"`` never appears in the source text at all, and
      this pre-filter will skip `raw_decode` at that position — even though
      the object genuinely would decode to a valid tool call payload.

      KNOWN ACCEPTED LIMITATION, not a bug to fix here: such an object is
      scored "none" instead of "native" — a false negative that slightly
      understates a candidate's native-tool-call rate. Real-world impact is
      judged LOW (local LLMs essentially never `\\u`-escape ASCII key
      names), and contorting this scan to also decode escaped keys before
      knowing whether a `{` is even JSON-shaped is not worth the
      complexity for a shape that hasn't been observed in practice.

      Given that assumption holds, if the literal substring ``"name"``
      does not occur anywhere in `text` at or after `idx`, no object
      starting at `idx` can be a tool call payload (under the assumption),
      and `raw_decode` is skipped — same for ``"arguments"``. The last
      occurrence of each substring in the whole text (`str.rfind`,
      computed once, O(n)) is enough for the "at or after idx" check:
      `idx <= last_occurrence` is a valid over-approximation (it may still
      attempt a decode that turns out to fail, but it can never skip a
      position that could genuinely succeed under the assumption, since
      such a key literally lives at some position >= idx, so the LAST such
      position is also >= idx).

      This is sound (given the assumption) but not a complete defense
      against every conceivable adversarial input: a document that
      densely repeats BOTH literal substrings ``"name"`` and
      ``"arguments"`` alongside many invalid-JSON-shaped braces could
      still accumulate failed-decode cost. That shape does not match any
      of the round-3 finding's reproducers (trailing comma, single-quoted
      key, truncated data-URI) or this probe's real input distribution
      (model-generated HTML/CSS/JS demos), so it's out of scope here.
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

    for candidate in _iter_tool_call_payloads(text):
        if _payload_from_string(candidate):
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
