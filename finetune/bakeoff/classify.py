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


class _CheapFailureDoc(str):
    """A ``str`` subclass that makes a failed decode's exception O(1).

    ROOT CAUSE (round 5): every prior round (2, 3, 4) narrowed *which*
    ``{`` positions reach the JSON decoder, but never touched the actual
    cost of a *failed* decode there, so each round a new input shape that
    slipped past the latest pre-filter reproduced the same O(n^2) blowup.
    The real cost sits in ``json.JSONDecodeError.__init__``, which computes
    ``doc.count("\\n", 0, pos)`` and ``doc.rfind("\\n", 0, pos)`` — both
    O(position-in-document) — every time it's constructed. That happens on
    *every* failed attempt, regardless of which decoder entry point
    triggered it: not only the "Expecting value" case (which
    ``json.JSONDecoder.raw_decode`` builds by wrapping a cheap
    ``StopIteration``), but also structural errors raised *directly* deep
    inside object parsing (e.g. "Expecting ':' delimiter") — which
    ``scan_once`` raises as a full ``JSONDecodeError`` too, at the exact
    same O(pos) cost. This was verified directly against this
    interpreter's C-accelerated scanner: for the round-5 surviving
    reproducer below, ``scan_once`` and ``raw_decode`` cost the *same*
    (~28ms per 200 failing attempts at a position ~736,000 chars in) —
    switching decoder methods alone does not fix this input shape.

    `_has_bare_json_tool_call` never reads a raised exception's message,
    ``.lineno``, or ``.colno`` — it only branches on the exception's type.
    So the correctness of those fields is irrelevant here; only their cost
    is. Wrapping `text` in this subclass ONCE (a single O(n) copy, done
    once per `_has_bare_json_tool_call` call — not once per candidate
    position) makes `count`/`rfind` O(1) stubs instead of real O(pos)
    scans, so every failed-decode exception this scan constructs is O(1)
    to build no matter how deep into the document it fails. That removes
    the root cause structurally: it no longer matters how many positions
    make it past the pre-filters below, or what specific error message the
    decoder raises at each one — a failure is cheap everywhere in the
    document, not just near its start.
    """

    __slots__ = ()

    def count(self, *_args, **_kwargs) -> int:
        return 0

    def rfind(self, *_args, **_kwargs) -> int:
        return -1


def _has_bare_json_tool_call(text: str) -> bool:
    """Scan every ``{`` position for a standalone decodable JSON object.

    A single greedy regex spanning the first ``{`` to the last ``}`` in the
    whole response does not isolate one JSON object — any stray brace
    elsewhere (trailing prose, a second tool call) breaks the parse or
    silently merges unrelated objects. Trying at each ``{`` with
    ``json.JSONDecoder.scan_once`` (the decoder's underlying scanner) lets
    the real JSON grammar find each object's true extent instead of
    guessing with a regex.

    Two things narrow which positions even reach the decoder — cheap
    performance filters now, not correctness-critical after round 5 (see
    below), but still worth keeping since they avoid pointless scanner
    calls on the majority of brace-dense non-JSON positions:

    - Brace-dense non-JSON text whose ``{`` is not even shaped like an
      object opening (CSS rule blocks, JS control-flow blocks) is pruned by
      `_looks_like_json_object_start` before the decoder is ever called.
    - A real tool call payload is a dict with both a "name" key and an
      "arguments" key (`_is_tool_call_payload`). This pre-filter ASSUMES
      those keys are written literally as the substrings ``"name"`` and
      ``"arguments"`` in the source text — true whenever a model emits
      ordinary ASCII key names un-escaped, which is how every model this
      probe has observed writes JSON, but NOT a property JSON's grammar
      guarantees. A key can legally be written with a `\\u` escape (e.g.
      ``"na\\u006de"`` decodes to ``"name"``), in which case the literal
      substring ``"name"`` never appears in the source text at all, and
      this pre-filter will skip the decoder at that position — even though
      the object genuinely would decode to a valid tool call payload.

      KNOWN ACCEPTED LIMITATION, not a bug to fix here: such an object is
      scored "none" instead of "native" — a false negative that slightly
      understates a candidate's native-tool-call rate. Real-world impact is
      judged LOW (local LLMs essentially never `\\u`-escape ASCII key
      names), and contorting this scan to also decode escaped keys before
      knowing whether a `{` is even JSON-shaped is not worth the
      complexity for a shape that hasn't been observed in practice.

      The last occurrence of each substring in the whole text
      (`str.rfind`, computed once, O(n)) bounds which positions can
      possibly succeed: `idx <= last_occurrence` for both "name" and
      "arguments" is a valid over-approximation (it may still attempt a
      decode that turns out to fail, but it can never skip a position that
      could genuinely succeed under the assumption).

    Neither filter is (or needs to be) a *complete* defense against every
    conceivable adversarial brace-dense input — that used to matter a
    great deal, because a failed decode that slipped past both filters
    cost O(position-in-document). Round 5 removed that dependency: thanks
    to `_CheapFailureDoc`, a failed decode is O(1) everywhere in the
    document, so a position that slips through these filters just costs a
    cheap, bounded scanner attempt, not a potential O(n^2) blowup. The
    round-5 surviving reproducer — ``'{"name":"a","arguments"' * n``, which
    passes both filters above (object-start shaped, contains both literal
    substrings) but never has a closing brace — is exactly this case: it
    still reaches the decoder at every qualifying position, but each
    failure is now cheap regardless of how far into the document it is.

    Two exception types still need explicit handling per position (not
    just at the top of `classify_response`), since either one at a single
    `{` must not abort the scan for the rest of the text:

    - `scan_once` raises a bare `StopIteration` (not wrapped into
      `JSONDecodeError`) for its own "no value here" case. This function is
      an ordinary function, not a generator, so catching `StopIteration`
      here is unremarkable — but this scan is invoked from
      `classify_response`, which is itself called per-candidate from a
      probe driver loop, not from inside `_iter_tool_call_payloads`'s
      generator frame (that generator never touches JSON decoding at all),
      so there is no PEP 479 escaping-`StopIteration`-becomes-`RuntimeError`
      risk here regardless.
    - Deeply-nested JSON makes the scanner raise `RecursionError`, not
      `ValueError` (`RecursionError` is a `RuntimeError` subclass). Left
      uncaught this would abort the whole scan for one degenerate
      response. Caught per-position, a RecursionError at one `{` still
      lets the scan continue and find a genuine tool call elsewhere in the
      same text.

    `json.JSONDecodeError` is itself a `ValueError` subclass, so catching
    `(StopIteration, ValueError, RecursionError)` covers every failure mode
    the scanner is documented to raise.
    """
    last_name_pos = text.rfind('"name"')
    if last_name_pos == -1:
        return False
    last_arguments_pos = text.rfind('"arguments"')
    if last_arguments_pos == -1:
        return False

    cheap_text = _CheapFailureDoc(text)
    scan_once = json.JSONDecoder().scan_once
    for idx, ch in enumerate(text):
        if ch != "{" or not _looks_like_json_object_start(text, idx):
            continue
        if idx > last_name_pos or idx > last_arguments_pos:
            continue
        try:
            obj, _end = scan_once(cheap_text, idx)
        except (StopIteration, ValueError, RecursionError):
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
