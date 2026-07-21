import time

from finetune.bakeoff.classify import classify_response


def test_qwen_style_tool_call_tags_are_native():
    text = '<tool_call>{"name": "preview_demo", "arguments": {"html": "<html></html>"}}</tool_call>'
    assert classify_response(text) == "native"


def test_bare_json_tool_call_is_native():
    text = '{"name": "preview_demo", "arguments": {"html": "<html></html>"}}'
    assert classify_response(text) == "native"


def test_native_detected_with_surrounding_prose():
    text = 'Sure, rendering that now.\n<tool_call>{"name": "save_demo", "arguments": {"demo_id": "d1"}}</tool_call>'
    assert classify_response(text) == "native"


def test_op_block_is_fallback():
    text = '```op:preview_demo\n{"html": "<html></html>"}\n```'
    assert classify_response(text) == "fallback"


def test_unfenced_op_block_is_fallback():
    # Back-compat: some earlier prompt revisions asked for the bare
    # `op:<name> {...}` form with no code fence. Still counts as fallback.
    text = 'op:preview_demo {"html": "<html></html>"}'
    assert classify_response(text) == "fallback"


def test_prose_only_is_none():
    assert classify_response("Here is some HTML you could use.") == "none"


def test_malformed_json_in_tags_is_not_native():
    text = '<tool_call>{"name": "preview_demo", "arguments":</tool_call>'
    assert classify_response(text) == "none"


def test_json_without_name_is_not_native():
    assert classify_response('{"arguments": {"html": "x"}}') == "none"


def test_native_wins_when_both_present():
    text = 'op:preview_demo {"html":"x"}\n<tool_call>{"name":"preview_demo","arguments":{"html":"x"}}</tool_call>'
    assert classify_response(text) == "native"


def test_empty_string_is_none():
    assert classify_response("") == "none"


def test_bare_json_call_followed_by_prose_with_stray_brace_is_native():
    # A greedy first-`{`-to-last-`}` regex would span from the tool call's
    # opening brace all the way to the brace in "{like colors}", producing a
    # blob that fails to parse as JSON at all.
    text = (
        '{"name": "preview_demo", "arguments": {"html": "<p>hi</p>"}}\n\n'
        "Tell me if you want changes {like colors}."
    )
    assert classify_response(text) == "native"


def test_two_bare_json_tool_calls_in_one_response_is_native():
    # A greedy regex spans from the first `{` to the LAST `}` here, capturing
    # both objects concatenated together, which is not valid JSON.
    text = (
        '{"name":"preview_demo","arguments":{"html":"<p>1</p>"}}\n'
        '{"name":"save_demo","arguments":{"demo_id":"d1"}}'
    )
    assert classify_response(text) == "native"


def test_js_object_literal_with_name_and_arguments_keys_in_code_fence_is_native():
    # The probe prompts ask the model to write HTML/JS demos, so generated
    # source commonly contains object literals. If such a literal happens to
    # be valid JSON *and* happens to use exactly the keys "name" and
    # "arguments", it is structurally identical to a real bare-JSON tool
    # call under this module's own definition of one (Task 4 interface:
    # "a bare JSON object with name and arguments") — there is no reliable
    # surface-syntax signal that distinguishes "demo source that coincidentally
    # matches" from "an actual tool call the model meant to make". We choose
    # NATIVE rather than trying to special-case code fences, since fence-aware
    # exclusion would also suppress genuine tool calls that some models wrap
    # in ```json fences, which is a worse failure mode for this measurement.
    text = (
        "Here's your demo:\n"
        "```html\n"
        "<script>\n"
        'const sceneConfig = {"name": "particle-system", "arguments": {"count": 500}};\n'
        "</script>\n"
        "```"
    )
    assert classify_response(text) == "native"


def test_native_wins_over_fenced_op_block():
    text = (
        '```op:preview_demo\n{"html":"x"}\n```\n'
        '<tool_call>{"name":"preview_demo","arguments":{"html":"x"}}</tool_call>'
    )
    assert classify_response(text) == "native"


def test_deeply_nested_json_does_not_raise_and_is_none():
    # `raw_decode` recurses into nested objects and raises RecursionError
    # (a RuntimeError subclass, NOT a ValueError) on input like this. The
    # bake-off probe driver calls classify_response once per model response
    # and must never abort a whole run because one candidate degenerated
    # into repetitive output -- a well-known LLM failure mode, and more
    # likely here since the bake-off is deliberately stress-testing
    # candidates. Degenerate/deeply-nested input has no valid top-level
    # tool-call payload, so "none" is correct, not a raised exception.
    #
    # NOTE: the finding's repro used depth 5000, but empirically on this
    # interpreter (CPython 3.13, C-accelerated json scanner) the threshold
    # where raw_decode flips from a plain JSONDecodeError to a RecursionError
    # sits at ~10,000 nested opens, not 5,000 -- and it does not track
    # sys.setrecursionlimit() (confirmed: lowering the limit to 200 does not
    # move the threshold), so it can't be forced down cheaply. 20,000 gives
    # 2x margin above the observed threshold so this reliably exercises the
    # RecursionError path even if another interpreter/platform's threshold
    # differs somewhat. This test is deliberately not fast (a few seconds):
    # it is proving crash-safety on genuinely pathological input, which is a
    # different concern from the O(n^2) brace-dense-but-valid-shaped-text
    # case covered by the timing test below.
    text = '{"a":' * 20000 + '1'
    assert classify_response(text) == "none"


def test_large_brace_dense_non_json_text_is_fast():
    # Synthesize >500KB of CSS-like text -- the normal case for this probe,
    # since prompts explicitly ask models to write full HTML/CSS/JS demos --
    # containing no tool call at all. The prior implementation was O(n^2)
    # here: every failed `raw_decode` attempt at every `{` constructed a
    # `json.JSONDecodeError`, whose line/column computation is
    # O(position-in-document), summed over every brace in the document.
    # Measured on the old code: 5,000 braces/89KB -> 0.048s, 10,000/179KB ->
    # 0.181s, 20,000/369KB -> 0.721s, 40,000/749KB -> 2.939s (~n^2 growth).
    # 1.0s is a generous bound: the fast (linear) path should finish in well
    # under 0.1s even on a loaded machine, but 1.0s is comfortably below
    # where the old quadratic scan would land on a document this size
    # (~2.9s at 749KB), so this is not flaky yet still catches a regression
    # back to O(n^2).
    rules = []
    total_len = 0
    i = 0
    while total_len < 550_000:
        rule = f".cls-{i} {{ color: #{i % 999:03x}; margin: {i % 40}px; }}\n"
        rules.append(rule)
        total_len += len(rule)
        i += 1
    text = "".join(rules)
    assert len(text) > 500_000

    start = time.perf_counter()
    result = classify_response(text)
    elapsed = time.perf_counter() - start

    assert result == "none"
    assert elapsed < 1.0


def test_large_payload_with_real_tool_call_near_end_is_native():
    # Guards against a pre-filter or cap that skips real calls in long
    # responses: the genuine tool call sits at the very end of a >500KB
    # brace-dense preamble, which is exactly the case a naive early-exit
    # cap (e.g. "only scan the first N bytes") would break.
    rules = []
    total_len = 0
    i = 0
    while total_len < 550_000:
        rule = f".cls-{i} {{ color: #{i % 999:03x}; margin: {i % 40}px; }}\n"
        rules.append(rule)
        total_len += len(rule)
        i += 1
    preamble = "".join(rules)
    text = preamble + (
        '{"name": "preview_demo", "arguments": {"html": "<p>done</p>"}}'
    )
    assert len(text) > 500_000
    assert classify_response(text) == "native"


def test_trailing_comma_objects_are_fast_and_none():
    # Round 3 finding: `_looks_like_json_object_start` only screens by SHAPE
    # (is the char after `{`+ws a quote or `}`?). A trailing-comma object
    # like `{"a": 1, "b": 2,}` passes that shape check -- `"` follows `{` --
    # but is not valid strict JSON, so every occurrence still falls through
    # to `raw_decode`, which raises `JSONDecodeError`. That exception's
    # `__init__` computes `doc.count("\n", 0, pos)`, an O(pos) scan of the
    # WHOLE document, so summed over every occurrence in a large document
    # this is O(n^2) again -- just moved from "fails the shape check" to
    # "passes the shape check but fails to decode". Measured on the
    # round-2 code with this exact repro: n=2000/127KB->0.039s,
    # 4000/254KB->0.148s, 8000/508KB->0.579s, 16000/1016KB->2.206s (us/char
    # roughly doubling each time n doubles = quadratic). None of these
    # fragments contain the literal substrings `"name"` or `"arguments"`
    # anywhere in the document, so a sound fix can skip `raw_decode`
    # entirely here.
    #
    # Bound: 0.5s has enormous headroom over what a linear scan should take
    # (low tens of milliseconds expected) while being over 4x below the
    # round-2 code's observed 2.2s at this size -- not flaky, but still
    # catches a regression back to quadratic behavior.
    frag = '{"a": 1, "b": 2,}\n'
    filler = "some prose text here padding out the document. "
    text = (frag + filler) * 16000
    assert len(text) > 1_000_000

    start = time.perf_counter()
    result = classify_response(text)
    elapsed = time.perf_counter() - start

    assert result == "none"
    assert elapsed < 0.5


def test_reversed_key_order_bare_json_is_native():
    # Soundness guard for a key-presence pre-filter: JSON objects are
    # unordered, so `{"arguments": {...}, "name": "..."}` is exactly as
    # valid a tool call as the more common `{"name": ..., "arguments": ...}`
    # ordering. A pre-filter that assumed "name" must appear before
    # "arguments" in the text would wrongly prune this position and turn a
    # real tool call into "none".
    text = '{"arguments": {"demo_id": "d1"}, "name": "save_demo"}'
    assert classify_response(text) == "native"


def test_truncated_base64_data_uri_is_none_and_does_not_raise():
    # Another everyday shape named in the round-3 finding: a candidate that
    # hits its output token limit mid-generation while emitting an embedded
    # image leaves an unterminated JSON string. `{"image": "data:..."` has
    # no closing quote or brace at all, so `raw_decode` must fail cleanly
    # (not raise out of `classify_response`) and the result must be "none",
    # not "native" -- an incomplete payload is not a genuine tool call.
    text = '{"image": "data:image/png;base64,' + "A" * 100_000
    assert classify_response(text) == "none"


def test_tool_call_tag_repetition_is_fast_and_none():
    # Round 4 finding: `_TOOL_CALL_TAG`'s regex
    # `<tool_call>\s*(\{.*?\})\s*</tool_call>` with re.DOTALL has its own
    # O(n^2) blowup, entirely independent of the bare-JSON scan hardened in
    # rounds 1-3 (that scan never runs here -- `classify_response` calls the
    # tag scan FIRST, unconditionally). When a `{` follows `<tool_call>` but
    # no `}` ever appears, the non-greedy `.*?` must scan to end-of-string
    # trying to complete the match, and it does this at every one of the n
    # `<tool_call>` occurrences in an input like `'<tool_call>{' * n`.
    # Measured on the pre-round-4 regex-based code with this exact repro:
    # n=2000/23KB->0.099s, 4000/47KB->0.401s, 8000/94KB->1.596s,
    # 16000/188KB->6.349s (~4x time per 2x size = quadratic). Control:
    # `'<tool_call>' * 8000` with NO `{` runs in 0.0004s -- confirms the `{`
    # (not the tag repetition itself) triggers the scan-to-end-of-string.
    #
    # Bound: 1.0s has enormous headroom over the low milliseconds a linear
    # str.find-based scan should take, while sitting more than 6x below the
    # pre-fix quadratic time at this size (6.349s) -- not flaky, but still
    # catches a regression back to O(n^2).
    text = "<tool_call>{" * 16000

    start = time.perf_counter()
    result = classify_response(text)
    elapsed = time.perf_counter() - start

    assert result == "none"
    assert elapsed < 1.0


def test_second_of_multiple_tool_call_pairs_is_native():
    # Guards the find-based open/close tag pairing's advance logic: after
    # the FIRST `<tool_call>...</tool_call>` pair is consumed (it parses as
    # JSON but lacks "name"/"arguments", so it isn't a tool call), the scan
    # must continue searching from just past that pair's closing tag and
    # find the second, genuinely valid pair -- not stop at the first pair,
    # and not re-scan from the document start.
    text = (
        '<tool_call>{"not": "a tool call"}</tool_call>\n'
        '<tool_call>{"name":"preview_demo","arguments":{"html":"<p>hi</p>"}}</tool_call>'
    )
    assert classify_response(text) == "native"


def test_unterminated_tool_call_tag_then_bare_json_call_is_native():
    # Guards against the find-based pairing loop consuming the rest of the
    # document on an unmatched open tag. `<tool_call>` here is never closed
    # anywhere in the text (no `</tool_call>` at all), so
    # `_iter_tool_call_payloads` must yield nothing and get out of the way --
    # the classifier still has to fall through to the bare-JSON scan and
    # find the genuine tool call written later in the same response.
    text = (
        "<tool_call>{oops, I forgot to close this tag\n"
        'Anyway, {"name": "preview_demo", "arguments": {"html": "<p>hi</p>"}}'
    )
    assert classify_response(text) == "native"
