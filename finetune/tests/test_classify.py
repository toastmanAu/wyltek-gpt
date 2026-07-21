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
