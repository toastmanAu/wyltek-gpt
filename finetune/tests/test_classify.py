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
