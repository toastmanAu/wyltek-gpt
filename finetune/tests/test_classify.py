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
