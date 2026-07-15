from backend import app as app_module


def test_is_html_demo_intent():
    assert app_module._is_html_demo_intent("make a self-contained html demo of pong")
    assert app_module._is_html_demo_intent("build me a canvas game")
    assert app_module._is_html_demo_intent("a physics demo with bouncing balls")
    assert not app_module._is_html_demo_intent("what OS are you on?")


def test_partition_tool_calls_three_way():
    calls = [
        {"function": {"name": "cellc_check", "arguments": {}}},
        {"function": {"name": "preview_demo", "arguments": {}}},
        {"function": {"name": "translate", "arguments": {}}},
    ]
    cellc, demo, op = app_module._partition_tool_calls(calls)
    assert [c["function"]["name"] for c in cellc] == ["cellc_check"]
    assert [c["function"]["name"] for c in demo] == ["preview_demo"]
    assert [c["function"]["name"] for c in op] == ["translate"]


def test_summarize_demo_step():
    ok = app_module._summarize_demo_step("preview_demo",
        {"loaded": True, "console_errors": [], "exceptions": [], "raf_ran": True, "demo_id": "demo1"})
    assert "demo1" in ok and "✓" in ok
    bad = app_module._summarize_demo_step("preview_demo",
        {"loaded": True, "console_errors": ["error: x"], "exceptions": ["Boom"], "raf_ran": False, "demo_id": "demo1"})
    assert "1 err" in bad or "error" in bad.lower()
    saved = app_module._summarize_demo_step("save_demo", {"saved": True, "slug": "pong"})
    assert "pong" in saved


def test_inject_html_demo_context_on_intent(monkeypatch):
    monkeypatch.setattr(app_module.html_demo, "available", lambda: True)
    monkeypatch.setattr(app_module.html_demo, "guidance", lambda: "GUIDANCE_TOKEN")
    monkeypatch.setattr(app_module.html_demo, "examples",
                        lambda: [{"name": "x", "html": "EXAMPLE_TOKEN"}])
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "make an html demo"}]
    hit = app_module._inject_html_demo_context(messages, "make an html demo")
    assert hit is True
    assert "GUIDANCE_TOKEN" in messages[0]["content"]
    assert "EXAMPLE_TOKEN" in messages[0]["content"]


def test_inject_html_demo_context_skips_non_intent(monkeypatch):
    monkeypatch.setattr(app_module.html_demo, "available", lambda: True)
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hello"}]
    assert app_module._inject_html_demo_context(messages, "hello") is False
    assert messages[0]["content"] == "sys"
