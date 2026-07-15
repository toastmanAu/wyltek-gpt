import pytest
from backend.skills import html_demo


@pytest.fixture(autouse=True)
def _store(tmp_path):
    html_demo.init_store(tmp_path)
    yield


def test_tool_schemas_shape():
    names = {t["function"]["name"] for t in html_demo.tool_schemas()}
    assert names == {"preview_demo", "patch_demo", "save_demo"}
    assert names == set(html_demo.HTML_DEMO_TOOL_NAMES)


def test_guidance_is_lean_text():
    g = html_demo.guidance()
    assert "self-contained" in g.lower()
    assert len(g) < 6000  # lean: shares the loop's context budget


def test_examples_present():
    ex = html_demo.examples()
    assert ex and all("html" in e and "name" in e for e in ex)


def test_dispatch_patch_unknown_id_errors():
    r = html_demo.dispatch("patch_demo", {"demo_id": "nope", "edits": []})
    assert r["tool_error"] is True


def test_dispatch_save_roundtrip(tmp_path):
    # create via store directly, then save through dispatch
    did = html_demo.store().create("<html>S</html>", "demo")
    r = html_demo.dispatch("save_demo", {"demo_id": did})
    assert r["saved"] is True
    assert r["preview_url"].startswith("/demos/")


def test_dispatch_unknown_tool_errors():
    r = html_demo.dispatch("nonsense", {})
    assert r["tool_error"] is True


@pytest.mark.skipif(not html_demo.available(), reason="Playwright/Chromium not installed")
def test_dispatch_preview_creates_and_renders():
    r = html_demo.dispatch("preview_demo", {"html": "<h1>hi</h1>", "settle_ms": 100})
    assert r["loaded"] is True
    assert isinstance(r["demo_id"], str)
    # screenshot suppressed unless caller opts in
    assert r.get("_screenshot_b64") is None


def test_dispatch_bad_settle_ms_does_not_raise():
    did = html_demo.store().create("<html>S</html>", "demo")
    r = html_demo.dispatch("save_demo", {"demo_id": did, "settle_ms": "not-a-number"})
    assert r.get("saved") is True  # save_demo ignores settle_ms; must not raise


def test_dispatch_non_string_name_does_not_raise():
    did = html_demo.store().create("<html>S</html>", "demo")
    r = html_demo.dispatch("save_demo", {"demo_id": did, "name": 12345})
    assert r.get("saved") is True
    assert r["preview_url"].startswith("/demos/")
