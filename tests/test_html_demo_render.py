import pytest
from backend.skills.html_demo import render


pytestmark = pytest.mark.skipif(not render.available(), reason="Playwright/Chromium not installed")


def test_clean_page_loads_no_errors():
    html = "<!doctype html><html><body><h1 id='t'>hi</h1></body></html>"
    r = render.render(html, settle_ms=100)
    assert r["loaded"] is True
    assert r["console_errors"] == []
    assert r["exceptions"] == []
    assert r["render_sanity"]["text_len"] >= 2
    assert r["_screenshot_b64"] is None  # not requested


def test_js_exception_is_captured():
    html = "<!doctype html><html><body><script>throw new Error('BOOM')</script></body></html>"
    r = render.render(html, settle_ms=100)
    assert r["loaded"] is True
    assert any("BOOM" in e for e in r["exceptions"])


def test_raf_detected():
    html = ("<!doctype html><html><body><canvas></canvas>"
            "<script>requestAnimationFrame(()=>{})</script></body></html>")
    r = render.render(html, settle_ms=150)
    assert r["raf_ran"] is True
    assert r["render_sanity"]["has_canvas"] is True


def test_screenshot_returned_when_requested():
    html = "<!doctype html><html><body style='background:#0af'>x</body></html>"
    r = render.render(html, settle_ms=100, want_screenshot=True)
    assert isinstance(r["_screenshot_b64"], str) and len(r["_screenshot_b64"]) > 100
