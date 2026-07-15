from pathlib import Path
import pytest
from backend.skills.html_demo.store import DemoStore, _slugify


def test_create_get_roundtrip(tmp_path):
    s = DemoStore(tmp_path)
    did = s.create("<html>A</html>", "My Demo")
    assert isinstance(did, str) and did
    rec = s.get(did)
    assert rec["html"] == "<html>A</html>"
    assert rec["name"] == "My Demo"


def test_replace_updates_html(tmp_path):
    s = DemoStore(tmp_path)
    did = s.create("<html>A</html>")
    assert s.replace(did, "<html>B</html>") is True
    assert s.get(did)["html"] == "<html>B</html>"
    assert s.replace("nope", "x") is False


def test_apply_patch_unique_find(tmp_path):
    s = DemoStore(tmp_path)
    did = s.create("<h1>hello</h1><p>world</p>")
    ok, reason = s.apply_patch(did, [{"find": "world", "replace": "there"}])
    assert ok is True and reason == "ok"
    assert s.get(did)["html"] == "<h1>hello</h1><p>there</p>"


def test_apply_patch_missing_find_fails(tmp_path):
    s = DemoStore(tmp_path)
    did = s.create("<h1>hello</h1>")
    ok, reason = s.apply_patch(did, [{"find": "absent", "replace": "x"}])
    assert ok is False
    assert "not found" in reason


def test_apply_patch_ambiguous_find_fails(tmp_path):
    s = DemoStore(tmp_path)
    did = s.create("<b>x</b><b>x</b>")
    ok, reason = s.apply_patch(did, [{"find": "<b>x</b>", "replace": "y"}])
    assert ok is False
    assert "unique" in reason


def test_save_writes_file_and_urls(tmp_path):
    s = DemoStore(tmp_path)
    did = s.create("<html>Z</html>", "Cool Thing!")
    res = s.save(did)
    assert Path(res["path"]).read_text() == "<html>Z</html>"
    assert res["slug"] == "cool-thing"
    assert res["preview_url"] == "/demos/cool-thing.html"


def test_save_collision_gets_suffix(tmp_path):
    s = DemoStore(tmp_path)
    a = s.create("<html>ONE</html>", "dupe")
    b = s.create("<html>TWO</html>", "dupe")
    ra = s.save(a)
    rb = s.save(b)
    assert ra["preview_url"] != rb["preview_url"]
    assert Path(ra["path"]).read_text() == "<html>ONE</html>"
    assert Path(rb["path"]).read_text() == "<html>TWO</html>"


def test_slugify():
    assert _slugify("Hello World") == "hello-world"
    assert _slugify("  ") == "demo"
    assert _slugify("A/B:C") == "a-b-c"


def test_save_same_demo_overwrites_not_suffixes(tmp_path):
    s = DemoStore(tmp_path)
    did = s.create("<html>V1</html>", "widget")
    r1 = s.save(did)
    s.replace(did, "<html>V2</html>")
    r2 = s.save(did)
    assert r1["path"] == r2["path"]                      # same file, not widget-1.html
    assert Path(r2["path"]).read_text() == "<html>V2</html>"
    assert list(tmp_path.glob("*.html")) == [Path(r2["path"])]  # exactly one file on disk


def test_apply_patch_non_dict_edit_fails_gracefully(tmp_path):
    s = DemoStore(tmp_path)
    did = s.create("<h1>hi</h1>")
    ok, reason = s.apply_patch(did, ["not a dict"])
    assert ok is False and "object" in reason


def test_slugify_non_string_returns_demo():
    assert _slugify(12345) == "demo"
    assert _slugify(None) == "demo"
    assert _slugify(["x"]) == "demo"
