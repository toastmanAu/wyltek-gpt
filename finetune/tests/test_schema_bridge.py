import json
from pathlib import Path

SCHEMA_PATH = Path(__file__).parent.parent / "bakeoff" / "schemas" / "html_demo_tools.json"


def test_schema_file_exists():
    assert SCHEMA_PATH.is_file(), f"missing {SCHEMA_PATH}; run dump_schemas.py"


def test_schema_has_three_html_demo_tools():
    tools = json.loads(SCHEMA_PATH.read_text())
    names = [t["function"]["name"] for t in tools]
    assert names == ["preview_demo", "patch_demo", "save_demo"]


def test_schemas_are_openai_shaped():
    tools = json.loads(SCHEMA_PATH.read_text())
    for t in tools:
        assert t["type"] == "function"
        fn = t["function"]
        assert isinstance(fn["name"], str) and fn["name"]
        assert isinstance(fn["description"], str) and fn["description"]
        assert fn["parameters"]["type"] == "object"
        assert isinstance(fn["parameters"]["required"], list)


def test_preview_demo_requires_html():
    tools = json.loads(SCHEMA_PATH.read_text())
    preview = next(t for t in tools if t["function"]["name"] == "preview_demo")
    assert "html" in preview["function"]["parameters"]["required"]
