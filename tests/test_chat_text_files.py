"""Text-file passthrough injection + generation-budget regression tests.

Plain .txt/.md/.json uploads have no converter, so before this feature their
bytes never reached the model — it only ever saw the filename. These tests
pin the two fixes: (1) named text files are read from the session workspace
and appended to the last user message; (2) the default chat num_ctx is large
enough that a heavy single-shot generation (e.g. an embedded BIP39 wordlist)
plus reasoning tokens does not truncate.
"""

from fastapi.testclient import TestClient

from backend import app as app_module

client = TestClient(app_module.app)


def _seed(tmp_path, monkeypatch, name, content, session="default"):
    """Point WORKSPACE at tmp_path and drop a file in the session dir."""
    monkeypatch.setattr(app_module, "WORKSPACE", tmp_path)
    sess = tmp_path / session
    sess.mkdir(parents=True, exist_ok=True)
    f = sess / name
    f.write_text(content)
    return f


# ── unit: the injection helper ───────────────────────────────────────


def test_inject_text_files_appends_to_last_user_message(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, "ref.txt", "REFERENCE_BODY_TOKEN")
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "use the file"},
    ]
    n = app_module._inject_text_files(messages, "default", ["ref.txt"])
    assert n == 1
    assert "REFERENCE_BODY_TOKEN" in messages[-1]["content"]
    assert messages[-1]["content"].startswith("use the file")  # original kept
    assert "ref.txt" in messages[-1]["content"]                 # labelled


def test_inject_text_files_skips_non_text_and_missing(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, "pic.png", "not really an image")
    messages = [{"role": "user", "content": "hi"}]
    n = app_module._inject_text_files(
        messages, "default", ["pic.png", "ghost.txt"]
    )
    assert n == 0
    assert messages[-1]["content"] == "hi"  # untouched


def test_inject_text_files_truncates_oversized(tmp_path, monkeypatch):
    cap = app_module._MAX_TEXT_INJECT_CHARS
    _seed(tmp_path, monkeypatch, "big.md", "A" * (cap + 5000))
    messages = [{"role": "user", "content": "summarise"}]
    n = app_module._inject_text_files(messages, "default", ["big.md"])
    assert n == 1
    body = messages[-1]["content"]
    assert body.count("A") <= cap        # content clamped to the cap
    assert "truncated" in body.lower()   # user/model told it was clipped


def test_inject_text_files_rejects_path_traversal(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, "ok.txt", "ok")
    messages = [{"role": "user", "content": "hi"}]
    n = app_module._inject_text_files(
        messages, "default", ["../../etc/passwd"]
    )
    assert n == 0
    assert messages[-1]["content"] == "hi"


# ── integration: the /api/chat endpoint ──────────────────────────────


def test_chat_endpoint_injects_text_file_and_uses_big_ctx(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, "spec.txt", "ZZTOP_SPEC_MARKER")

    captured = {}

    def fake_stream_one(body):
        captured.setdefault("body", body)

        async def gen(_b):
            yield ("chunk", "ok")
            yield ("done", None)

        return gen(body)

    monkeypatch.setattr(app_module, "_stream_one", fake_stream_one, raising=False)

    resp = client.post(
        "/api/chat",
        json={
            "model": "m",
            "messages": [{"role": "user", "content": "build from spec"}],
            "text_files": ["spec.txt"],
            "session_id": "default",
        },
    )
    assert resp.status_code == 200
    _ = resp.text  # drain the streaming response so the generator runs

    body = captured["body"]
    last_user = [m for m in body["messages"] if m["role"] == "user"][-1]
    assert "ZZTOP_SPEC_MARKER" in last_user["content"]
    assert body["options"]["num_ctx"] >= 24576


def test_chat_default_num_ctx_is_bumped():
    # Guards the budget fix: heavy single-shot generations need headroom.
    assert app_module._CHAT_DEFAULT_NUM_CTX >= 24576
