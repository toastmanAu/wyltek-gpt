import io
import zipfile

from fastapi.testclient import TestClient

from backend import app as app_module

client = TestClient(app_module.app)


def _archive() -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_STORED) as archive:
        archive.writestr("demo.html", "<!doctype html><title>Demo</title>")
    return output.getvalue()


def test_ios_share_is_named_one_shot_zip():
    app_module._IOS_SHARE_FILES.clear()
    content = _archive()
    created = client.post(
        "/api/ios-share",
        content=content,
        headers={
            "Content-Type": "application/zip",
            "X-Wyltek-Filename": "HTMLocal%20Demo.zip",
        },
    )

    assert created.status_code == 200
    url = created.json()["url"]
    assert url.startswith("/api/ios-share/")
    assert not url.endswith(".zip")

    handoff = client.get(url)
    assert handoff.status_code == 200
    assert handoff.headers["content-type"].startswith("text/html")
    assert handoff.headers["cache-control"] == "no-store"
    assert "HTMLocal Demo.zip" in handoff.text
    download_url = f"{url}/HTMLocal%20Demo.zip"
    assert f'href="{download_url}"' in handoff.text

    downloaded = client.get(download_url)
    assert downloaded.status_code == 200
    assert downloaded.content == content
    assert downloaded.headers["content-type"] == "application/zip"
    assert "HTMLocal%20Demo.zip" in downloaded.headers["content-disposition"]
    assert downloaded.headers["cache-control"] == "no-store"
    assert client.get(download_url).status_code == 404
    assert client.get(url).status_code == 404


def test_ios_share_rejects_wrong_type_invalid_zip_and_unsafe_name():
    assert client.post("/api/ios-share", content=b"nope", headers={"Content-Type": "text/html"}).status_code == 415
    assert client.post("/api/ios-share", content=b"nope", headers={"Content-Type": "application/zip"}).status_code == 400

    created = client.post(
        "/api/ios-share",
        content=_archive(),
        headers={"Content-Type": "application/zip", "X-Wyltek-Filename": "..%2F..%2Fdemo.zip"},
    )
    url = created.json()["url"]
    handoff = client.get(url)
    assert "demo.zip" in handoff.text
    assert f'href="{url}/demo.zip"' in handoff.text
