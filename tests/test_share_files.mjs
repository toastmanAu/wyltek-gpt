import assert from "node:assert/strict";
import test from "node:test";
import {
  createStoredZip,
  iosShareFallbackFile,
  publishIOSShareFallback,
} from "../frontend/share-files.js";

test("creates a valid single-file stored ZIP", () => {
  const content = new TextEncoder().encode("hello");
  const archive = createStoredZip("demo.html", content);
  const view = new DataView(archive.buffer);
  const nameLength = view.getUint16(26, true);
  const name = new TextDecoder().decode(archive.slice(30, 30 + nameLength));
  const payload = archive.slice(30 + nameLength, 30 + nameLength + content.length);

  assert.equal(view.getUint32(0, true), 0x04034b50);
  assert.equal(view.getUint32(14, true), 0x3610a686);
  assert.equal(name, "demo.html");
  assert.deepEqual(payload, content);
  assert.equal(view.getUint32(archive.length - 22, true), 0x06054b50);
});

test("turns HTML into a named ZIP fallback", async () => {
  const source = new File(["<!doctype html><title>Demo</title>"], "demo.html", {
    type: "text/html;charset=utf-8",
    lastModified: 123,
  });
  const fallback = await iosShareFallbackFile(source);

  assert.equal(fallback.name, "demo.zip");
  assert.equal(fallback.type, "application/zip");
  assert.equal(fallback.lastModified, 123);
  assert.equal(await iosShareFallbackFile(new File(["text"], "note.txt")), null);
});

test("publishes a named ZIP fallback through the one-shot endpoint", async () => {
  const file = new File([createStoredZip("demo.html", new TextEncoder().encode("demo"))], "demo.zip", {
    type: "application/zip",
  });
  let request;
  const url = await publishIOSShareFallback(file, async (...args) => {
    request = args;
    return { ok: true, json: async () => ({ url: "/api/ios-share/token" }) };
  });

  assert.equal(url, "/api/ios-share/token");
  assert.equal(request[0], "/api/ios-share");
  assert.equal(request[1].method, "POST");
  assert.equal(request[1].headers["Content-Type"], "application/zip");
  assert.equal(request[1].headers["X-Wyltek-Filename"], "demo.zip");
  assert.equal(request[1].body, file);
});
