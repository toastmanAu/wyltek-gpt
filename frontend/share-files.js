const encoder = new TextEncoder();

const crcTable = Uint32Array.from({ length: 256 }, (_, value) => {
  let crc = value;
  for (let bit = 0; bit < 8; bit += 1) {
    crc = (crc >>> 1) ^ (crc & 1 ? 0xedb88320 : 0);
  }
  return crc >>> 0;
});

function crc32(bytes) {
  let crc = 0xffffffff;
  for (const byte of bytes) crc = (crc >>> 8) ^ crcTable[(crc ^ byte) & 0xff];
  return (crc ^ 0xffffffff) >>> 0;
}

function safeEntryName(name) {
  const base = String(name || "").split(/[\\/]/).pop().replaceAll("\0", "").slice(0, 240);
  return /\.(?:html?|xhtml)$/i.test(base) ? base : "index.html";
}

export function createStoredZip(entryName, input) {
  const content = input instanceof Uint8Array ? input : new Uint8Array(input);
  if (content.byteLength > 0xffffffff) throw new RangeError("File is too large for a ZIP share fallback");

  const name = encoder.encode(safeEntryName(entryName));
  const localSize = 30 + name.byteLength + content.byteLength;
  const centralSize = 46 + name.byteLength;
  const output = new Uint8Array(localSize + centralSize + 22);
  const view = new DataView(output.buffer);
  const checksum = crc32(content);

  view.setUint32(0, 0x04034b50, true);
  view.setUint16(4, 20, true);
  view.setUint16(6, 0x0800, true);
  view.setUint16(8, 0, true);
  view.setUint16(10, 0, true);
  view.setUint16(12, 0x0021, true);
  view.setUint32(14, checksum, true);
  view.setUint32(18, content.byteLength, true);
  view.setUint32(22, content.byteLength, true);
  view.setUint16(26, name.byteLength, true);
  view.setUint16(28, 0, true);
  output.set(name, 30);
  output.set(content, 30 + name.byteLength);

  const central = localSize;
  view.setUint32(central, 0x02014b50, true);
  view.setUint16(central + 4, 20, true);
  view.setUint16(central + 6, 20, true);
  view.setUint16(central + 8, 0x0800, true);
  view.setUint16(central + 10, 0, true);
  view.setUint16(central + 12, 0, true);
  view.setUint16(central + 14, 0x0021, true);
  view.setUint32(central + 16, checksum, true);
  view.setUint32(central + 20, content.byteLength, true);
  view.setUint32(central + 24, content.byteLength, true);
  view.setUint16(central + 28, name.byteLength, true);
  view.setUint16(central + 30, 0, true);
  view.setUint16(central + 32, 0, true);
  view.setUint16(central + 34, 0, true);
  view.setUint16(central + 36, 0, true);
  view.setUint32(central + 38, 0, true);
  view.setUint32(central + 42, 0, true);
  output.set(name, central + 46);

  const end = central + centralSize;
  view.setUint32(end, 0x06054b50, true);
  view.setUint16(end + 4, 0, true);
  view.setUint16(end + 6, 0, true);
  view.setUint16(end + 8, 1, true);
  view.setUint16(end + 10, 1, true);
  view.setUint32(end + 12, centralSize, true);
  view.setUint32(end + 16, central, true);
  view.setUint16(end + 20, 0, true);
  return output;
}

export async function iosShareFallbackFile(file) {
  if (!(file instanceof File) || !/\.(?:html?|xhtml)$/i.test(file.name)) return null;
  const entryName = safeEntryName(file.name);
  const stem = entryName.replace(/\.(?:html?|xhtml)$/i, "") || "page";
  const archive = createStoredZip(entryName, new Uint8Array(await file.arrayBuffer()));
  return new File([archive], `${stem}.zip`, {
    type: "application/zip",
    lastModified: file.lastModified,
  });
}

export async function publishIOSShareFallback(file, fetchImpl = fetch) {
  if (!(file instanceof File) || file.type !== "application/zip") {
    throw new TypeError("iOS share fallback must be an application/zip File");
  }
  const response = await fetchImpl("/api/ios-share", {
    method: "POST",
    headers: {
      "Content-Type": "application/zip",
      "X-Wyltek-Filename": encodeURIComponent(file.name),
    },
    body: file,
  });
  if (!response.ok) throw new Error(`iOS share fallback upload returned HTTP ${response.status}`);
  const payload = await response.json();
  if (typeof payload.url !== "string" || !payload.url.startsWith("/api/ios-share/")) {
    throw new Error("iOS share fallback returned an invalid URL");
  }
  return payload.url;
}
