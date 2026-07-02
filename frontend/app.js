const $ = (sel) => document.querySelector(sel);
const messagesEl = $("#messages");
const modelSel = $("#model");
const themeSel = $("#settings-theme");  // moved into settings drawer
const fileInput = $("#file");
const tray = $("#tray");
const trayName = $("#tray-name");
const trayTarget = $("#tray-target");
const trayConvert = $("#tray-convert");
const trayClear = $("#tray-clear");
const input = $("#input");
const composer = $("#composer");
const sendBtn = $("#send");
const statsBar = $("#stats-bar");

const STORAGE = {
  theme: "lcb.theme",
  model: "lcb.model",
  translateSrc: "lcb.translate.src",
  translateTgt: "lcb.translate.tgt",
};
const SESSION = "default";  // multi-session support arrives in v1.2

// Subset of TranslateGemma's 55-language coverage, ordered to surface
// common pairs first. Codes are ISO 639-1; regional variants (e.g. pt-BR,
// zh-Hans) are accepted by the backend regex if you need them — extend
// this list to expose them in the dropdown.
const TRANSLATE_LANGS = [
  ["en", "English"], ["es", "Spanish"], ["fr", "French"], ["de", "German"],
  ["it", "Italian"], ["pt", "Portuguese"], ["nl", "Dutch"], ["pl", "Polish"],
  ["ru", "Russian"], ["uk", "Ukrainian"],
  ["zh", "Chinese"], ["ja", "Japanese"], ["ko", "Korean"],
  ["ar", "Arabic"], ["hi", "Hindi"], ["tr", "Turkish"], ["vi", "Vietnamese"],
  ["id", "Indonesian"], ["th", "Thai"], ["he", "Hebrew"],
  ["sv", "Swedish"], ["nb", "Norwegian"], ["da", "Danish"], ["fi", "Finnish"],
  ["cs", "Czech"], ["hu", "Hungarian"], ["ro", "Romanian"], ["el", "Greek"],
];

// Match only the canonical TranslateGemma base GGUFs (uploaded as
// hf.co/<user>/translategemma-<size>-it-GGUF:<quant>). The per-pair
// Modelfiles like `translategemma-4b-en-es` already bake the T3
// envelope into their TEMPLATE; routing them through /api/operations/
// translate would double-wrap the envelope and surface raw JSON in
// the chat bubble. Keep them in the standard chat path instead.
const isTranslateModel = (name) =>
  /^hf\.co\/[^\/]+\/translategemma-.+-it-GGUF:/i.test(name || "");

// Spinner registry — frame sets borrowed from indicatif (Rust) and gum (Go).
// Each entry's interval is tuned for that specific frame set; don't share it.
const SPINNERS = {
  braille: { frames: ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"], interval: 80 },
  dots:    { frames: ["⣾","⣽","⣻","⢿","⡿","⣟","⣯","⣷"], interval: 80 },
  line:    { frames: ["|","/","-","\\"], interval: 130 },
  mini:    { frames: ["⠂","⠃","⠁","⠉","⠈","⠘","⠐","⠰"], interval: 100 },
  bar:     { frames: ["▏","▎","▍","▌","▋","▊","▉","▊","▋","▌","▍","▎"], interval: 80 },
  arrow:   { frames: ["←","↖","↑","↗","→","↘","↓","↙"], interval: 100 },
  bounce:  { frames: ["⠁","⠂","⠄","⠂"], interval: 120 },
  pulse:   { frames: ["•","○","●","○"], interval: 200 },
  moon:    { frames: ["🌑","🌒","🌓","🌔","🌕","🌖","🌗","🌘"], interval: 150 },
  clock:   { frames: ["🕛","🕐","🕑","🕒","🕓","🕔","🕕","🕖","🕗","🕘","🕙","🕚"], interval: 100 },
};

const SETTINGS_KEY = "lcb.settings";
const DEFAULT_SETTINGS = { spinner: "braille", collapseReasoning: false };
let settings = { ...DEFAULT_SETTINGS, ...safeParse(localStorage.getItem(SETTINGS_KEY)) };

function safeParse(s) {
  try { return JSON.parse(s) || {}; } catch { return {}; }
}
function saveSettings() {
  localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
}

function startSpinner(targetEl, label) {
  const cfg = SPINNERS[settings.spinner] || SPINNERS.braille;
  let i = 0;
  targetEl.textContent = ` ${cfg.frames[0]} ${label}`;
  const id = setInterval(() => {
    i = (i + 1) % cfg.frames.length;
    targetEl.textContent = ` ${cfg.frames[i]} ${label}`;
  }, cfg.interval);
  return () => clearInterval(id);
}

let history = [];
let streaming = false;
let converters = { enabled: [], disabled: [] };
let operations = { enabled: [], disabled: [], bridges_available: [], output_dir: null };
let capabilities = {};  // model_name -> {tool_calling, vision, audio, reasoning, ...}
let probing = false;    // true while a /api/capabilities/probe is in flight
let pending = null;  // {name, size}
let pendingOpCard = null;  // active op-card awaiting y/n/e
let autoRouter = { captioner_model: "" };  // populated from /api/config

// Caption-intent regexes — fire the auto-router when ALL of:
//   1. a pending file is an image
//   2. one of these matches the user's prompt
//   3. captioner_model is configured AND present in the model list
const CAPTION_INTENT = [
  /\bdescribe (this|the|that|it)\b/i,
  /\bwhat'?s in (this|the|that)\b/i,
  /\bwhat is in (this|the|that)\b/i,
  /\bwhat does (this|the|that) (image|picture|photo|pic)\b/i,
  /\b(caption|summari[sz]e) (this|that|it|the|the image|the picture)\b/i,
  /\btell me about (this|the|that) (image|picture|photo)\b/i,
  /\bwhat do you see\b/i,
  /\bidentify (this|the|that|what)\b/i,
];

const IMAGE_EXT_RE = /\.(png|jpg|jpeg|webp|gif|bmp|tiff?)$/i;

function isImageFile(name) {
  return !!name && IMAGE_EXT_RE.test(name);
}

function shouldAutoCaption(prompt) {
  if (!pending || !isImageFile(pending.name)) return false;
  if (!autoRouter.captioner_model) return false;
  // If the user has the captioner itself selected, no point auto-routing.
  if (modelSel.value === autoRouter.captioner_model) return false;
  return CAPTION_INTENT.some((re) => re.test(prompt));
}

// Glyph language for capability display. Single emoji each, mobile-safe.
// Any change here should also update the README capability matrix.
const CAP_GLYPHS = {
  tool_calling_native: "⚒",   // model emits valid tool_calls
  tool_calling_ignored: "⚙",  // model accepts tools= but ignores them
  vision: "👁",
  audio: "🎙",
  reasoning: "🧠",
  unknown: "❓",               // not yet probed
};

function glyphsForModel(name) {
  const caps = capabilities[name];
  if (!caps) return CAP_GLYPHS.unknown;
  const out = [];
  if (caps.tool_calling === "native") out.push(CAP_GLYPHS.tool_calling_native);
  else if (caps.tool_calling === "ignored") out.push(CAP_GLYPHS.tool_calling_ignored);
  if (caps.vision) out.push(CAP_GLYPHS.vision);
  if (caps.audio) out.push(CAP_GLYPHS.audio);
  if (caps.reasoning) out.push(CAP_GLYPHS.reasoning);
  return out.join("") || "·";  // dot if probed but no positive capabilities
}

function tooltipForModel(name) {
  const caps = capabilities[name];
  if (!caps) return "Capabilities not yet probed — will probe on first use";
  const parts = [];
  parts.push(`tool calling: ${caps.tool_calling}`);
  parts.push(`vision: ${caps.vision ? "yes" : "no"}`);
  parts.push(`audio: ${caps.audio ? "yes" : "no"}`);
  parts.push(`reasoning: ${caps.reasoning ? "yes" : "no"}`);
  if (caps.last_probed) parts.push(`last probed: ${caps.last_probed}`);
  return parts.join("\n");
}

function makeOption(value, label = value) {
  const opt = document.createElement("option");
  opt.value = value;
  opt.textContent = label;
  return opt;
}

// Bootstrap: every optional fetch is wrapped so a failure can't take
// down the critical path (model dropdown). Themes, converters, and
// operations are nice-to-have at startup; chat-with-a-model is the
// product. Always load model dropdown FIRST.
async function safeJSON(url, fallback) {
  try {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return await r.json();
  } catch (e) {
    appendMsg("error", `${url} failed: ${e.message}`);
    return fallback;
  }
}

async function bootstrap() {
  // Capabilities cache loaded BEFORE the dropdown so glyphs are available
  // for the first render. If it 404s/errs, glyphs gracefully show ❓.
  capabilities = await safeJSON("/api/capabilities", {});

  // 1. Critical: model dropdown. Load this first so a backend hiccup
  // elsewhere can't leave it spinning forever.
  modelSel.replaceChildren();
  try {
    const r = await fetch("/api/models");
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const models = await r.json();
    if (!models.length) {
      appendMsg("error", "Ollama is up but no models are installed. Run `ollama pull <name>`.");
    } else {
      modelSel.replaceChildren(...models.map((m) => makeModelOption(m)));
      const savedModel = localStorage.getItem(STORAGE.model);
      if (savedModel && models.includes(savedModel)) modelSel.value = savedModel;
    }
  } catch (e) {
    appendMsg("error", `Could not list models: ${e.message}`);
    modelSel.replaceChildren(makeOption("", "(no models)"));
  }

  // 2. Optional: config + themes.
  const cfg = await safeJSON("/api/config", { system_prompt: "", default_theme: "tokyo-night", auto_router: {} });
  history = [{ role: "system", content: cfg.system_prompt }];
  autoRouter = cfg.auto_router || { captioner_model: "" };

  const themes = await safeJSON("/api/themes", []);
  if (themes.length) {
    themeSel.replaceChildren(...themes.map((t) => makeOption(t)));
    const savedTheme = localStorage.getItem(STORAGE.theme) || cfg.default_theme;
    setTheme(themes.includes(savedTheme) ? savedTheme : themes[0]);
    themeSel.value = document.documentElement.dataset.theme;
  }

  // 3. Optional: converters + operations.
  converters = await safeJSON("/api/converters", { enabled: [], disabled: [] });
  if (converters.disabled.length) {
    appendMsg(
      "system",
      `${converters.enabled.length} converters enabled, ${converters.disabled.length} skipped (missing: ${converters.disabled.map((d) => d.missing).join(", ")})`,
    );
  } else if (converters.enabled.length) {
    appendMsg("system", `${converters.enabled.length} converters ready`);
  }

  operations = await safeJSON("/api/operations", { enabled: [], disabled: [], bridges_available: [], output_dir: null });
  if (operations.enabled.length) {
    const bridges = operations.bridges_available.length
      ? ` (bridges: ${operations.bridges_available.join(", ")})`
      : "";
    appendMsg("system", `${operations.enabled.length} operations available${bridges}`);
  }
  if (operations.disabled.length) {
    appendMsg(
      "system",
      `${operations.disabled.length} operation(s) disabled: ${operations.disabled.map((d) => `${d.id} (${d.reason})`).join("; ")}`,
    );
  }
  if (operations.output_dir) {
    appendMsg("system", `output mirror: ${operations.output_dir}`);
  }

  // Restored or default-selected model may need its first probe.
  if (modelSel.value && !capabilities[modelSel.value]) {
    probeModelIfNeeded(modelSel.value);  // fire-and-forget, doesn't block bootstrap
  }

  // Set initial composer mode based on the restored model. Must run
  // after populateTranslateLangs() so the dropdowns have options when
  // the placeholder is composed.
  populateTranslateLangs();
  updateComposerMode();
}

function setTheme(name) {
  document.documentElement.setAttribute("data-theme", name);
  document.getElementById("theme-link").href = `/static/themes/${name}.css`;
  localStorage.setItem(STORAGE.theme, name);
}

themeSel.addEventListener("change", () => setTheme(themeSel.value));

function makeModelOption(name) {
  const opt = document.createElement("option");
  opt.value = name;
  opt.textContent = `${name} ${glyphsForModel(name)}`;
  opt.title = tooltipForModel(name);
  return opt;
}

function refreshModelDropdown() {
  const current = modelSel.value;
  const opts = [...modelSel.options].map((o) => o.value);
  modelSel.replaceChildren(...opts.map((m) => makeModelOption(m)));
  if (current && opts.includes(current)) modelSel.value = current;
}

async function probeModelIfNeeded(model) {
  if (!model || capabilities[model] || probing) return;
  probing = true;
  const status = appendMsg("system", "");
  const stopSpin = startSpinner(status, `first-run capability probe: ${model} (~30-180s)...`);
  try {
    const r = await fetch("/api/capabilities/probe", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model }),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const caps = await r.json();
    capabilities[model] = caps;
    refreshModelDropdown();
    stopSpin();
    status.parentElement.remove();
    const summary = [];
    if (caps.tool_calling === "native") summary.push(`tools ${CAP_GLYPHS.tool_calling_native}`);
    else if (caps.tool_calling === "ignored") summary.push(`tools accepted but ignored ${CAP_GLYPHS.tool_calling_ignored}`);
    else if (caps.tool_calling === "rejected") summary.push("tools rejected");
    else summary.push(`tools ${caps.tool_calling}`);
    if (caps.vision) summary.push(`vision ${CAP_GLYPHS.vision}`);
    if (caps.reasoning) summary.push(`reasoning ${CAP_GLYPHS.reasoning}`);
    appendMsg("system", `${model}: ${summary.join(" · ")}`);
  } catch (e) {
    stopSpin();
    status.textContent = ` capability probe failed: ${e.message}`;
    status.parentElement.classList.add("error");
  } finally {
    probing = false;
  }
}

modelSel.addEventListener("change", async () => {
  localStorage.setItem(STORAGE.model, modelSel.value);
  updateComposerMode();
  await probeModelIfNeeded(modelSel.value);
});

// ─── translate composer mode ───────────────────────────────────────
// Composer swaps between chat-input and translate-input based on the
// selected model. Translate mode shows src/tgt selects above the
// textarea and routes submit to /api/operations/translate, which
// builds the TranslateGemma T3 envelope server-side. Chat mode is the
// default for everything else.

const translateControls = $("#translate-controls");
const translateSrcSel = $("#translate-src");
const translateTgtSel = $("#translate-tgt");
const translateSwapBtn = $("#translate-swap");

function populateTranslateLangs() {
  const buildOptions = () =>
    TRANSLATE_LANGS.map(([code, label]) => {
      const o = document.createElement("option");
      o.value = code;
      o.textContent = `${label} (${code})`;
      return o;
    });
  translateSrcSel.replaceChildren(...buildOptions());
  translateTgtSel.replaceChildren(...buildOptions());
  translateSrcSel.value = localStorage.getItem(STORAGE.translateSrc) || "en";
  translateTgtSel.value = localStorage.getItem(STORAGE.translateTgt) || "es";

  translateSrcSel.addEventListener("change", () => {
    localStorage.setItem(STORAGE.translateSrc, translateSrcSel.value);
    updateComposerMode();  // refresh placeholder
  });
  translateTgtSel.addEventListener("change", () => {
    localStorage.setItem(STORAGE.translateTgt, translateTgtSel.value);
    updateComposerMode();
  });
  translateSwapBtn.addEventListener("click", () => {
    const s = translateSrcSel.value;
    translateSrcSel.value = translateTgtSel.value;
    translateTgtSel.value = s;
    localStorage.setItem(STORAGE.translateSrc, translateSrcSel.value);
    localStorage.setItem(STORAGE.translateTgt, translateTgtSel.value);
    updateComposerMode();
  });
}

function updateComposerMode() {
  const translateMode = isTranslateModel(modelSel.value);
  translateControls.classList.toggle("hidden", !translateMode);
  if (translateMode) {
    input.placeholder = `Text to translate (${translateSrcSel.value} → ${translateTgtSel.value})...`;
  } else {
    input.placeholder = "Ask local inference...";
  }
}

// ─── messages ──────────────────────────────────────────────────────

function appendMsg(role, text) {
  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;
  const r = document.createElement("span");
  r.className = "role";
  r.textContent = `${role}>`;
  const c = document.createElement("span");
  c.className = "content";
  c.textContent = ` ${text}`;
  wrap.append(r, c);
  messagesEl.appendChild(wrap);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return c;
}

// iOS standalone PWA: tapping <a download> opens an embedded QuickLook preview
// with no back button — swipe-kill is the only escape. Route through the share
// sheet instead, which has a Cancel button and returns to the PWA cleanly.
const IS_IOS_PWA = window.navigator.standalone === true;

async function shareInsteadOfDownload(e, href, name) {
  e.preventDefault();
  try {
    const res = await fetch(href);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const blob = await res.blob();
    const file = new File([blob], name, {
      type: blob.type || "application/octet-stream",
    });
    if (navigator.canShare && navigator.canShare({ files: [file] })) {
      await navigator.share({ files: [file], title: name });
      return;
    }
    // Web Share Level 2 unavailable on this iOS — open in mobile Safari,
    // which at least gives the user a Done button to return.
    window.open(href, "_blank");
  } catch (err) {
    if (err.name === "AbortError") return; // user tapped Cancel on share sheet
    appendMsg("error", `share failed: ${err.message}`);
  }
}

function appendDownload(role, prefix, name, href) {
  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;
  const r = document.createElement("span");
  r.className = "role";
  r.textContent = `${role}>`;
  const c = document.createElement("span");
  c.className = "content";
  c.textContent = ` ${prefix} `;
  const a = document.createElement("a");
  a.className = "download";
  a.href = href;
  a.textContent = name;
  a.setAttribute("download", name);
  if (IS_IOS_PWA) {
    a.addEventListener("click", (e) => shareInsteadOfDownload(e, href, name));
  }
  c.appendChild(a);
  wrap.append(r, c);
  messagesEl.appendChild(wrap);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

// ─── upload + convert ──────────────────────────────────────────────

function reachableTargets(filename) {
  const ext = (filename.split(".").pop() || "").toLowerCase();
  const targets = new Set();
  for (const c of converters.enabled) {
    if (c.from.includes(ext)) c.to.forEach((t) => targets.add(t));
  }
  targets.delete(ext);
  return [...targets].sort();
}

function findConverter(filename, target) {
  const ext = (filename.split(".").pop() || "").toLowerCase();
  const tgt = (target || "").toLowerCase();
  return converters.enabled.find((c) => c.from.includes(ext) && c.to.includes(tgt));
}

function renderParams(converter) {
  const container = document.getElementById("tray-params");
  container.replaceChildren();
  const params = (converter && converter.params) || {};
  for (const [name, def] of Object.entries(params)) {
    const wrap = document.createElement("label");
    wrap.className = "param";
    const lbl = document.createElement("span");
    lbl.textContent = `${def.label || name}:`;
    const sel = document.createElement("select");
    sel.dataset.param = name;
    for (const c of (def.choices || [])) {
      const opt = document.createElement("option");
      opt.value = String(c.value);
      opt.textContent = c.label || c.value;
      if (String(c.value) === String(def.default)) opt.selected = true;
      sel.appendChild(opt);
    }
    wrap.append(lbl, sel);
    container.appendChild(wrap);
  }
}

function collectParams() {
  const result = {};
  for (const sel of document.querySelectorAll("#tray-params select")) {
    result[sel.dataset.param] = sel.value;
  }
  return result;
}

function showTray(file) {
  pending = file;
  trayName.textContent = file.size
    ? `${file.name} (${formatBytes(file.size)})`
    : file.name;
  const targets = reachableTargets(file.name);
  trayTarget.replaceChildren(...targets.map((t) => makeOption(t)));
  trayConvert.disabled = targets.length === 0;
  if (targets.length === 0) {
    appendMsg("system", `no converters available for .${file.name.split(".").pop()}`);
  }
  renderParams(findConverter(file.name, trayTarget.value));
  tray.classList.remove("hidden");
}

trayTarget.addEventListener("change", () => {
  if (pending) renderParams(findConverter(pending.name, trayTarget.value));
});

async function handleSharedParam() {
  const params = new URLSearchParams(window.location.search);
  const shared = params.get("shared");
  if (!shared) return;
  const names = shared.split(",").map((s) => s.trim()).filter(Boolean);
  if (!names.length) return;
  history.replaceState({}, "", "/");  // clean the URL bar
  for (const name of names) {
    try {
      const meta = await fetch(
        `/api/file-info?session_id=${encodeURIComponent(SESSION)}&name=${encodeURIComponent(name)}`
      ).then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      });
      appendDownload(
        "system",
        `received via share (${formatBytes(meta.size)}):`,
        meta.name,
        `/api/files/${SESSION}/${encodeURIComponent(meta.name)}`,
      );
    } catch (e) {
      appendMsg("error", `share metadata failed for ${name}: ${e.message}`);
    }
  }
  // Pre-populate tray with the first shared file
  showTray({ name: names[0] });
}

function hideTray() {
  pending = null;
  tray.classList.add("hidden");
  fileInput.value = "";
}

function formatBytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

fileInput.addEventListener("change", async () => {
  const f = fileInput.files[0];
  if (!f) return;
  const placeholder = appendMsg("system", "");
  const stopSpinner = startSpinner(placeholder, `uploading ${f.name} (${formatBytes(f.size)})...`);
  const fd = new FormData();
  fd.append("file", f);
  try {
    const r = await fetch(`/api/upload?session_id=${encodeURIComponent(SESSION)}`, {
      method: "POST",
      body: fd,
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const meta = await r.json();
    stopSpinner();
    placeholder.parentElement.remove();
    appendDownload(
      "system",
      `uploaded ${formatBytes(meta.size)}:`,
      meta.name,
      `/api/files/${SESSION}/${encodeURIComponent(meta.name)}`,
    );
    // Inject into chat history so the model knows the file exists and
    // can reference it by name in operation calls. Without this, the
    // model's context contains zero indication that a file was uploaded.
    history.push({
      role: "system",
      content: `User uploaded file "${meta.name}" (${formatBytes(meta.size)}) — available in the workspace as "${meta.name}". When the user asks you to process, convert, edit, or modify this file, use one of the available operations with "${meta.name}" as the source.`,
    });
    showTray(meta);
  } catch (e) {
    stopSpinner();
    placeholder.textContent = ` upload failed: ${e.message}`;
    placeholder.parentElement.classList.add("error");
    hideTray();
  }
});

// ─── Drop folder panel ────────────────────────────────────────────
const dropBtn = $("#drop-btn");
const dropPanel = $("#drop-panel");
const dropBackdrop = $("#drop-backdrop");
const dropList = $("#drop-list");
const dropAdd = $("#drop-add");

function dropRelTime(epochSeconds) {
  const secs = Math.max(0, Date.now() / 1000 - epochSeconds);
  const units = [[86400, "d"], [3600, "h"], [60, "m"]];
  for (const [size, label] of units) {
    if (secs >= size) return `${Math.floor(secs / size)}${label} ago`;
  }
  return "just now";
}

function dropCheckedNames() {
  return [...dropList.querySelectorAll("input[type=checkbox]:checked")].map(
    (c) => c.value,
  );
}

function dropSyncAddState() {
  dropAdd.disabled = dropCheckedNames().length === 0;
}

async function dropRefresh() {
  dropList.innerHTML = "<li class='drop-empty'>loading...</li>";
  try {
    const r = await fetch("/api/dropbox");
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const files = (await r.json()).files || [];
    if (files.length === 0) {
      dropList.innerHTML =
        "<li class='drop-empty'>Drop folder is empty — add files to it and refresh.</li>";
      dropSyncAddState();
      return;
    }
    dropList.innerHTML = "";
    for (const f of files) {
      const li = document.createElement("li");
      const label = document.createElement("label");
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.value = f.name;
      cb.addEventListener("change", dropSyncAddState);
      const name = document.createElement("span");
      name.className = "drop-name";
      name.textContent = f.name;
      const meta = document.createElement("span");
      meta.className = "drop-meta";
      meta.textContent = `${formatBytes(f.size)} · ${dropRelTime(f.modified)}`;
      label.append(cb, name, meta);
      li.append(label);
      dropList.append(li);
    }
  } catch (e) {
    dropList.innerHTML = `<li class='drop-empty'>failed to load: ${e.message}</li>`;
  }
  dropSyncAddState();
}

function openDropPanel() {
  dropPanel.classList.remove("hidden");
  dropBackdrop.classList.remove("hidden");
  dropPanel.setAttribute("aria-hidden", "false");
  dropRefresh();
}

function closeDropPanel() {
  dropPanel.classList.add("hidden");
  dropBackdrop.classList.add("hidden");
  dropPanel.setAttribute("aria-hidden", "true");
}

// Reuse the upload registration path so the model learns each imported file.
function registerImportedFile(meta) {
  appendDownload(
    "system",
    `added from drop folder (${formatBytes(meta.size)}):`,
    meta.name,
    `/api/files/${SESSION}/${encodeURIComponent(meta.name)}`,
  );
  history.push({
    role: "system",
    content: `User added file "${meta.name}" (${formatBytes(meta.size)}) from the drop folder — available in the workspace as "${meta.name}". When the user asks you to process, convert, edit, or modify this file, use one of the available operations with "${meta.name}" as the source.`,
  });
}

dropBtn.addEventListener("click", openDropPanel);
$("#drop-close").addEventListener("click", closeDropPanel);
dropBackdrop.addEventListener("click", closeDropPanel);
$("#drop-refresh").addEventListener("click", dropRefresh);

dropAdd.addEventListener("click", async () => {
  const names = dropCheckedNames();
  if (names.length === 0) return;
  dropAdd.disabled = true;
  try {
    const r = await fetch("/api/dropbox/import", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: SESSION, names }),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const body = await r.json();
    for (const meta of body.imported) registerImportedFile(meta);
    if (body.imported.length === 1) {
      showTray({ ...body.imported[0], session_id: SESSION });
    }
    for (const s of body.skipped) {
      appendMsg("system", `could not add "${s.name}": ${s.reason}`);
    }
    closeDropPanel();
  } catch (e) {
    appendMsg("system", `drop import failed: ${e.message}`);
    dropAdd.disabled = false;
  }
});

trayClear.addEventListener("click", hideTray);

trayConvert.addEventListener("click", async () => {
  if (!pending || !trayTarget.value) return;
  trayConvert.disabled = true;
  const target = trayTarget.value;
  const placeholder = appendMsg("system", "");
  const stopSpinner = startSpinner(placeholder, `converting ${pending.name} → .${target}...`);
  try {
    const r = await fetch("/api/convert", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: SESSION,
        name: pending.name,
        target,
        params: collectParams(),
      }),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
    stopSpinner();
    placeholder.parentElement.remove();
    appendDownload(
      "system",
      `converted via ${data.via} (${formatBytes(data.size)}):`,
      data.name,
      `/api/files/${SESSION}/${encodeURIComponent(data.name)}`,
    );
  } catch (e) {
    stopSpinner();
    placeholder.textContent = ` conversion failed: ${e.message}`;
    placeholder.parentElement.classList.add("error");
  } finally {
    trayConvert.disabled = false;
  }
});

// ─── operations: parse model output, render confirm card, run on Y ────

// Match a fenced ```op:<id>\n{json}\n``` block. Tolerant of trailing
// whitespace and of the model forgetting the closing fence.
const OP_FENCE_RE = /```op:([a-zA-Z0-9_]+)\s*\n([\s\S]*?)(?:```|$)/g;

function findOperation(opId) {
  return operations.enabled.find((op) => op.id === opId) || null;
}

function parseOpFromText(text) {
  const out = [];
  for (const match of text.matchAll(OP_FENCE_RE)) {
    const [, opId, body] = match;
    let params;
    try {
      params = JSON.parse(body.trim());
    } catch {
      continue;  // skip malformed blocks, the model can retry
    }
    out.push({ operation: opId, params });
  }
  return out;
}

function parseToolCallSentinel(text) {
  // Backend emits a one-line JSON sentinel `{"__tool_calls__": [...]}` for
  // native Ollama tool_calls. Find it, strip it from displayed text.
  const idx = text.lastIndexOf('{"__tool_calls__"');
  if (idx < 0) return { calls: [], cleanedText: text };
  const tail = text.slice(idx);
  const newline = tail.indexOf("\n");
  const jsonLine = newline === -1 ? tail : tail.slice(0, newline);
  try {
    const parsed = JSON.parse(jsonLine);
    const calls = (parsed.__tool_calls__ || []).map((tc) => ({
      operation: tc.function?.name,
      params: tc.function?.arguments || {},
    }));
    return { calls, cleanedText: text.slice(0, idx).trimEnd() };
  } catch {
    return { calls: [], cleanedText: text };
  }
}

// Mirror parseToolCallSentinel for `{"__stats__": {...}}` lines emitted at
// stream end. Returns the parsed stats object (or null) plus the text with
// the sentinel removed.
function parseStatsSentinel(text) {
  const idx = text.lastIndexOf('{"__stats__"');
  if (idx < 0) return { stats: null, cleanedText: text };
  const tail = text.slice(idx);
  const newline = tail.indexOf("\n");
  const jsonLine = newline === -1 ? tail : tail.slice(0, newline);
  try {
    const parsed = JSON.parse(jsonLine);
    return { stats: parsed.__stats__ || null, cleanedText: text.slice(0, idx).trimEnd() };
  } catch {
    return { stats: null, cleanedText: text };
  }
}

// Extract ALL `{"__cellc_step__": {...}}` sentinel lines (they interleave with
// text across loop iterations, unlike the single __stats__/__tool_calls__).
// Returns {steps: [{tool, summary}], cleanedText}.
function parseCellcSteps(text) {
  const steps = [];
  const lines = text.split("\n");
  const kept = [];
  for (const line of lines) {
    const t = line.trim();
    if (t.startsWith('{"__cellc_step__"')) {
      try { steps.push(JSON.parse(t).__cellc_step__); continue; } catch {}
    }
    kept.push(line);
  }
  return { steps, cleanedText: kept.join("\n") };
}

function renderCellcSteps(steps) {
  if (!steps.length) return "";
  return steps.map((s) => `\n🔧 ${s.summary}`).join("");
}

// ─── reasoning (thinking) channel ──────────────────────────────────
// Reasoning models (GLM-4.7-flash, qwen3.6) stream their chain-of-thought
// on a separate Ollama `thinking` field. The backend splices that run into
// the text stream bracketed by THINK_OPEN/THINK_CLOSE sentinels (two \x01
// control chars + tag) so we can render it in a distinct, collapsible block
// — and still scan it for code blocks, since some models (GLM-4.7-flash)
// emit the actual artifact inside their reasoning rather than the answer.
const _THINK_SENTINEL = String.fromCharCode(1, 1);
const THINK_OPEN = `${_THINK_SENTINEL}THINK${_THINK_SENTINEL}`;
const THINK_CLOSE = `${_THINK_SENTINEL}/THINK${_THINK_SENTINEL}`;

// Split a streamed assistant string into { thinking, content }. Handles the
// common single-block case (reasoning first, then answer) and the still-
// streaming case where THINK_CLOSE hasn't landed yet.
function splitThinking(text) {
  const open = text.indexOf(THINK_OPEN);
  if (open === -1) return { thinking: "", content: text };
  const afterOpen = open + THINK_OPEN.length;
  const close = text.indexOf(THINK_CLOSE, afterOpen);
  if (close === -1) {
    // Reasoning still streaming — everything past the marker is thinking.
    return { thinking: text.slice(afterOpen), content: text.slice(0, open) };
  }
  const thinking = text.slice(afterOpen, close);
  const content = text.slice(0, open) + text.slice(close + THINK_CLOSE.length);
  return { thinking, content };
}

// Lazily create (once) the collapsible reasoning block for an assistant
// bubble and return its body node. Inserted before the content span so the
// thought process reads above the answer.
function reasoningBody(out) {
  const wrap = out.parentElement;
  let el = wrap.querySelector(":scope > .reasoning");
  if (!el) {
    el = document.createElement("details");
    el.className = "reasoning";
    el.open = !settings.collapseReasoning;
    const summary = document.createElement("summary");
    summary.textContent = "reasoning";
    const body = document.createElement("div");
    body.className = "reasoning-body";
    el.append(summary, body);
    wrap.insertBefore(el, out);
  }
  return el.querySelector(".reasoning-body");
}

// Render a streamed assistant accumulator live: reasoning to its own block,
// answer to the content span.
function renderAssistant(out, acc) {
  const { thinking, content } = splitThinking(acc);
  if (thinking) reasoningBody(out).textContent = thinking;
  out.textContent = ` ${content}`;
}

// ─── code-block downloads ──────────────────────────────────────────
// When the model emits fenced code blocks with a language tag (e.g.
// ```html), surface a download button next to the assistant bubble so
// the user can save it as a file on their device. Frontend-only via Blob
// URLs — no backend round-trip, works the same on phone and desktop.

// Map fence language to file extension. Unknown langs fall back to the
// lang string itself as the extension (lowercased) so e.g. ```rs gives
// snippet-N.rs and ```julia gives snippet-N.julia.
const LANG_TO_EXT = {
  html: "html", htm: "html", xhtml: "html",
  md: "md", markdown: "md",
  json: "json", jsonc: "json", json5: "json",
  js: "js", javascript: "js", mjs: "mjs", cjs: "cjs",
  ts: "ts", typescript: "ts",
  jsx: "jsx", tsx: "tsx",
  py: "py", python: "py",
  c: "c", h: "h",
  cpp: "cpp", "c++": "cpp", cxx: "cpp", cc: "cpp", hpp: "hpp", hxx: "hpp",
  rs: "rs", rust: "rs",
  go: "go", golang: "go",
  java: "java", kt: "kt", kotlin: "kt", swift: "swift",
  rb: "rb", ruby: "rb", php: "php",
  yaml: "yaml", yml: "yaml", toml: "toml", ini: "ini",
  xml: "xml", svg: "svg",
  css: "css", scss: "scss", sass: "sass", less: "less",
  sh: "sh", bash: "sh", zsh: "sh", shell: "sh", ps1: "ps1", powershell: "ps1",
  sql: "sql", graphql: "graphql", gql: "graphql",
  diff: "diff", patch: "patch",
  csv: "csv", tsv: "tsv",
  text: "txt", txt: "txt", plain: "txt",
  vue: "vue", svelte: "svelte",
};
// Some langs map to a filename, not an extension — set the filename
// directly rather than appending `.<lang>` to "snippet-N".
const LANG_TO_FILENAME = {
  dockerfile: "Dockerfile",
  makefile: "Makefile",
  cmake: "CMakeLists.txt",
};

// Captures ```<info>?\n...body...\n``` blocks. The info string is anything
// up to the newline — CommonMark allows arbitrary text there ("python",
// "file:foo.py", "python title='example'"). We parse it in parseFenceInfo
// to extract language and/or filename. Models trained on GitHub READMEs
// commonly emit ```file:<name> as an explicit filename signal — that's
// the case we most want to honour. Op fences are stripped upstream so we
// won't double-up here.
const CODE_FENCE_FOR_DOWNLOAD_RE = /```([^\n]*)\n([\s\S]*?)```/g;

// Pull a {lang, filename} hint out of a fence info string. Patterns seen
// in the wild:
//   ```python                        → lang=python
//   ```file:snake.html               → filename=snake.html
//   ```snake.html                    → filename=snake.html (looks like a path)
//   ```python title="example.py"     → lang=python, filename=example.py
//   ```html name=index.html          → lang=html, filename=index.html
//   ```python:example.py             → lang=python, filename=example.py
function parseFenceInfo(info) {
  const raw = (info || "").trim();
  if (!raw) return { lang: "", filename: "" };
  // Quoted name=... / title=... / filename=... attribute pattern.
  const attr = raw.match(/(?:file(?:name)?|name|title)\s*=\s*["']?([^"'\s]+\.[a-zA-Z0-9]+)["']?/i);
  if (attr) {
    const firstToken = raw.split(/[\s:=]+/)[0].toLowerCase();
    const lang = /^[a-zA-Z0-9_+\-]+$/.test(firstToken) ? firstToken : "";
    return { lang, filename: attr[1] };
  }
  // Explicit `file:foo.ext` or `filename:foo.ext` prefix.
  const filePrefix = raw.match(/^(?:file(?:name)?)\s*:\s*([^\s]+)/i);
  if (filePrefix) return { lang: "", filename: filePrefix[1] };
  // `lang:filename.ext` compound (uncommon but seen).
  const langColon = raw.match(/^([a-zA-Z0-9_+\-]+)\s*:\s*([^\s]+\.[a-zA-Z0-9]+)/);
  if (langColon) return { lang: langColon[1].toLowerCase(), filename: langColon[2] };
  // Bare filename-shaped token (`snake.html`, `src/main.rs`).
  if (/^[a-zA-Z0-9_.\-/]+\.[a-zA-Z0-9]+$/.test(raw)) return { lang: "", filename: raw };
  // Bare language token (`python`, `c++`).
  const first = raw.split(/\s+/)[0].toLowerCase();
  if (/^[a-zA-Z0-9_+\-]+$/.test(first)) return { lang: first, filename: "" };
  return { lang: "", filename: "" };
}

// Look in the line just before the fence for a filename hint. Matches:
//   // index.html        // file: index.html
//   # script.py           # file: script.py
//   <!-- page.html -->
//   **foo.c**             `bar.h`
const FILENAME_HINT_RE = /(?:\/\/|#|<!--|\*\*|`)\s*(?:file:\s*)?([a-zA-Z0-9_.\-/]+\.[a-zA-Z0-9]+)\s*(?:-->|\*\*|`)?\s*$/;

// Best-effort extension sniff for naked fences. Reads the first non-blank
// line and pattern-matches against well-known signatures. Falls back to
// "txt" so the file is always downloadable even when we can't classify.
function sniffExtension(body) {
  const first = (body.split("\n").find((l) => l.trim()) || "").trim();
  const lower = first.toLowerCase();
  // Shebangs first — most decisive signal.
  if (first.startsWith("#!/bin/bash") || first.startsWith("#!/usr/bin/env bash") || first.startsWith("#!/bin/sh") || first.startsWith("#!/usr/bin/env sh")) return "sh";
  if (first.startsWith("#!/usr/bin/env python") || first.startsWith("#!/usr/bin/python")) return "py";
  if (first.startsWith("#!/usr/bin/env node")) return "js";
  if (first.startsWith("#!/usr/bin/env ruby")) return "rb";
  // Markup / declarations.
  if (lower.startsWith("<!doctype") || lower.startsWith("<html")) return "html";
  if (first.startsWith("<?xml")) return "xml";
  if (first.startsWith("<?php") || body.includes("<?php")) return "php";
  if (first.startsWith("<svg")) return "svg";
  // Language signatures detectable without parsing.
  if (first.startsWith("package ") && body.includes("\nfunc ")) return "go";
  if (first.startsWith("package ") && body.includes("\nimport ")) return "java";
  if (first.startsWith("use ") && body.includes("\nfn ")) return "rs";
  if (first.startsWith("fn ") || body.includes("\nfn main(")) return "rs";
  // Python: imports, def, class — Pygame/Flask/etc. scripts rarely
  // start with a shebang, so this signature catches the common case.
  if (first.startsWith("import ") || first.startsWith("from ")) return "py";
  if (first.startsWith("def ") || first.startsWith("class ") || body.includes("\ndef ") || body.includes("\nclass ")) return "py";
  // JS/TS — function/const/let/var at top, or import/export ESM.
  if (first.startsWith("function ") || /^(const|let|var) \w/.test(first)) return "js";
  if (first.startsWith("export ") || (first.startsWith("import ") && first.includes(" from "))) return "js";
  // CSS — selector { ... }.
  if (/^[#.@*a-zA-Z][\w\-:,>\s\[\]"=()]*\{/.test(first)) return "css";
  // JSON — opens/closes with brackets and parses cleanly.
  const trimmed = body.trim();
  if ((trimmed.startsWith("{") && trimmed.endsWith("}")) || (trimmed.startsWith("[") && trimmed.endsWith("]"))) {
    try { JSON.parse(trimmed); return "json"; } catch { /* not strict JSON */ }
  }
  // YAML — `key: value` style with no JSON braces.
  if (/^[a-zA-Z_][\w\-]*:\s/.test(first) && !trimmed.startsWith("{")) return "yaml";
  return "txt";
}

function extractCodeBlocks(text) {
  const blocks = [];
  let counter = 0;
  for (const match of text.matchAll(CODE_FENCE_FOR_DOWNLOAD_RE)) {
    const info = match[1] || "";
    const body = match[2] || "";
    // op:<id> fences are confirm-card material, not files.
    if (info.trim().toLowerCase().startsWith("op:")) continue;
    const { lang, filename: infoFilename } = parseFenceInfo(info);
    // Filename precedence (highest to lowest):
    //   1. filename in the fence info string  (`file:foo.py`, `python:foo.py`)
    //   2. filename hint on the line just before the fence (`**foo.py**`)
    //   3. LANG_TO_FILENAME map (Dockerfile, Makefile, ...)
    //   4. lang-derived `snippet-N.<ext>`
    //   5. content-sniffed `snippet-N.<sniffed-ext>`
    const before = text.slice(0, match.index).trimEnd();
    const lastLine = before.slice(before.lastIndexOf("\n") + 1);
    const hint = lastLine.match(FILENAME_HINT_RE);
    // Gate naked fences (no lang, no filename anywhere): skip only the
    // obvious one-line inline samples. Anything multi-line is plausibly
    // an artifact — bias toward more buttons, not fewer.
    const bodyLines = body.split("\n").filter((l) => l.length > 0).length;
    const truly_naked = !lang && !infoFilename && !hint;
    if (truly_naked && bodyLines < 2) continue;
    counter++;
    let filename;
    if (infoFilename) {
      filename = infoFilename;
    } else if (hint && hint[1]) {
      filename = hint[1];
    } else if (LANG_TO_FILENAME[lang]) {
      filename = LANG_TO_FILENAME[lang];
    } else if (lang) {
      const ext = LANG_TO_EXT[lang] || lang;
      filename = `snippet-${counter}.${ext}`;
    } else {
      filename = `snippet-${counter}.${sniffExtension(body)}`;
    }
    blocks.push({ filename, body: body.replace(/\s+$/, "") + "\n", lang: lang || "text" });
  }
  // Dedupe identical snippets: a reasoning model may emit the same artifact
  // in both its answer and its thinking channel, and we scan both — keep the
  // first occurrence (answer-channel wins, since scanText lists it first).
  const seen = new Set();
  return blocks.filter((b) => {
    const key = b.body.trim();
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function appendCodeDownloads(msgWrap, blocks) {
  if (!msgWrap || !blocks.length) return;
  const row = document.createElement("div");
  row.className = "code-downloads";
  for (const b of blocks) {
    const blob = new Blob([b.body], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.className = "code-download";
    a.href = url;
    a.download = b.filename;
    a.textContent = `⬇ ${b.filename}`;
    a.title = `${b.body.length} bytes · ${b.lang || "text"}`;
    if (IS_IOS_PWA) {
      a.addEventListener("click", (e) => shareInsteadOfDownload(e, url, b.filename));
    }
    row.appendChild(a);
  }
  msgWrap.appendChild(row);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function setStatsBusy(label) {
  if (!statsBar) return;
  statsBar.classList.remove("idle");
  statsBar.classList.add("busy");
  statsBar.textContent = label || "generating...";
}

function updateStatsBar(stats) {
  if (!statsBar || !stats) return;
  const parts = [];
  const loadS = (stats.load_ns || 0) / 1e9;
  const evalS = (stats.eval_ns || 0) / 1e9;
  const promptS = (stats.prompt_eval_ns || 0) / 1e9;
  const tokOut = stats.eval_count || 0;
  const tokIn = stats.prompt_eval_count || 0;
  const tps = evalS > 0 ? tokOut / evalS : 0;

  // Surface load time only when meaningful (Ollama returns 0 for warm
  // requests). Eviction-and-reload events show up as a spike here — the
  // whole point of the bar is making those visible.
  if (loadS >= 0.05) parts.push(`load ${loadS.toFixed(1)}s`);
  if (promptS >= 0.05) parts.push(`prompt ${tokIn}t/${promptS.toFixed(1)}s`);
  if (evalS > 0) parts.push(`gen ${evalS.toFixed(1)}s`);
  if (tps > 0) parts.push(`${tps.toFixed(1)} tok/s`);
  if (tokOut) parts.push(`${tokOut} out`);
  if (stats.model) parts.push(stats.model);

  statsBar.classList.remove("busy", "idle");
  statsBar.textContent = parts.join(" · ") || "done";
}

async function maybeEnhancePrompt(op, params) {
  if (!op.enhance) return null;
  const userPrompt = String(params.prompt || "").trim();
  if (!userPrompt) return null;
  try {
    const r = await fetch("/api/operations/enhance", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: userPrompt,
        chat_model: modelSel.value,
        image_model: params.model || "",
        mode: params.mode || "auto",
      }),
    });
    if (!r.ok) return null;
    return await r.json();
  } catch {
    return null;
  }
}

// Returns a list of capability keys the current model lacks for this op.
// "text" is universal (every chat model satisfies it) and skipped.
function missingCapabilities(opMeta) {
  const required = opMeta?.capabilities_required || ["text"];
  const caps = capabilities[modelSel.value] || {};
  const missing = [];
  for (const need of required) {
    if (need === "text") continue;
    if (need === "tool_calling" && caps.tool_calling === "native") continue;
    if (need === "vision" && caps.vision) continue;
    if (need === "audio" && caps.audio) continue;
    if (need === "reasoning" && caps.reasoning) continue;
    missing.push(need);
  }
  return missing;
}

// Returns the list of model names that DO satisfy every required cap.
function modelsSatisfying(required) {
  const out = [];
  for (const [name, caps] of Object.entries(capabilities)) {
    let ok = true;
    for (const need of required) {
      if (need === "text") continue;
      if (need === "tool_calling" && caps.tool_calling !== "native") { ok = false; break; }
      if (need === "vision" && !caps.vision) { ok = false; break; }
      if (need === "audio" && !caps.audio) { ok = false; break; }
      if (need === "reasoning" && !caps.reasoning) { ok = false; break; }
    }
    if (ok) out.push(name);
  }
  return out;
}

function buildOpCard(call, opMeta, enhanced) {
  const card = document.createElement("div");
  card.className = "op-card";

  const head = document.createElement("div");
  head.className = "op-card-head";
  head.textContent = `operation> ${call.operation}`;
  card.appendChild(head);

  // Capability warn banner — warn-and-allow per fork B. The user can
  // still hit Y; the warning is informational. Suggests other models
  // that DO satisfy the requirement.
  const missing = missingCapabilities(opMeta);
  if (missing.length) {
    const warn = document.createElement("div");
    warn.className = "op-warn";
    const candidates = modelsSatisfying(opMeta.capabilities_required || []);
    const pick = candidates.filter((n) => n !== modelSel.value).slice(0, 3);
    const suggest = pick.length
      ? ` Models that have it: ${pick.join(", ")}.`
      : " No probed model in your dropdown satisfies this — you'll need to switch models or add capability via vLLM.";
    warn.textContent = `⚠ ${modelSel.value} doesn't have: ${missing.join(", ")}.${suggest} Y still runs.`;
    card.appendChild(warn);
  }

  const body = document.createElement("div");
  body.className = "op-card-body";
  card.appendChild(body);

  const editable = {};  // name → element with .value

  function row(label, value, editKey = null) {
    const r = document.createElement("div");
    r.className = "op-row";
    const l = document.createElement("span");
    l.className = "op-label";
    l.textContent = label;
    r.appendChild(l);
    if (editKey) {
      const ta = document.createElement("textarea");
      ta.className = "op-edit";
      ta.value = value || "";
      ta.rows = Math.min(6, Math.max(2, String(value || "").split("\n").length + 1));
      r.appendChild(ta);
      editable[editKey] = ta;
    } else {
      const v = document.createElement("span");
      v.className = "op-value";
      v.textContent = value;
      r.appendChild(v);
    }
    body.appendChild(r);
  }

  if (enhanced && enhanced.mode !== "passthrough") {
    row("original", enhanced.original_prompt);
    row("enhanced", enhanced.enhanced_prompt, "prompt");
    row("negative", enhanced.negative_prompt, "negative_prompt");
    if (enhanced.changes) row("changes", enhanced.changes);
    row("mode", enhanced.mode);
  } else if (call.params.prompt !== undefined) {
    row("prompt", String(call.params.prompt || ""), "prompt");
    if (call.params.negative_prompt !== undefined) {
      row("negative", String(call.params.negative_prompt || ""), "negative_prompt");
    }
  }

  // Show every other param for transparency, non-editable.
  for (const [name, value] of Object.entries(call.params)) {
    if (["prompt", "negative_prompt", "mode"].includes(name)) continue;
    row(name, String(value));
  }
  if (opMeta?.source_param && pending?.name) {
    row("source", pending.name);
  }

  const actions = document.createElement("div");
  actions.className = "op-actions";
  const yes = document.createElement("button");
  yes.type = "button";
  yes.className = "op-btn";
  yes.textContent = "[y] run";
  const no = document.createElement("button");
  no.type = "button";
  no.className = "op-btn ghost";
  no.textContent = "[n] cancel";
  actions.append(yes, no);
  card.appendChild(actions);

  const hint = document.createElement("div");
  hint.className = "op-hint";
  hint.textContent = "y to run · n to cancel · edit fields above first if needed";
  card.appendChild(hint);

  return { card, yes, no, editable };
}

async function presentOpConfirm(call) {
  const opMeta = findOperation(call.operation);
  if (!opMeta) {
    appendMsg("error", `model requested unknown operation '${call.operation}'`);
    return;
  }

  const status = appendMsg("system", "");
  const stopSpin = startSpinner(status, `preparing ${call.operation}...`);
  let enhanced = null;
  try {
    enhanced = await maybeEnhancePrompt(opMeta, call.params);
  } finally {
    stopSpin();
    status.parentElement.remove();
  }

  const { card, yes, no, editable } = buildOpCard(call, opMeta, enhanced);
  const wrap = document.createElement("div");
  wrap.className = "msg system";
  const role = document.createElement("span");
  role.className = "role";
  role.textContent = "operation>";
  const content = document.createElement("span");
  content.className = "content";
  content.appendChild(card);
  wrap.append(role, content);
  messagesEl.appendChild(wrap);
  messagesEl.scrollTop = messagesEl.scrollHeight;

  pendingOpCard = { card, yes, no, editable, call, opMeta, enhanced, wrap };

  const onYes = () => runConfirmedOp();
  const onNo = () => cancelConfirmedOp();
  yes.addEventListener("click", onYes);
  no.addEventListener("click", onNo);
}

async function runConfirmedOp() {
  if (!pendingOpCard) return;
  const { card, yes, no, editable, call, opMeta, enhanced, wrap } = pendingOpCard;
  pendingOpCard = null;
  yes.disabled = true; no.disabled = true;

  // Merge edits back into params.
  const finalParams = { ...call.params };
  for (const [k, el] of Object.entries(editable)) {
    finalParams[k] = el.value.trim();
  }
  // If we ran an enhance, the negative prompt should also flow through.
  if (enhanced && enhanced.mode !== "passthrough") {
    if (editable.prompt) finalParams.prompt = editable.prompt.value.trim();
    if (editable.negative_prompt) finalParams.negative_prompt = editable.negative_prompt.value.trim();
  }

  const placeholder = document.createElement("div");
  placeholder.className = "op-hint";
  card.appendChild(placeholder);
  const stopSpin = startSpinner(placeholder, `running ${call.operation}...`);

  const body = {
    operation: call.operation,
    session_id: SESSION,
    params: finalParams,
  };
  if (opMeta.source_param && pending?.name) {
    body.source = pending.name;
  }

  try {
    const r = await fetch("/api/operations/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
    stopSpin();
    placeholder.remove();
    card.classList.add("completed");
    appendDownload(
      "system",
      `${data.via} produced ${formatBytes(data.size)}${data.mirror ? ` (mirrored: ${data.mirror})` : ""}:`,
      data.name,
      data.url || `/api/files/${SESSION}/${encodeURIComponent(data.name)}`,
    );
    // Feed the result back into chat history so the model can reference it.
    history.push({
      role: "assistant",
      content: `[operation ${data.via} produced ${data.name}]`,
    });
  } catch (e) {
    stopSpin();
    placeholder.textContent = ` operation failed: ${e.message}`;
    wrap.classList.add("error");
  }
}

function cancelConfirmedOp() {
  if (!pendingOpCard) return;
  const { card, yes, no } = pendingOpCard;
  yes.disabled = true; no.disabled = true;
  card.classList.add("completed");
  appendMsg("system", `cancelled — operation not run`);
  pendingOpCard = null;
}

document.addEventListener("keydown", (e) => {
  // y/n/e shortcuts only fire when a card is awaiting and the user isn't
  // typing in the composer (so 'y' inside a prompt doesn't fire-and-forget).
  if (!pendingOpCard) return;
  if (document.activeElement && ["TEXTAREA", "INPUT", "SELECT"].includes(document.activeElement.tagName)) return;
  if (e.key === "y" || e.key === "Y") { e.preventDefault(); runConfirmedOp(); }
  else if (e.key === "n" || e.key === "N") { e.preventDefault(); cancelConfirmedOp(); }
  else if (e.key === "e" || e.key === "E") {
    e.preventDefault();
    const first = pendingOpCard.editable.prompt || Object.values(pendingOpCard.editable)[0];
    if (first) first.focus();
  }
});

// ─── chat ──────────────────────────────────────────────────────────

async function send() {
  const text = input.value.trim();
  if (!text || streaming) return;
  if (!modelSel.value) {
    appendMsg("error", "No model selected.");
    return;
  }

  input.value = "";
  autosize();

  // ─── /check slash command (cellc, Phase 1) ────────────────────────
  if (text.startsWith("/check") && typeof window.cellc !== "undefined") {
    const tail = text.slice("/check".length).trim();
    let src = tail;
    if (!src) {
      // Scan history for the most recent assistant .cell/.cellscript fence
      for (let i = history.length - 1; i >= 0; i--) {
        if (history[i].role !== "assistant") continue;
        const m = history[i].content.match(/```(?:cell|cellscript)\n([\s\S]*?)```/i);
        if (m) { src = m[1]; break; }
      }
    }
    if (!src) {
      appendMsg("system", "/check: provide source after /check, or ask the model to write a .cell block first");
    } else {
      const out = appendMsg("system", "checking…");
      const result = await window.cellc.checkSource(src);
      out.textContent = ` ${result}`;
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }
    return;
  }

  // Translate mode: routed when the active model is a TranslateGemma
  // tag. We don't push to chat history — each translation is fresh —
  // but the user's input still appears in the messages stream so they
  // can see what was asked.
  if (isTranslateModel(modelSel.value)) {
    const src = translateSrcSel.value;
    const tgt = translateTgtSel.value;
    appendMsg("user", text);
    appendMsg("system", `→ ${src} → ${tgt}`);
    return translateSend(text, src, tgt);
  }

  // Auto-router: caption intent + uploaded image + captioner present →
  // bypass the selected chat model and route directly to the captioner.
  // The user sees a clear notification; falls through to normal chat
  // on any error so the request never silently fails.
  if (shouldAutoCaption(text)) {
    history.push({ role: "user", content: text });
    appendMsg("user", text);
    appendMsg(
      "system",
      `→ auto-routing to captioner (${autoRouter.captioner_model}) — specialised for image description, faster than ${modelSel.value} for this`,
    );
    return autoCaptionSend(text);
  }

  history.push({ role: "user", content: text });
  appendMsg("user", text);
  const out = appendMsg("assistant", "");
  out.parentElement.classList.add("streaming");

  // "Thinking" indicator — Claude-Code-style spinner + label + elapsed
  // timer that fills the dead-air gap before the first token arrives.
  // Auto-cancels when the first chunk lands; the blinking cursor takes
  // over from there. Spinner frames + interval inherit from settings.
  const thinkCfg = SPINNERS[settings.spinner] || SPINNERS.braille;
  const thinkStart = Date.now();
  let thinkFrame = 0;
  let thinking = true;
  out.classList.add("thinking");
  // Initial paint as fallback even if the spinner interval is somehow
  // suppressed (e.g. by a backgrounded tab throttling timers): the user
  // always sees at minimum a static "thinking · 0s" the moment they hit send.
  out.textContent = ` ${thinkCfg.frames[0]} thinking · 0s`;
  const renderThinking = () => {
    if (!thinking) return;
    const elapsed = Math.floor((Date.now() - thinkStart) / 1000);
    out.textContent = ` ${thinkCfg.frames[thinkFrame]} thinking · ${elapsed}s`;
    thinkFrame = (thinkFrame + 1) % thinkCfg.frames.length;
  };
  const thinkInterval = setInterval(renderThinking, thinkCfg.interval);
  const stopThinking = () => {
    if (!thinking) return;
    thinking = false;
    clearInterval(thinkInterval);
    out.classList.remove("thinking");
    out.classList.add("streaming");
  };

  streaming = true;
  sendBtn.disabled = true;

  // Sticky image attachment: if a pending image is in the tray and the
  // active model has vision, ride it along on every turn. Ollama accepts
  // base64 image bytes per-message; the backend re-reads from the
  // workspace and attaches to the last user message.
  const attachImage = !!(
    pending &&
    isImageFile(pending.name) &&
    capabilities[modelSel.value]?.vision
  );
  const chatBody = { model: modelSel.value, messages: history };
  if (attachImage) {
    chatBody.image_files = [pending.name];
    chatBody.session_id = SESSION;
  }

  setStatsBusy(`${modelSel.value} · generating...`);
  let acc = "";
  try {
    const r = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(chatBody),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const reader = r.body.getReader();
    const dec = new TextDecoder();
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      acc += dec.decode(value, { stream: true });
      if (thinking) stopThinking();  // first chunk wins
      renderAssistant(out, acc);
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    // Strip all sentinels out of displayed text. cellc steps are stripped
    // first (multiple interleaved occurrences), then the once-at-end
    // stats/tool_calls sentinels. Each parser hands its cleanedText to
    // the next in the chain.
    const { steps: cellcSteps, cleanedText: noSteps } = parseCellcSteps(acc);
    const { stats, cleanedText: noStats } = parseStatsSentinel(noSteps);
    if (stats) updateStatsBar(stats);
    const { calls: toolCalls, cleanedText } = parseToolCallSentinel(noStats);
    // Separate the reasoning channel before op/fence processing so tool
    // detection and the visible answer consider only the model's answer.
    const { thinking: reasoning, content: answerText } = splitThinking(cleanedText);
    // Strip op fences from displayed text too — they're already shown as
    // confirm cards below, so leaving them as raw JSON in the assistant
    // line is just noise.
    const opCalls = parseOpFromText(answerText);
    const allCalls = [...toolCalls, ...opCalls];
    const textWithoutFences = answerText.replace(OP_FENCE_RE, "").trim();

    // Pure-tool-caller path: model emitted only structured calls, no prose.
    // Replace the blank line with a one-line summary so the user knows
    // *something* happened. The card below has the detail.
    let displayedText = textWithoutFences;
    if (!displayedText && allCalls.length) {
      const ids = allCalls.map((c) => c.operation).filter(Boolean);
      displayedText = `→ requested ${ids.length > 1 ? "operations" : "operation"}: ${ids.join(", ")}`;
    } else if (!displayedText && !allCalls.length) {
      displayedText = "[empty response]";
    }
    if (reasoning) reasoningBody(out).textContent = reasoning;
    const stepsHint = renderCellcSteps(cellcSteps);
    out.textContent = stepsHint
      ? ` ${stepsHint.trim()}\n\n${displayedText}`
      : ` ${displayedText}`;
    history.push({ role: "assistant", content: displayedText });

    // Surface any fenced code blocks (```html, ```py, etc.) as download
    // buttons under the assistant bubble. Scans the answer AND the reasoning
    // — some models (GLM-4.7-flash) emit the actual artifact inside their
    // thinking, so scanning only the answer would drop the download button.
    const scanText = reasoning ? `${displayedText}\n\n${reasoning}` : displayedText;
    appendCodeDownloads(out.parentElement, extractCodeBlocks(scanText));

    // One card at a time — y/n/e handler operates on a single pending card.
    // If the model emits multiple, queue them sequentially.
    for (const call of allCalls) {
      if (call.operation) await presentOpConfirm(call);
    }
  } catch (e) {
    stopThinking();
    out.textContent = ` [error: ${e.message}]`;
    out.parentElement.classList.add("error");
    if (statsBar) {
      statsBar.classList.remove("busy");
      statsBar.classList.add("idle");
      statsBar.textContent = `error: ${e.message}`;
    }
  } finally {
    stopThinking();  // idempotent — covers stream-completed-with-no-chunks edge
    out.classList.remove("streaming");
    out.parentElement.classList.remove("streaming");
    streaming = false;
    sendBtn.disabled = false;
    input.focus();
  }
}

async function autoCaptionSend(text) {
  const out = appendMsg("assistant", "");
  out.parentElement.classList.add("streaming");

  // Reuse the thinking spinner — captioner model load is the main wait.
  const thinkCfg = SPINNERS[settings.spinner] || SPINNERS.braille;
  const thinkStart = Date.now();
  let thinkFrame = 0;
  let thinking = true;
  out.classList.add("thinking");
  out.textContent = ` ${thinkCfg.frames[0]} captioning · 0s`;
  const renderThinking = () => {
    if (!thinking) return;
    const elapsed = Math.floor((Date.now() - thinkStart) / 1000);
    out.textContent = ` ${thinkCfg.frames[thinkFrame]} captioning · ${elapsed}s`;
    thinkFrame = (thinkFrame + 1) % thinkCfg.frames.length;
  };
  const thinkInterval = setInterval(renderThinking, thinkCfg.interval);
  const stopThinking = () => {
    if (!thinking) return;
    thinking = false;
    clearInterval(thinkInterval);
    out.classList.remove("thinking");
    out.classList.add("streaming");
  };

  streaming = true;
  sendBtn.disabled = true;

  let acc = "";
  try {
    const r = await fetch("/api/auto-caption", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: SESSION,
        source: pending.name,
        prompt: text,
      }),
    });
    if (!r.ok) {
      const err = await r.text();
      throw new Error(`HTTP ${r.status}: ${err.slice(0, 200)}`);
    }
    const reader = r.body.getReader();
    const dec = new TextDecoder();
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      acc += dec.decode(value, { stream: true });
      if (thinking) stopThinking();
      out.textContent = ` ${acc}`;
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }
    history.push({ role: "assistant", content: acc });
  } catch (e) {
    stopThinking();
    out.textContent = ` [auto-caption failed: ${e.message}]`;
    out.parentElement.classList.add("error");
  } finally {
    stopThinking();
    out.classList.remove("streaming");
    out.parentElement.classList.remove("streaming");
    streaming = false;
    sendBtn.disabled = false;
    input.focus();
  }
}

async function translateSend(text, sourceLang, targetLang) {
  const out = appendMsg("assistant", "");
  out.parentElement.classList.add("streaming");

  // Reuse the thinking-spinner pattern — first translation on a
  // cold-loaded 27 B is the slow case, subsequent ones are sub-second.
  const thinkCfg = SPINNERS[settings.spinner] || SPINNERS.braille;
  const thinkStart = Date.now();
  let thinkFrame = 0;
  let thinking = true;
  out.classList.add("thinking");
  out.textContent = ` ${thinkCfg.frames[0]} translating · 0s`;
  const renderThinking = () => {
    if (!thinking) return;
    const elapsed = Math.floor((Date.now() - thinkStart) / 1000);
    out.textContent = ` ${thinkCfg.frames[thinkFrame]} translating · ${elapsed}s`;
    thinkFrame = (thinkFrame + 1) % thinkCfg.frames.length;
  };
  const thinkInterval = setInterval(renderThinking, thinkCfg.interval);
  const stopThinking = () => {
    if (!thinking) return;
    thinking = false;
    clearInterval(thinkInterval);
    out.classList.remove("thinking");
    out.classList.add("streaming");
  };

  streaming = true;
  sendBtn.disabled = true;

  let acc = "";
  try {
    const r = await fetch("/api/operations/translate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: modelSel.value,
        source_lang: sourceLang,
        target_lang: targetLang,
        text,
      }),
    });
    if (!r.ok) {
      const err = await r.text();
      throw new Error(`HTTP ${r.status}: ${err.slice(0, 200)}`);
    }
    const reader = r.body.getReader();
    const dec = new TextDecoder();
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      acc += dec.decode(value, { stream: true });
      if (thinking) stopThinking();
      out.textContent = ` ${acc}`;
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }
  } catch (e) {
    stopThinking();
    out.textContent = ` [translate failed: ${e.message}]`;
    out.parentElement.classList.add("error");
  } finally {
    stopThinking();
    out.classList.remove("streaming");
    out.parentElement.classList.remove("streaming");
    streaming = false;
    sendBtn.disabled = false;
    input.focus();
  }
}

composer.addEventListener("submit", (e) => {
  e.preventDefault();
  send();
});

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    send();
  }
});

function autosize() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, window.innerHeight * 0.4) + "px";
}
input.addEventListener("input", autosize);

// ─── settings drawer ───────────────────────────────────────────────

const settingsBtn = document.getElementById("settings-btn");
const settingsPanel = document.getElementById("settings");
const settingsBackdrop = document.getElementById("settings-backdrop");
const settingsClose = document.getElementById("settings-close");
const settingsSpinnerSel = document.getElementById("settings-spinner");
const settingsPreview = document.getElementById("settings-spinner-preview");
const settingsCollapseReasoning = document.getElementById("settings-collapse-reasoning");

let previewStop = null;

function startPreview() {
  if (previewStop) previewStop();
  previewStop = startSpinner(settingsPreview, `${settings.spinner}`);
}

function openSettings() {
  settingsSpinnerSel.replaceChildren(...Object.keys(SPINNERS).map((k) => makeOption(k)));
  settingsSpinnerSel.value = settings.spinner;
  settingsCollapseReasoning.checked = !!settings.collapseReasoning;
  startPreview();
  renderCapsDisplay();
  settingsPanel.classList.remove("hidden");
  settingsBackdrop.classList.remove("hidden");
}

function closeSettings() {
  settingsPanel.classList.add("hidden");
  settingsBackdrop.classList.add("hidden");
  if (previewStop) { previewStop(); previewStop = null; }
}

settingsBtn.addEventListener("click", openSettings);
settingsClose.addEventListener("click", closeSettings);
settingsBackdrop.addEventListener("click", closeSettings);
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !settingsPanel.classList.contains("hidden")) closeSettings();
});
settingsSpinnerSel.addEventListener("change", () => {
  settings.spinner = settingsSpinnerSel.value;
  saveSettings();
  startPreview();
});
settingsCollapseReasoning.addEventListener("change", () => {
  settings.collapseReasoning = settingsCollapseReasoning.checked;
  saveSettings();
  // Apply immediately to any reasoning blocks already on screen.
  document.querySelectorAll(".msg.assistant .reasoning").forEach((el) => {
    el.open = !settings.collapseReasoning;
  });
});

// ─── settings: capability re-probe ────────────────────────────────

const capsDisplay = document.getElementById("settings-caps-display");
const reprobeBtn = document.getElementById("settings-reprobe");

function renderCapsDisplay() {
  const model = modelSel.value;
  capsDisplay.replaceChildren();
  if (!model) {
    const hint = document.createElement("span");
    hint.className = "settings-hint";
    hint.textContent = "No model selected";
    capsDisplay.appendChild(hint);
    return;
  }
  const caps = capabilities[model];
  if (!caps) {
    const hint = document.createElement("span");
    hint.className = "settings-hint";
    hint.textContent = `${model} — not yet probed`;
    capsDisplay.appendChild(hint);
    return;
  }
  const rows = [
    ["model", model],
    ["tool calling", String(caps.tool_calling)],
    ["vision", caps.vision ? "yes" : "no"],
    ["audio", caps.audio ? "yes" : "no"],
    ["reasoning", caps.reasoning ? "yes" : "no"],
    ["last probed", caps.last_probed || "—"],
    ["probe count", String(caps.probe_count ?? 1)],
  ];
  for (const [k, v] of rows) {
    const row = document.createElement("div");
    row.className = "caps-row";
    const key = document.createElement("span");
    key.className = "caps-key";
    key.textContent = k;
    const val = document.createElement("span");
    val.className = "caps-val";
    val.textContent = v;
    row.append(key, val);
    capsDisplay.appendChild(row);
  }
}

reprobeBtn.addEventListener("click", async () => {
  const model = modelSel.value;
  if (!model || probing) return;
  reprobeBtn.disabled = true;
  reprobeBtn.textContent = "probing...";
  delete capabilities[model];
  refreshModelDropdown();
  await probeModelIfNeeded(model);
  renderCapsDisplay();
  reprobeBtn.disabled = false;
  reprobeBtn.textContent = "Re-probe selected model";
});

const probeAllBtn = document.getElementById("settings-probe-all");
probeAllBtn.addEventListener("click", async () => {
  if (probing) return;
  if (!confirm("Wipe the capability cache and re-probe all installed Ollama models? Takes a few seconds.")) return;
  probeAllBtn.disabled = true;
  probeAllBtn.textContent = "probing all...";
  const status = appendMsg("system", "");
  const stopSpin = startSpinner(status, "re-probing all models via /api/show metadata...");
  try {
    const r = await fetch("/api/capabilities/probe-all", { method: "POST" });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    capabilities = await r.json();
    refreshModelDropdown();
    renderCapsDisplay();
    stopSpin();
    status.parentElement.remove();
    appendMsg("system", `re-probed ${Object.keys(capabilities).length} models`);
  } catch (e) {
    stopSpin();
    status.textContent = ` probe-all failed: ${e.message}`;
    status.parentElement.classList.add("error");
  } finally {
    probeAllBtn.disabled = false;
    probeAllBtn.textContent = "Re-probe all models (wipe cache)";
  }
});

modelSel.addEventListener("change", renderCapsDisplay);

document.getElementById("settings-reset").addEventListener("click", () => {
  if (!confirm("Reset all settings (theme, spinner, model preference) to defaults? Workspace files are not affected.")) return;
  localStorage.removeItem(SETTINGS_KEY);
  localStorage.removeItem(STORAGE.theme);
  localStorage.removeItem(STORAGE.model);
  location.reload();
});

bootstrap().then(handleSharedParam);

// ─── cellc (CellScript) check affordance — Phase 1, human-driven ──────
(function cellcAffordance() {
  let cellcEnabled = false;

  async function refreshCellcStatus() {
    try {
      const r = await fetch("/api/cellc/status");
      cellcEnabled = r.ok && (await r.json()).available === true;
    } catch { cellcEnabled = false; }
    document.body.classList.toggle("cellc-enabled", cellcEnabled);
  }

  function renderCheckResult(data) {
    if (data.tool_error) return `cellc error: ${data.stderr || "unknown"}`;
    if (data.ok) return "✓ cellc check passed";
    const lines = (data.diagnostics || []).map(
      (d) => `  L${d.line}:C${d.column} [${d.code || "—"}] ${d.message}`
    );
    let out = `✗ ${data.error_count} error(s)\n` + lines.join("\n");
    if (data.truncated) out += `\n  …(+${data.truncated} more)`;
    return out;
  }

  async function checkSource(source) {
    if (!cellcEnabled) return "cellc is not available on this server.";
    try {
      const r = await fetch("/api/cellc/check", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source }),
      });
      if (r.status === 503) return "cellc is not available on this server.";
      if (!r.ok) return `cellc check failed: HTTP ${r.status}`;
      return renderCheckResult(await r.json());
    } catch (e) { return `cellc check failed: ${e}`; }
  }

  // Expose for the slash-command handler + fence buttons to call.
  window.cellc = { checkSource, refreshCellcStatus, get enabled() { return cellcEnabled; } };
  refreshCellcStatus();
})();
