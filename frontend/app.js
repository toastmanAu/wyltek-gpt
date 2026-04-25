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

const STORAGE = { theme: "lcb.theme", model: "lcb.model" };
const SESSION = "default";  // multi-session support arrives in v1.2

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
const DEFAULT_SETTINGS = { spinner: "braille" };
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
let pending = null;  // {name, size}

function makeOption(value, label = value) {
  const opt = document.createElement("option");
  opt.value = value;
  opt.textContent = label;
  return opt;
}

async function bootstrap() {
  const cfg = await fetch("/api/config").then((r) => r.json());
  history = [{ role: "system", content: cfg.system_prompt }];

  const themes = await fetch("/api/themes").then((r) => r.json());
  themeSel.replaceChildren(...themes.map((t) => makeOption(t)));
  const savedTheme = localStorage.getItem(STORAGE.theme) || cfg.default_theme;
  setTheme(themes.includes(savedTheme) ? savedTheme : themes[0]);
  themeSel.value = document.documentElement.dataset.theme;

  converters = await fetch("/api/converters").then((r) => r.json());
  if (converters.disabled.length) {
    appendMsg(
      "system",
      `${converters.enabled.length} converters enabled, ${converters.disabled.length} skipped (missing: ${converters.disabled.map((d) => d.missing).join(", ")})`,
    );
  } else {
    appendMsg("system", `${converters.enabled.length} converters ready`);
  }

  modelSel.replaceChildren();
  try {
    const models = await fetch("/api/models").then((r) => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.json();
    });
    if (!models.length) {
      appendMsg("error", "Ollama is up but no models are installed. Run `ollama pull <name>`.");
      return;
    }
    modelSel.replaceChildren(...models.map((m) => makeOption(m)));
    const savedModel = localStorage.getItem(STORAGE.model);
    if (savedModel && models.includes(savedModel)) modelSel.value = savedModel;
  } catch (e) {
    appendMsg("error", `Could not list models: ${e.message}`);
  }
}

function setTheme(name) {
  document.documentElement.setAttribute("data-theme", name);
  document.getElementById("theme-link").href = `/static/themes/${name}.css`;
  localStorage.setItem(STORAGE.theme, name);
}

themeSel.addEventListener("change", () => setTheme(themeSel.value));
modelSel.addEventListener("change", () => localStorage.setItem(STORAGE.model, modelSel.value));

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
    showTray(meta);
  } catch (e) {
    stopSpinner();
    placeholder.textContent = ` upload failed: ${e.message}`;
    placeholder.parentElement.classList.add("error");
    hideTray();
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

  history.push({ role: "user", content: text });
  appendMsg("user", text);
  const out = appendMsg("assistant", "");
  out.classList.add("streaming");
  out.parentElement.classList.add("streaming");

  streaming = true;
  sendBtn.disabled = true;

  let acc = "";
  try {
    const r = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: modelSel.value, messages: history }),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const reader = r.body.getReader();
    const dec = new TextDecoder();
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      acc += dec.decode(value, { stream: true });
      out.textContent = ` ${acc}`;
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }
    history.push({ role: "assistant", content: acc });
  } catch (e) {
    out.textContent = ` [error: ${e.message}]`;
    out.parentElement.classList.add("error");
  } finally {
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

let previewStop = null;

function startPreview() {
  if (previewStop) previewStop();
  previewStop = startSpinner(settingsPreview, `${settings.spinner}`);
}

function openSettings() {
  settingsSpinnerSel.replaceChildren(...Object.keys(SPINNERS).map((k) => makeOption(k)));
  settingsSpinnerSel.value = settings.spinner;
  startPreview();
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

document.getElementById("settings-reset").addEventListener("click", () => {
  if (!confirm("Reset all settings (theme, spinner, model preference) to defaults? Workspace files are not affected.")) return;
  localStorage.removeItem(SETTINGS_KEY);
  localStorage.removeItem(STORAGE.theme);
  localStorage.removeItem(STORAGE.model);
  location.reload();
});

bootstrap().then(handleSharedParam);
