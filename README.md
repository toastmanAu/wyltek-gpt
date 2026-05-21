# wyltek-gpt

Mobile-first browser terminal for local Ollama models — with a 25-converter
multi-format file pipeline, PWA install, Android share-target, skinnable
themes, and a settings drawer with terminal-style spinners.

Built to be a daily-driver local inference UI you can install on your phone
via Tailscale and use to convert files between formats without leaving the
browser.

![A pixel-art llama in a teal shield](frontend/icons/icon-192.png)

## Quickstart

```bash
git clone https://github.com/<you>/wyltek-gpt.git
cd wyltek-gpt

# Smart installer: audits what's installed, prompts before adding what's missing.
./install.sh

# Make sure Ollama has at least one model:
#   ollama pull qwen2.5:7b

./.venv/bin/uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` from the same machine, or expose to your phone
via Tailscale (see [Installing as a PWA](#installing-as-a-pwa-on-your-phone)
below for the HTTPS requirement).

## What it does

- **Chat with any local Ollama model** — model dropdown, streaming output,
  per-request system prompt assembled with live host facts (current date,
  OS, hostname) so models don't hallucinate about your environment
- **Convert files between 25 formats** — `markitdown` for "anything → md",
  `libreoffice` for office → pdf, `ffmpeg` per-target for video (with
  faststart for browser-streamable mp4), `whisper` for audio → text,
  `pandoc` / `weasyprint` / `calibre` for documents and ebooks, `vips` /
  `cwebp` / `rsvg-convert` for images. Quality params per converter where
  it makes sense.
- **Mobile share-target** (Android) — long-press a file in any app, share
  to wyltek-gpt, the conversion tray pre-loads with target formats filtered
  to what's actually possible
- **Skinnable** — five built-in themes (tokyo-night, catppuccin-mocha, nord,
  classic-green, custom). Drop a new `.css` file in `frontend/themes/` and
  it shows up in the picker on next reload.
- **Indicatif-style spinners** for loading states, configurable in the
  settings drawer (braille, dots, line, bar, arrow, bounce, pulse, moon,
  clock — all rendered as Unicode, no Rust required)

## Two things to fill in

There are two `TODO(phill)` markers in the project:

1. **`config.yaml` → `assistant.system_prompt`** — defines voice and
   default behavior. Replace the placeholder with something that reflects
   how you want the assistant to talk.
2. **`frontend/themes/custom.css`** — your personal accent. Ships with a
   deliberately-wrong orange so you spot it the first time and know what
   to change.

## Theming

Themes are pure CSS files under `frontend/themes/`. Each defines a single
`[data-theme="<name>"] { ... }` block of CSS custom properties. To add one:

1. Copy `frontend/themes/custom.css` to `frontend/themes/<your-name>.css`
2. Edit the tokens
3. Reload — the dropdown picks it up automatically

| Token | Controls |
|-------|----------|
| `--bg-0`, `--bg-1` | Page background, elevated surfaces (header, composer) |
| `--fg-0`, `--fg-1` | Main text, muted text |
| `--accent`, `--accent-contrast` | Send button, focus rings, role tags |
| `--border` | All borders and dividers |
| `--role-user`, `--role-assistant` | The `user>` / `assistant>` prefix colors |
| `--error` | Error messages |
| `--font-mono` | Font stack for everything |

## Installing as a PWA on your phone

Once installed, the app gets a home-screen icon, runs without browser chrome,
and (on Android) registers as a share target so you can send files from any
app's share sheet.

### Critical: PWAs require a secure context

Service workers (which power install + share target) only work over HTTPS —
except `localhost`/`127.0.0.1`, which are exempt. Plain LAN IPs like
`http://192.168.x.x:8000` will *silently fail* to register the SW, and "Add
to Home Screen" won't appear.

### The Tailscale path (recommended)

Tailscale Serve gives you HTTPS automatically with a `*.ts.net` cert,
visible only inside your tailnet:

```bash
# On the host running uvicorn:
uvicorn backend.app:app --host 127.0.0.1 --port 8000
tailscale serve --bg --https=443 http://127.0.0.1:8000
tailscale serve status        # shows the URL to open on your phone
```

Open the resulting `https://<machine>.<tailnet>.ts.net` URL on your phone.
The browser will offer **Install app** / **Add to Home Screen**.

### iOS specifics

- Open in Safari → Share → "Add to Home Screen" → done.
- iOS does **not** support Web Share Target. Workaround: build an iOS
  Shortcut that POSTs to `/api/upload` — quick to set up, link from your
  home screen.

### Android specifics

- Chrome surfaces the install prompt automatically once SW registers.
- After install, wyltek-gpt appears in the share sheet of any app that
  exports files. Long-press → Share → wyltek-gpt → tray pre-loads with
  conversion options filtered to what's possible.

## Configuration

Everything lives in `config.yaml`:

- `ollama.url` — usually `http://localhost:11434`; override for remote
  Ollama (e.g. via tailnet)
- `server.host` / `server.port` — bind address; `127.0.0.1` for local-only,
  `0.0.0.0` for LAN
- `ui.default_theme` — initial theme on first visit
- `assistant.system_prompt` — base prompt; host facts (current date, OS,
  hostname) are appended automatically per request
- `converters` — array of 25 entries, each declaring source/target
  extensions, the binary it requires, the argv template, and optional
  user-tunable params (quality, bitrate, model size, etc)

## Security model

- The model **never** assembles shell commands. argv is fixed in config;
  user-supplied params are validated against a per-converter whitelist.
- All file paths are resolved with `Path.resolve()` and checked against the
  session workspace root before any subprocess runs.
- `subprocess.run([...], shell=False)` everywhere. No `shell=True` ever.
- Filenames at the upload boundary are normalised with `Path(name).name` to
  prevent path traversal.
- Missing converter binaries are soft-skipped at startup (logged warning),
  so the enabled list shown to the model/UI is always the truth.

## Model capability manifest

wyltek-gpt probes every Ollama model the first time you select it and records
what it can do. The result drives capability glyphs in the model dropdown
and an in-card warning when a model is asked to do something it lacks the
capability for. Cache lives at `data/capabilities.json`.

### Glyph language

| Glyph | Capability | Meaning |
|-------|-----------|---------|
| ⚒ | tool calling (native) | Model emits valid `tool_calls` when given a `tools=[]` parameter |
| ⚙ | tool calling (ignored) | Model accepts the `tools=[]` parameter but doesn't invoke anything — falls back to the structured `op:<name>` fence path |
| 👁 | vision | Model accepts inline base64 images and produces sensible descriptions |
| 🎙 | audio | Model accepts audio inputs (always false for Ollama-served models — Ollama's chat API doesn't accept audio) |
| 🧠 | reasoning | Model emits `<think>` blocks during its response (Qwen3 distill, DeepSeek R1, etc.) |
| ❓ | unknown | Not yet probed — first selection triggers the probe |
| · | probed, none of the above | Pure text-output model, no detected capabilities beyond chat |

### How probing works

1. Probes are **lazy** — the first time you pick a model in the dropdown, the
   backend runs three probes against it (tool calling, vision, reasoning).
   Audio is hardcoded `false` because Ollama's chat API doesn't accept it.
2. Each probe is a single chat request to Ollama with a minimal prompt
   tailored to that capability axis.
3. Results are cached in `data/capabilities.json` and never re-probed
   automatically — use the **Re-probe selected model** button in the
   settings drawer after `ollama pull`-ing an updated model.
4. Frontend shows a "first-run capability probe" notification with a
   spinner and elapsed-seconds counter while a probe is in flight.

### Operations declare what they need

Each operation in `config.yaml` has a `capabilities_required` list:

```yaml
operations:
  - id: trim_video
    capabilities_required: [text]      # universal — every model satisfies

  - id: caption_image                  # hypothetical
    capabilities_required: [text, vision]
```

When you ask the model to invoke an op whose `capabilities_required` exceeds
the selected model's capabilities, the confirm card shows a yellow warning
strip listing what's missing and which probed models DO have it. **Y still
runs the op** — the gate is warn-and-allow, not block-and-redirect, because
some failures are environmental and the user gets to decide.

## Roadmap

Done:
- [x] Mobile-first chat UI with streaming
- [x] Ollama model dropdown, host-facts injection in system prompt
- [x] Skinnable theme system (5 built-in)
- [x] File upload + Convert tray with format-aware target dropdown
- [x] 25 converters across documents/ebooks/images/audio/video/transcription
- [x] Per-converter quality params (CRF, bitrate, model size, fps, etc)
- [x] PWA install + Android Web Share Target
- [x] Settings drawer with spinner picker + theme selector
- [x] Smart `install.sh` (Ubuntu/Debian/Fedora/Arch/Brew)

Next:
- [ ] Multi-file batch upload + bulk convert
- [ ] Conversion presets (saved recipes with pinned flags)
- [ ] Workspace browser (manage all uploaded/converted files)
- [ ] Persistent chat history (SQLite)
- [ ] Skill registry + filesystem.read / filesystem.write
- [ ] Restricted shell tool
- [ ] Auth (token) + LAN/Tailscale hardening
- [ ] iOS Shortcut for share-to-upload (workaround for missing Web Share Target)

## License

MIT — see [LICENSE](LICENSE).
