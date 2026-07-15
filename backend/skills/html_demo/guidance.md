# Self-contained HTML demo — house style

You are building a SINGLE self-contained `.html` file. Follow these rules,
then call `preview_demo` to render and test it. Fix using `patch_demo`.

## Self-contained (non-negotiable)
- One file. Inline ALL CSS in `<style>` and ALL JS in `<script>`.
- No CDNs, no external `<script src>`, no web-fonts, no remote images.
- Embed any asset as a `data:` URI. Must work fully offline.
- Start with `<!doctype html>` and a `<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">`.

## Physics & animation
- Use one `requestAnimationFrame` loop. Track delta-time:
  `let last=performance.now(); function frame(t){ const dt=(t-last)/1000; last=t; /*update(dt); draw();*/ requestAnimationFrame(frame);} requestAnimationFrame(frame);`
- Integrate with dt (position += velocity*dt). Clamp dt (~0.05) to survive tab stalls.
- Simple collisions: AABB overlap or circle distance; resolve by reflecting velocity.
- Ease with `t*t*(3-2*t)` (smoothstep) rather than linear when it reads better.

## Layout
- Prefer `100dvh` over `100vh` (mobile browser chrome). Use `min-height:100dvh`.
- Respect notches: `padding: env(safe-area-inset-top) env(safe-area-inset-right) ...`.
- Use flexbox/grid; avoid fixed pixel widths that overflow small screens.
- Size a full-bleed `<canvas>` to `devicePixelRatio` for crispness.

## Touch (assume a phone)
- Use Pointer Events, not mouse-only: `pointerdown/pointermove/pointerup`.
- Set `touch-action: none` on interactive canvases to stop scroll-hijack.
- `e.preventDefault()` on gesture handlers; test taps and drags.

## Design
- Dark-mode aware: honour `@media (prefers-color-scheme: dark)`.
- Consistent spacing scale (4/8/16/24) and a small type scale.
- Motion with purpose — animate state changes, keep durations 150–300ms.

## Loop discipline (SAVES YOUR CONTEXT)
- Send the FULL html to `preview_demo` ONCE. It returns a `demo_id`.
- To fix, call `patch_demo(demo_id, edits=[{find, replace}])` with SMALL unique
  find-strings — do NOT resend the whole file. If a patch fails, only then
  resend full html via `preview_demo` reusing the same demo_id.
- Read `console_errors`, `exceptions`, and `raf_ran` before editing.
- When it works, call `save_demo(demo_id)`.
