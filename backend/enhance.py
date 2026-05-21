"""Image-prompt strengthener — runs in-process against the user's selected
chat model. No second model load, no extra config dependency on
open-palette's ``prompt_optimizer`` toggle.

Four branches dispatched by hybrid rule:

1. Explicit ``mode`` in the call wins (e.g. user asks for "a pixel-art
   version of this Flux image" → chat model passes ``mode="pixel"``).
2. Otherwise, sniff the open-palette ``model`` name for known substrings
   (``flux`` / ``sd3`` → flux, ``pixel`` → pixel, ``anime`` / ``pony`` /
   ``animagine`` / ``noobai`` → anime).
3. Default to SDXL.

The system prompts are deliberately strict about preserving user intent
(named subjects/places/styles stay verbatim; no invented colours or
backgrounds). The Y/N/E confirm card in the chat UI gives the user a
last sanity check before the GPU job runs.
"""
from __future__ import annotations

import json
import logging
import re

import httpx

log = logging.getLogger(__name__)


# ─── System prompts — the security/quality seam, hand-tuned ───────────

ENHANCE_SDXL_PROMPT = """You are an image-prompt strengthener for Stable Diffusion / SDXL (CLIP encoder).

PRESERVE the user's intent exactly:
- Keep every subject, character, place, named artist, named style, named IP — verbatim.
- If the user named a colour, background, mood, or composition, keep it.
- Do NOT change the subject. Do NOT add or remove characters.

INVENTION (don't):
- Do NOT invent colours the user didn't name.
- Do NOT invent backgrounds, settings, or environments the user didn't name.
- Do NOT add named artists, named styles, named IPs, or named places the user didn't name.
- No "Greg Rutkowski", no "Studio Ghibli", no "in the style of …" unless the user said it.

ALLOWED (use sparingly, only when they sharpen intent):
- Composition: close-up, wide shot, profile, three-quarter view
- Lighting: soft, dramatic, backlit, golden hour, overcast
- One quality descriptor: cinematic, detailed, sharp focus

Format: comma-separated phrases. No full sentences. Cap ~75 tokens.
Negative prompt: cover common artifacts (blurry, distorted anatomy, extra limbs, watermark, text, low quality).

Respond ONLY with JSON, no fences:
{"enhanced_prompt": "...", "negative_prompt": "...", "changes": "one short line"}"""


ENHANCE_FLUX_PROMPT = """You are an image-prompt strengthener for Flux / SD3 (T5-XXL encoder, natural-language friendly).

PRESERVE the user's intent exactly:
- Keep every subject, character, place, named artist, named style, named IP — verbatim.
- If the user named a colour, background, mood, or composition, keep it.
- Do NOT change the subject. Do NOT add or remove characters.

INVENTION (don't):
- Do NOT invent colours the user didn't name.
- Do NOT invent backgrounds, settings, or environments the user didn't name.
- Do NOT add named artists, named styles, named IPs, or named places the user didn't name.

ALLOWED (use sparingly, only when they sharpen intent):
- Composition: close-up, wide shot, profile, three-quarter view
- Lighting: soft, dramatic, backlit, golden hour
- One quality descriptor: cinematic, detailed, photorealistic OR illustrated (pick one)

Format: ONE OR TWO natural-language sentences (T5 rewards this). Cap ~150 tokens.
Negative prompt: short, covering common artifacts.

Respond ONLY with JSON, no fences:
{"enhanced_prompt": "...", "negative_prompt": "...", "changes": "one short line"}"""


ENHANCE_PIXEL_PROMPT = """You are enhancing a pixel-art sprite prompt for SDXL with a pixel-art LoRA or model.

PRESERVE the user's subject, characters, named IPs/franchises, named view angle,
named colour palette, and named background — verbatim.

ALLOWED (intrinsic to pixel art — these are style fundamentals, not inventions):
- View angle if the user didn't pick one: side-view default, top-down for game-world objects.
- "transparent background" unless the user named a background.
- "single sprite, centered" for framing clarity.
- Style tokens: 16-bit, retro game asset, sprite, clean outline, limited palette.

INVENTION (don't):
- Do NOT invent specific colours the user didn't name.
- Do NOT invent mood, story, or environment.
- Do NOT add named franchises (no "Castlevania", "Final Fantasy", etc.) unless the user named them.

Format: comma-separated phrases. Cap ~60 tokens.
Negative prompt: blurry, anti-aliased, photorealistic, 3D render, gradient, smooth shading, text, watermark.

Respond ONLY with JSON, no fences:
{"enhanced_prompt": "...", "negative_prompt": "...", "changes": "one short line"}"""


ENHANCE_ANIME_PROMPT = """You are enhancing a prompt for an anime model (Animagine, Pony, NoobAI, anime-SDXL etc).
These models are trained on booru-style tag soup — comma-separated tags beat sentences.

PRESERVE every subject, character name, franchise, booru tag (1girl, solo, etc.),
named hair / eye / outfit colour, and named pose — verbatim.

ALLOWED:
- Composition: close-up, full-body, three-quarter view, from above, from below
- Lighting: soft lighting, dramatic lighting, backlit
- Trained quality tokens: masterpiece, best quality, detailed
  (these are real training tokens these models respond to, not generic flattery)
- "anime style, cel shading" if the user implied anime but didn't tag it

INVENTION (don't):
- Do NOT invent hair colour, eye colour, outfit colour, or background not named.
- Do NOT invent character traits, franchise tags, or named artists.
- Do NOT add NSFW tags ever — that's a user choice, never the enhancer's.

Format: comma-separated booru-style tags. Cap ~75 tokens.
Negative prompt: lowres, bad anatomy, extra fingers, missing fingers, text, watermark, signature, blurry.

Respond ONLY with JSON, no fences:
{"enhanced_prompt": "...", "negative_prompt": "...", "changes": "one short line"}"""


_PROMPTS: dict[str, str] = {
    "sdxl": ENHANCE_SDXL_PROMPT,
    "flux": ENHANCE_FLUX_PROMPT,
    "pixel": ENHANCE_PIXEL_PROMPT,
    "anime": ENHANCE_ANIME_PROMPT,
}


# ─── Branch detection ─────────────────────────────────────────────────


_FLUX_HINTS = ("flux", "sd3")
_PIXEL_HINTS = ("pixel", "sprite")
_ANIME_HINTS = ("anime", "pony", "animagine", "noobai", "waifu", "illustrious")


def detect_mode(model_name: str | None, explicit_mode: str | None = None) -> str:
    """Hybrid dispatch: explicit mode wins, else sniff model name, else SDXL.

    Returns one of ``sdxl``, ``flux``, ``pixel``, ``anime``.
    """
    if explicit_mode and explicit_mode in _PROMPTS:
        return explicit_mode
    name = (model_name or "").lower()
    if any(h in name for h in _FLUX_HINTS):
        return "flux"
    if any(h in name for h in _PIXEL_HINTS):
        return "pixel"
    if any(h in name for h in _ANIME_HINTS):
        return "anime"
    return "sdxl"


# ─── Ollama call ──────────────────────────────────────────────────────


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


async def enhance_prompt(
    *,
    ollama_url: str,
    chat_model: str,
    user_prompt: str,
    image_model: str | None = None,
    explicit_mode: str | None = None,
    timeout: float = 120.0,
) -> dict:
    """Strengthen a user image prompt using their selected chat model.

    Returns a dict with keys:
        ``original_prompt``, ``enhanced_prompt``, ``negative_prompt``,
        ``changes``, ``mode``.

    On any failure (network, parse, missing keys), returns the original
    prompt unchanged with ``mode="passthrough"`` and a populated
    ``error`` field. Never raises — image gen should still work even if
    enhancement is offline.
    """
    mode = detect_mode(image_model, explicit_mode)
    system_prompt = _PROMPTS[mode]

    fallback = {
        "original_prompt": user_prompt,
        "enhanced_prompt": user_prompt,
        "negative_prompt": "",
        "changes": "",
        "mode": mode,
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(
                f"{ollama_url}/api/generate",
                json={
                    "model": chat_model,
                    "system": system_prompt,
                    "prompt": user_prompt,
                    "stream": False,
                    "options": {"temperature": 0.3, "num_gpu": -1},
                },
            )
        if r.status_code != 200:
            return {**fallback, "error": f"ollama HTTP {r.status_code}", "mode": "passthrough"}
        body = r.json().get("response", "")
        match = _JSON_BLOCK_RE.search(body)
        if not match:
            return {**fallback, "error": "no JSON in model response", "mode": "passthrough"}
        parsed = json.loads(match.group())
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        return {**fallback, "error": str(exc), "mode": "passthrough"}

    return {
        "original_prompt": user_prompt,
        "enhanced_prompt": str(parsed.get("enhanced_prompt") or user_prompt).strip(),
        "negative_prompt": str(parsed.get("negative_prompt") or "").strip(),
        "changes": str(parsed.get("changes") or "").strip(),
        "mode": mode,
    }
