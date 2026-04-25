"""Converter registry — config-driven argv templates with safe execution.

Design notes:
- The model never assembles commands. argv lives in config.yaml as a list,
  the runner only fills named slots ({input}, {output}, {out_dir}, {stem},
  plus any user-selected params declared by the converter).
- Missing binaries are soft-skipped at startup, not at request time, so the
  enabled converter list the model/UI sees is always the truth.
- User-supplied param values are validated against a per-converter
  whitelist (the `choices` list); anything else falls back to the declared
  default. Same allowlist principle as everywhere else in the system.
- Every input/output path is resolved and checked against the session
  workspace root before subprocess runs.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class Converter:
    id: str
    sources: tuple[str, ...]
    targets: tuple[str, ...]
    requires: str
    argv_template: tuple[str, ...]
    timeout: int = 120
    params: dict[str, dict] = field(default_factory=dict)

    def supports(self, src_ext: str, tgt_ext: str) -> bool:
        return src_ext.lower() in self.sources and tgt_ext.lower() in self.targets


class Registry:
    def __init__(self, entries: list[dict]):
        self._enabled: list[Converter] = []
        self._missing: list[tuple[str, str]] = []  # (id, missing-binary)

        for e in entries:
            conv = Converter(
                id=e["id"],
                sources=tuple(s.lower() for s in _as_list(e["from"])),
                targets=tuple(t.lower() for t in _as_list(e["to"])),
                requires=e["requires"],
                argv_template=tuple(e["argv"]),
                timeout=int(e.get("timeout", 120)),
                params=dict(e.get("params") or {}),
            )
            if shutil.which(conv.requires):
                self._enabled.append(conv)
            else:
                self._missing.append((conv.id, conv.requires))
                log.warning(
                    "converter '%s' disabled — missing binary '%s'",
                    conv.id, conv.requires,
                )

    @property
    def enabled(self) -> list[Converter]:
        return list(self._enabled)

    @property
    def missing(self) -> list[tuple[str, str]]:
        return list(self._missing)

    def reachable_from(self, src_ext: str) -> set[str]:
        src_ext = src_ext.lower()
        out: set[str] = set()
        for c in self._enabled:
            if src_ext in c.sources:
                out.update(c.targets)
        out.discard(src_ext)  # never offer same-format conversion
        return out

    def find(self, src_ext: str, tgt_ext: str) -> Converter | None:
        if src_ext.lower() == tgt_ext.lower():
            return None  # would overwrite input
        for c in self._enabled:
            if c.supports(src_ext, tgt_ext):
                return c
        return None


def run_conversion(
    conv: Converter,
    input_path: Path,
    workspace: Path,
    target_ext: str,
    user_params: dict[str, str] | None = None,
) -> Path:
    """Execute one conversion. Returns absolute path to output file.

    Raises:
        ValueError on path escape or template error
        RuntimeError on non-zero exit or missing output
        subprocess.TimeoutExpired on timeout
    """
    input_path = input_path.resolve()
    workspace = workspace.resolve()
    _assert_inside(input_path, workspace)

    out_dir = workspace
    stem = input_path.stem
    output_path = (out_dir / f"{stem}.{target_ext}").resolve()
    _assert_inside(output_path, workspace)

    slots: dict[str, str] = {
        "input": str(input_path),
        "output": str(output_path),
        "out_dir": str(out_dir),
        "stem": stem,
    }

    # Validated params: only values from the declared `choices` list are
    # accepted; anything else falls back to the converter's default.
    user_params = user_params or {}
    for name, defn in conv.params.items():
        valid = {str(c["value"]) for c in defn.get("choices", [])}
        chosen = str(user_params.get(name, ""))
        slots[name] = chosen if chosen in valid else str(defn.get("default", ""))

    argv = [_substitute(s, slots) for s in conv.argv_template]

    log.info("converting %s → .%s via %s (params=%s)",
             input_path.name, target_ext, conv.id,
             {k: slots[k] for k in conv.params} if conv.params else "{}")
    result = subprocess.run(  # noqa: S603 — argv list, never shell=True
        argv,
        capture_output=True,
        text=True,
        timeout=conv.timeout,
        check=False,
    )
    if result.returncode != 0:
        tail = (result.stderr or result.stdout).strip()[-500:]
        raise RuntimeError(f"{conv.id} exit {result.returncode}: {tail}")

    if output_path.exists():
        return output_path

    # LibreOffice and a few others ignore explicit output paths and just
    # write {stem}.{ext} into --outdir. Check for that before failing.
    fallback = out_dir / f"{stem}.{target_ext}"
    if fallback.exists():
        return fallback.resolve()

    raise RuntimeError(f"{conv.id} produced no output at {output_path}")


def _as_list(v):
    return v if isinstance(v, list) else [v]


def _substitute(template: str, slots: dict[str, str]) -> str:
    try:
        return template.format(**slots)
    except KeyError as e:
        raise ValueError(f"unknown slot {e} in argv template '{template}'")


def _assert_inside(path: Path, root: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError:
        raise ValueError(f"path escapes workspace: {path}")
