"""Operations registry — model-invoked file editing + generation.

Sister module to ``converters.py``. Two important differences from converters:

* Operations are invoked by the **model**, not the user. The chat model emits
  either an Ollama-native tool call OR a structured ``op:<name> {json}`` block,
  the backend parses it, validates every parameter, then runs it.
* Operations may produce same-format outputs (trim mp4 → mp4, scale png → png).
  Each operation declares its own output extension policy.

Two kinds of operations:

* ``local``  — runs a host binary via fixed argv template, slot-substituted
  with validated parameters. Same safety story as converters: the model
  never assembles shell, the binary list is closed, every slot is type-checked.
* ``bridge`` — proxied to an external service (currently ``open-palette``)
  declared in ``config.bridges``. The bridge module owns the wire protocol.

Adding a new local operation = one YAML entry. Adding a new bridge endpoint
= one YAML entry pointing at an existing bridge. Adding a new bridge =
new module under ``backend/bridges/`` plus a ``bridges:`` config block.
"""
from __future__ import annotations

import logging
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

log = logging.getLogger(__name__)


# ─── Slot validators ──────────────────────────────────────────────────
#
# The security-critical seam. Every model-supplied parameter passes
# through one of these before being placed into an argv template or HTTP
# form. Validators raise ValueError on bad input — the runner converts
# that into a 400 response back to the chat client.
#
# Add validator types here as new operation kinds need them.


class SlotValidator(Protocol):
    """Validates a single model-supplied parameter value."""

    def validate(self, raw: Any) -> str:
        """Return canonical string form, or raise ValueError."""

    def to_json_schema(self) -> dict:
        """JSON Schema fragment for the Ollama tools API."""


@dataclass(frozen=True)
class EnumValidator:
    choices: tuple[str, ...]
    description: str = ""

    def validate(self, raw: Any) -> str:
        s = str(raw)
        if s not in self.choices:
            raise ValueError(f"value {s!r} not in allowed choices {list(self.choices)}")
        return s

    def to_json_schema(self) -> dict:
        return {"type": "string", "enum": list(self.choices), "description": self.description}


@dataclass(frozen=True)
class IntRangeValidator:
    min: int
    max: int
    description: str = ""

    def validate(self, raw: Any) -> str:
        try:
            n = int(raw)
        except (TypeError, ValueError):
            raise ValueError(f"value {raw!r} is not an integer")
        if n < self.min or n > self.max:
            raise ValueError(f"value {n} outside range [{self.min}, {self.max}]")
        return str(n)

    def to_json_schema(self) -> dict:
        return {"type": "integer", "minimum": self.min, "maximum": self.max,
                "description": self.description}


@dataclass(frozen=True)
class FloatRangeValidator:
    min: float
    max: float
    description: str = ""

    def validate(self, raw: Any) -> str:
        try:
            n = float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"value {raw!r} is not a number")
        if n < self.min or n > self.max:
            raise ValueError(f"value {n} outside range [{self.min}, {self.max}]")
        return f"{n:g}"

    def to_json_schema(self) -> dict:
        return {"type": "number", "minimum": self.min, "maximum": self.max,
                "description": self.description}


# Matches HH:MM:SS, HH:MM:SS.mmm, MM:SS, MM:SS.mmm, or bare seconds (12.5).
# Rejects negative values and obviously-wrong shapes ("1:2:3:4", "abc").
_TIMESTAMP_RE = re.compile(
    r"^(?:\d+(?:\.\d+)?|\d{1,2}:[0-5]?\d(?:\.\d+)?|\d{1,3}:[0-5]?\d:[0-5]?\d(?:\.\d+)?)$"
)


@dataclass(frozen=True)
class TimestampValidator:
    """Accepts HH:MM:SS, HH:MM:SS.mmm, MM:SS, MM:SS.mmm, or bare seconds.

    ffmpeg accepts all of these natively, so we pass through the canonical
    string rather than normalising to seconds.
    """
    description: str = ""

    def validate(self, raw: Any) -> str:
        s = str(raw).strip()
        if not _TIMESTAMP_RE.fullmatch(s):
            raise ValueError(
                f"timestamp {s!r} not in HH:MM:SS, MM:SS, or seconds form"
            )
        return s

    def to_json_schema(self) -> dict:
        return {"type": "string", "pattern": _TIMESTAMP_RE.pattern,
                "description": self.description or
                "Timestamp as HH:MM:SS, MM:SS, or seconds (e.g. '00:01:30' or '90.5')"}


@dataclass(frozen=True)
class TextValidator:
    """Free-form text with a hard length cap — for prompts and similar."""
    max_len: int = 2000
    description: str = ""

    def validate(self, raw: Any) -> str:
        s = str(raw).strip()
        if not s:
            raise ValueError("text value is empty")
        if len(s) > self.max_len:
            raise ValueError(f"text length {len(s)} exceeds max {self.max_len}")
        return s

    def to_json_schema(self) -> dict:
        return {"type": "string", "maxLength": self.max_len,
                "description": self.description}


def _build_validator(spec: dict) -> SlotValidator:
    """Construct a validator from its YAML spec dict."""
    kind = spec.get("type")
    desc = spec.get("description", "")
    if kind == "enum":
        return EnumValidator(tuple(str(c) for c in spec["choices"]), desc)
    if kind == "int_range":
        return IntRangeValidator(int(spec["min"]), int(spec["max"]), desc)
    if kind == "float_range":
        return FloatRangeValidator(float(spec["min"]), float(spec["max"]), desc)
    if kind == "timestamp":
        return TimestampValidator(desc)
    if kind == "text":
        return TextValidator(int(spec.get("max_len", 2000)), desc)
    if kind == "dimension":
        # Convenience alias: enum of ints, declared in pixels.
        return EnumValidator(tuple(str(int(c)) for c in spec["choices"]), desc)
    raise ValueError(f"unknown param type {kind!r}")


# ─── Operation + Param ────────────────────────────────────────────────


@dataclass(frozen=True)
class Param:
    name: str
    validator: SlotValidator
    required: bool = False
    default: str | None = None


@dataclass
class Operation:
    id: str
    kind: str  # "local" | "bridge" | "converter"
    description: str
    params: tuple[Param, ...]

    # Capability gate — list of model capability keys this operation
    # depends on (e.g. ["text"], ["vision"]). Frontend warns (warn-and-allow,
    # not block) when the selected model lacks any of these. "text" is
    # universal and satisfied by every chat model — declared explicitly
    # so the contract is visible in YAML.
    capabilities_required: tuple[str, ...] = ("text",)

    # local-only
    requires: str | None = None  # binary on PATH (legacy field name, retained for back-compat)
    argv_template: tuple[str, ...] = ()
    source_param: str | None = None  # name of the file-input param
    accepts: tuple[str, ...] = ()    # allowed source extensions
    output_ext_from_source: bool = False
    output_ext: str | None = None
    timeout: int = 300

    # bridge-only
    bridge: str | None = None
    bridge_call: dict = field(default_factory=dict)
    enhance: bool = False  # if true, frontend should run prompt enhance first

    def to_tool_schema(self) -> dict:
        """Ollama-format tool definition for native function-calling."""
        properties: dict[str, dict] = {}
        required: list[str] = []
        for p in self.params:
            properties[p.name] = p.validator.to_json_schema()
            if p.required:
                required.append(p.name)
        return {
            "type": "function",
            "function": {
                "name": self.id,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def to_client_dict(self) -> dict:
        """Compact JSON-friendly view for the chat UI."""
        return {
            "id": self.id,
            "kind": self.kind,
            "description": self.description,
            "enhance": self.enhance,
            "source_param": self.source_param,
            "accepts": list(self.accepts),
            "capabilities_required": list(self.capabilities_required),
            "params": [
                {
                    "name": p.name,
                    "required": p.required,
                    "default": p.default,
                    "schema": p.validator.to_json_schema(),
                }
                for p in self.params
            ],
        }


# ─── Registry ─────────────────────────────────────────────────────────


class OperationRegistry:
    def __init__(self, entries: list[dict], bridges_available: set[str]):
        self._enabled: list[Operation] = []
        self._missing: list[tuple[str, str]] = []  # (op_id, reason)

        for e in entries:
            try:
                op = self._build(e)
            except (KeyError, ValueError) as exc:
                self._missing.append((e.get("id", "<unknown>"), f"config error: {exc}"))
                log.warning("operation '%s' disabled — config error: %s", e.get("id"), exc)
                continue

            if op.kind == "local" and not shutil.which(op.requires or ""):
                self._missing.append((op.id, f"missing binary '{op.requires}'"))
                log.warning("operation '%s' disabled — missing binary '%s'", op.id, op.requires)
                continue
            if op.kind == "bridge" and op.bridge not in bridges_available:
                self._missing.append((op.id, f"bridge '{op.bridge}' unavailable"))
                log.warning("operation '%s' disabled — bridge '%s' unavailable", op.id, op.bridge)
                continue

            self._enabled.append(op)

    @staticmethod
    def _build(entry: dict) -> Operation:
        params = tuple(
            Param(
                name=name,
                validator=_build_validator(spec),
                required=bool(spec.get("required", False)),
                default=str(spec["default"]) if "default" in spec else None,
            )
            for name, spec in (entry.get("params") or {}).items()
        )
        kind = entry["kind"]
        caps_req = tuple(str(c) for c in (entry.get("capabilities_required") or ["text"]))
        common = dict(
            id=entry["id"],
            kind=kind,
            description=entry.get("description", ""),
            params=params,
            capabilities_required=caps_req,
        )
        if kind == "local":
            return Operation(
                **common,
                requires=entry["requires"],
                argv_template=tuple(entry["argv"]),
                source_param=entry.get("source_param"),
                accepts=tuple(s.lower() for s in entry.get("accepts", [])),
                output_ext_from_source=bool(entry.get("output_ext_from_source", False)),
                output_ext=entry.get("output_ext"),
                timeout=int(entry.get("timeout", 300)),
            )
        if kind == "bridge":
            return Operation(
                **common,
                bridge=entry["bridge"],
                bridge_call=dict(entry.get("bridge_call") or {}),
                enhance=bool(entry.get("enhance", False)),
            )
        raise ValueError(f"unknown operation kind {kind!r}")

    @property
    def enabled(self) -> list[Operation]:
        return list(self._enabled)

    @property
    def missing(self) -> list[tuple[str, str]]:
        return list(self._missing)

    def find(self, op_id: str) -> Operation | None:
        for op in self._enabled:
            if op.id == op_id:
                return op
        return None

    def add(self, op: Operation) -> None:
        """Append a programmatically-built operation (e.g. the convert wrapper).

        Used for synthetic operations whose schema depends on runtime state
        like the converter registry — they can't be expressed in YAML."""
        self._enabled.append(op)

    def tool_schemas(self) -> list[dict]:
        return [op.to_tool_schema() for op in self._enabled]


def make_cellc_save_operation() -> Operation:
    """Synthetic op: write a model-supplied .cell source to the workspace.
    kind 'cellc_save' is handled specially in /api/operations/run (re-check + write)."""
    return Operation(
        id="cellc_save",
        kind="cellc_save",
        description=(
            "Save a checked CellScript contract to the workspace as <name>.cell. "
            "Only call after cellc_check passes; the backend re-checks and refuses "
            "to save a failing contract."
        ),
        capabilities_required=("text",),
        params=(
            Param(
                name="name",
                validator=TextValidator(max_len=64, description="File stem (letters, digits, _ or -)."),
                required=True,
            ),
            Param(
                name="source",
                validator=TextValidator(max_len=200_000, description="Full .cell source."),
                required=True,
            ),
        ),
        output_ext="cell",
    )


def make_converter_operation(reachable_targets: tuple[str, ...]) -> Operation:
    """Synthetic operation that wraps the converter registry — gives the
    model access to all 25 user-tray converters through one op call.

    The dispatch lives in ``app.py``: kind=converter is handled by looking
    up ``(source_ext, target_ext)`` in ``Registry.find()`` and running
    ``run_conversion()`` from converters.py. This op declares no argv
    template; runtime resolves it via the converter registry.
    """
    return Operation(
        id="convert",
        kind="converter",
        description=(
            "Convert an uploaded file to a different format. Use this whenever "
            "the user asks to convert, transcode, or change the format of a "
            "file they uploaded (e.g. webm → mp4, mp4 → mp3, docx → pdf, png "
            "→ jpg). Pass the uploaded filename verbatim as 'source' and the "
            "target file extension (no leading dot) as 'target'."
        ),
        capabilities_required=("text",),
        params=(
            Param(
                name="source",
                validator=TextValidator(max_len=300, description="Uploaded filename in the workspace."),
                required=True,
            ),
            Param(
                name="target",
                validator=EnumValidator(
                    reachable_targets,
                    description="Target file extension, e.g. mp4, mp3, pdf, jpg, png.",
                ),
                required=True,
            ),
        ),
        source_param="source",
    )


# ─── Validation + execution ───────────────────────────────────────────


def validate_params(op: Operation, raw: dict) -> dict[str, str]:
    """Validate every model-supplied parameter against its declared schema.

    Returns the canonical-string slot map. Raises ValueError on the first
    bad value — the message names the offending param so the model can
    self-correct on retry.
    """
    out: dict[str, str] = {}
    raw = raw or {}
    for p in op.params:
        if p.name in raw and raw[p.name] not in (None, ""):
            try:
                out[p.name] = p.validator.validate(raw[p.name])
            except ValueError as e:
                raise ValueError(f"param '{p.name}': {e}") from e
        elif p.required:
            raise ValueError(f"param '{p.name}' is required")
        elif p.default is not None:
            # Defaults are author-supplied YAML values — re-validate so a
            # bad default surfaces at request time, not in production silence.
            out[p.name] = p.validator.validate(p.default)
    return out


def run_local(
    op: Operation,
    workspace: Path,
    source_path: Path | None,
    validated: dict[str, str],
) -> Path:
    """Execute a ``kind: local`` operation. Returns absolute path to output."""
    workspace = workspace.resolve()
    if source_path is not None:
        source_path = source_path.resolve()
        _assert_inside(source_path, workspace)
        if op.accepts and source_path.suffix.lstrip(".").lower() not in op.accepts:
            raise ValueError(
                f"operation {op.id} does not accept .{source_path.suffix.lstrip('.')} input"
            )

    # Decide output extension + path.
    if op.output_ext_from_source:
        if source_path is None:
            raise ValueError(f"{op.id}: output_ext_from_source needs a source file")
        ext = source_path.suffix.lstrip(".")
    elif op.output_ext:
        ext = op.output_ext
    else:
        raise ValueError(f"{op.id}: no output extension policy declared")

    stem_base = source_path.stem if source_path else op.id
    output_path = _unique_output_path(workspace, stem_base, ext, op.id)
    _assert_inside(output_path, workspace)

    slots: dict[str, str] = {
        **validated,
        "output": str(output_path),
        "out_dir": str(workspace),
        "stem": stem_base,
    }
    if op.source_param and source_path is not None:
        slots[op.source_param] = str(source_path)
        slots["source"] = str(source_path)
    elif source_path is not None:
        slots["source"] = str(source_path)

    argv = [_substitute(s, slots) for s in op.argv_template]
    log.info("op %s: argv=%s", op.id, argv)

    result = subprocess.run(  # noqa: S603 — argv list, never shell=True
        argv,
        capture_output=True,
        text=True,
        timeout=op.timeout,
        check=False,
    )
    if result.returncode != 0:
        tail = (result.stderr or result.stdout).strip()[-500:]
        raise RuntimeError(f"{op.id} exit {result.returncode}: {tail}")
    if not output_path.exists():
        raise RuntimeError(f"{op.id} produced no output at {output_path}")
    return output_path


# ─── helpers ──────────────────────────────────────────────────────────


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


def _unique_output_path(workspace: Path, stem: str, ext: str, op_id: str) -> Path:
    """Avoid clobbering existing files by appending an op-id suffix and a counter."""
    base = f"{stem}.{op_id}"
    candidate = workspace / f"{base}.{ext}"
    n = 1
    while candidate.exists():
        candidate = workspace / f"{base}.{n}.{ext}"
        n += 1
    return candidate
