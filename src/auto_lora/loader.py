"""Resolve, load and apply matched LoRAs.

ComfyUI modules are imported lazily inside ``ComfyBackend`` so the rest of this
module can be unit tested with a fake backend.
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Protocol

from .matching import Match

log = logging.getLogger(__name__)

LOG_PREFIX = "[Auto LoRA Loader]"
# Loaded LoRA state dicts kept in RAM. Character LoRAs are ~100-250MB each.
DEFAULT_CACHE_SIZE = 4


class Backend(Protocol):
    def list_loras(self) -> list[str]: ...
    def full_path(self, name: str) -> str | None: ...
    def load_file(self, path: str) -> tuple[Any, Any]: ...
    def apply(self, model: Any, clip: Any, lora: Any, strength_model: float,
              strength_clip: float, metadata: Any) -> tuple[Any, Any]: ...


class ComfyBackend:
    """Thin adapter over the ComfyUI APIs used by the core LoraLoader node."""

    def list_loras(self) -> list[str]:
        import folder_paths

        return folder_paths.get_filename_list("loras")

    def full_path(self, name: str) -> str | None:
        import folder_paths

        return folder_paths.get_full_path("loras", name)

    def load_file(self, path: str) -> tuple[Any, Any]:
        import comfy.utils

        return comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)

    def apply(self, model, clip, lora, strength_model, strength_clip, metadata):
        import comfy.sd

        return comfy.sd.load_lora_for_models(
            model, clip, lora, strength_model, strength_clip, lora_metadata=metadata
        )


def _normalize(name: str) -> str:
    return name.replace("\\", "/")


def _stem(name: str) -> str:
    return os.path.splitext(os.path.basename(name))[0]


def resolve_lora_name(name: str, available: list[str]) -> str | None:
    """Find ``name`` in the LoRA list, falling back to a filename match.

    The fallback keeps workflows working when the same file lives in a different
    subfolder (or on another machine). Preference order: exact path, same file
    name, same file name without extension.
    """
    wanted = _normalize(name)
    normalized = {_normalize(a): a for a in available}
    if wanted in normalized:
        return normalized[wanted]

    wanted_base = os.path.basename(wanted)
    for norm, original in normalized.items():
        if os.path.basename(norm) == wanted_base:
            return original

    wanted_stem = _stem(wanted)
    for norm, original in normalized.items():
        if _stem(norm) == wanted_stem:
            return original
    return None


class LoraCache:
    """Small LRU of loaded LoRA files, invalidated when the file changes on disk."""

    def __init__(self, max_entries: int = DEFAULT_CACHE_SIZE):
        self.max_entries = max_entries
        self._entries: OrderedDict[str, tuple[tuple[float, int], Any, Any]] = OrderedDict()

    @staticmethod
    def _signature(path: str) -> tuple[float, int]:
        stat = os.stat(path)
        return (stat.st_mtime, stat.st_size)

    def get(self, path: str, load_file) -> tuple[Any, Any]:
        signature = self._signature(path)
        entry = self._entries.get(path)
        if entry is not None and entry[0] == signature:
            self._entries.move_to_end(path)
            return entry[1], entry[2]

        lora, metadata = load_file(path)
        self._entries[path] = (signature, lora, metadata)
        self._entries.move_to_end(path)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
        return lora, metadata

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()


@dataclass
class ApplyResult:
    model: Any
    clip: Any
    applied: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = list(self.applied) or ["(no LoRAs applied)"]
        lines += [f"WARNING: {w}" for w in self.warnings]
        return "\n".join(lines)


def apply_matches(model: Any, clip: Any, matches: list[Match], backend: Backend,
                  cache: LoraCache) -> ApplyResult:
    """Apply each matched LoRA in order. Problems are warnings, never errors."""
    result = ApplyResult(model=model, clip=clip)
    if not matches:
        return result

    available = backend.list_loras()
    for match in matches:
        row = match.row
        if row.strength == 0:
            continue

        name = resolve_lora_name(row.lora, available)
        path = backend.full_path(name) if name else None
        if not path:
            result.warnings.append(f"LoRA file not found: {row.lora}")
            continue

        try:
            lora, metadata = cache.get(path, backend.load_file)
        except Exception as err:  # corrupt or unreadable file: skip, don't fail the run
            result.warnings.append(f"could not load {name}: {err}")
            continue

        strength_clip = row.strength if result.clip is not None else 0.0
        result.model, result.clip = backend.apply(
            result.model, result.clip, lora, row.strength, strength_clip, metadata
        )
        result.applied.append(f"{name} @ {row.strength:g} (matched \"{match.matched_text}\")")

    for warning in result.warnings:
        log.warning("%s %s", LOG_PREFIX, warning)
    return result
