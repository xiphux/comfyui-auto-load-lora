"""Pure trigger-matching logic for the Auto LoRA Loader.

Nothing in this module imports ComfyUI, so it can be unit tested standalone.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any

# Space, hyphen and underscore are interchangeable separators inside a trigger.
_SEPARATOR_CLASS = r"[\s_\-]+"
# A trigger must not be glued to a letter or digit on either side. Underscore is
# treated as a separator rather than a word character, so "elara_voss" matches.
_WORD_BEFORE = r"(?<![^\W_])"
_WORD_AFTER = r"(?![^\W_])"

DEFAULT_STRENGTH = 1.0


class RowsFormatError(ValueError):
    """The serialized row list could not be parsed."""


@dataclass(frozen=True)
class LoraRow:
    lora: str
    triggers: str = ""
    regex: bool = False
    strength: float = DEFAULT_STRENGTH
    enabled: bool = True


@dataclass(frozen=True)
class Match:
    row: LoraRow
    matched_text: str


def parse_rows(value: Any) -> list[LoraRow]:
    """Parse the serialized row list.

    The widget stores a JSON string. API callers may also send the list directly,
    which ComfyUI coerces to a Python repr string, so that form is accepted too.
    """
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            data = json.loads(text)
        except json.JSONDecodeError as json_err:
            try:
                data = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                raise RowsFormatError(f"LoRA rows are not valid JSON: {json_err}") from None
    else:
        data = value

    if not isinstance(data, list):
        raise RowsFormatError("LoRA rows must be a list of objects")

    rows = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise RowsFormatError(f"LoRA row {index + 1} must be an object")
        lora = str(item.get("lora") or "").strip()
        try:
            strength = float(item.get("strength", DEFAULT_STRENGTH))
        except (TypeError, ValueError):
            raise RowsFormatError(f"LoRA row {index + 1} has an invalid strength") from None
        rows.append(
            LoraRow(
                lora=lora,
                triggers=str(item.get("triggers") or ""),
                regex=bool(item.get("regex", False)),
                strength=strength,
                enabled=bool(item.get("enabled", True)),
            )
        )
    return rows


def split_aliases(triggers: str) -> list[str]:
    """Split a comma-separated trigger field into non-empty aliases."""
    return [alias.strip() for alias in triggers.split(",") if alias.strip()]


def alias_pattern(alias: str) -> re.Pattern[str] | None:
    """Compile a plain (non-regex) alias into a case-insensitive whole-word pattern."""
    tokens = [t for t in re.split(_SEPARATOR_CLASS, alias.strip()) if t]
    if not tokens:
        return None
    body = _SEPARATOR_CLASS.join(re.escape(t) for t in tokens)
    return re.compile(_WORD_BEFORE + body + _WORD_AFTER, re.IGNORECASE)


def row_patterns(row: LoraRow) -> list[re.Pattern[str]]:
    """Compile the patterns for a row.

    In regex mode the whole field is one expression (commas are regex syntax, e.g.
    ``{1,3}``, so they are not treated as alias separators; use ``|`` instead).
    Raises ``re.error`` for an invalid expression.
    """
    if row.regex:
        expr = row.triggers.strip()
        return [re.compile(expr, re.IGNORECASE)] if expr else []
    return [p for p in (alias_pattern(a) for a in split_aliases(row.triggers)) if p]


def find_matches(rows: list[LoraRow], prompt: str) -> tuple[list[Match], list[str]]:
    """Return the rows whose triggers appear in ``prompt``, plus any warnings.

    Disabled rows, rows without a LoRA, and rows without triggers never match.
    Each LoRA file is applied at most once, even if several rows reference it.
    """
    matches: list[Match] = []
    warnings: list[str] = []
    seen: set[str] = set()
    prompt = prompt or ""

    for row in rows:
        if not row.enabled or not row.lora or row.lora in seen:
            continue
        try:
            patterns = row_patterns(row)
        except re.error as err:
            warnings.append(f"invalid regex for {row.lora}: {err}")
            continue
        for pattern in patterns:
            # Skip zero-length hits so a regex like "foo?" can't match everywhere.
            found = next((m for m in pattern.finditer(prompt) if m.group(0)), None)
            if found:
                matches.append(Match(row=row, matched_text=found.group(0)))
                seen.add(row.lora)
                break

    return matches, warnings
