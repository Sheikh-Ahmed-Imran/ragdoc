"""Status symbols for the terminal reporters: one source of truth.

Every marker the reporters print exists here in a Unicode form and an ASCII
form. :func:`resolve` picks the set for an output stream. The choice is a
display concern only: it is not part of ``RunConfig`` and never affects a
verdict.

This module imports nothing from the engine, checks, loader or any other stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# One glyph from the Unicode set that legacy codepages (437/850/1252) cannot
# encode; if the stream can encode it, it can encode the whole set.
_PROBE = "⚑"


@dataclass(frozen=True)
class Symbols:
    """One marker per field."""

    passed: str
    fail: str
    flagged: str
    gap: str
    not_reached: str
    found: str
    miss: str
    # Prose glyphs the reporters also print; defaulted so positional callers
    # built before they existed keep working.
    pointer: str = "←"
    ellipsis: str = "…"


UNICODE = Symbols(
    passed="✓",
    fail="✗",
    flagged="⚑",
    gap="⋮",
    not_reached="—",
    found="✓ found",
    miss="✗ MISS",
)

ASCII = Symbols(
    passed="[OK]",
    fail="[X]",
    flagged="[!]",
    gap="...",
    not_reached="-",
    found="[OK] found",
    miss="[X] MISS",
    pointer="<-",
    ellipsis="...",
)


def _can_encode(stream: Any) -> bool:
    encoding = getattr(stream, "encoding", None)
    if not encoding:
        return False
    try:
        _PROBE.encode(encoding)
    except (LookupError, UnicodeError, TypeError):
        return False
    return True


def resolve(stream: Any, force_ascii: bool = False) -> Symbols:
    """Return the ASCII set if forced or if ``stream`` cannot show Unicode."""
    if force_ascii or not _can_encode(stream):
        return ASCII
    return UNICODE
