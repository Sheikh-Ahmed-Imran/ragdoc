"""Shared columnar-table layout for the terminal reporters.

Symbol-agnostic: it pads whatever cell strings it is given, so a symbol-bearing
cell stays aligned in either ``Symbols`` mode. No color or ANSI.
"""

from __future__ import annotations

from typing import Sequence

__all__ = ["LEFT", "RIGHT", "render_table"]

LEFT = "l"
RIGHT = "r"

_GAP = "  "


def render_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    aligns: Sequence[str],
) -> list[str]:
    """Return header, a per-column ``-`` rule, then one line per row.

    ``aligns`` holds :data:`LEFT` or :data:`RIGHT` per column. Each width is
    ``max(len(header), longest cell)``; trailing spaces are stripped.
    """
    if len(aligns) != len(headers):
        raise ValueError("aligns must have one entry per header")
    for row in rows:
        if len(row) != len(headers):
            raise ValueError("every row must have one cell per header")
    widths = [
        max([len(h)] + [len(row[c]) for row in rows]) for c, h in enumerate(headers)
    ]

    def fmt(cells: Sequence[str]) -> str:
        padded = [
            cell.rjust(w) if align == RIGHT else cell.ljust(w)
            for cell, w, align in zip(cells, widths, aligns)
        ]
        return _GAP.join(padded).rstrip()

    return [fmt(headers), _GAP.join("-" * w for w in widths)] + [fmt(r) for r in rows]
