"""Terminal view of the top-k chunks the retriever returned for a question.

Renders the :class:`~ragdoc.artifacts.Ranking` and :class:`~ragdoc.artifacts.Chunk`
data it is handed and nothing else: it runs no retrieval, embedding or check, loads
no PDF, and imports no CLI or engine runner.

Scores are the ranking's cosine similarities, shown as-is. They are relative
within this list, not a calibrated or absolute quality score.
"""

from __future__ import annotations

import re
from typing import Optional, Sequence

from ragdoc.artifacts import Chunk, Ranking, RankedChunk
from ragdoc.report.compare import _escape
from ragdoc.report.symbols import UNICODE, Symbols

__all__ = ["render_retrieved", "SNIPPET_MAX"]

#: Default bound on a row's snippet, in characters (after escaping).
SNIPPET_MAX = 120

_MARKER = "your answer"
_NO_TARGET_NOTE = "(the answer was not located in a single chunk, so none is marked)"
# Control characters _escape leaves alone, plus the Unicode line separators.
_OTHER_BREAKS = re.compile("[\x00-\x1f\x7f\x85  ]")


def _one_line(text: str) -> str:
    escaped = _escape(text)
    return _OTHER_BREAKS.sub(lambda m: f"\\x{ord(m.group()):02x}", escaped)


def _snippet(text: str, limit: int, symbols: Symbols) -> str:
    shown = _one_line(text)
    if len(shown) <= limit:
        return f'"{shown}"'
    cut = shown[: max(limit, 0)]
    # Don't end the snippet half-way through an escape sequence.
    trailing = len(cut) - len(cut.rstrip("\\"))
    if trailing % 2:
        cut = cut[:-1]
    return f'"{cut}"{symbols.ellipsis}'


def _row(
    rank: int,
    entry: RankedChunk,
    by_id: dict[str, Chunk],
    target_chunk_id: Optional[str],
    limit: int,
    symbols: Symbols,
) -> Optional[str]:
    chunk = by_id.get(entry.chunk_id)
    if chunk is None:
        return None
    line = (
        f"  {rank:>3}. {entry.chunk_id}  p.{chunk.page}  "
        f"{entry.score:.4f}  {_snippet(chunk.text, limit, symbols)}"
    )
    if entry.chunk_id == target_chunk_id:
        line += f"  {symbols.pointer} {_MARKER}"
    return line


def render_retrieved(
    ranking: Ranking,
    chunks: Sequence[Chunk],
    target_chunk_id: Optional[str],
    top_k: int,
    limit: int = SNIPPET_MAX,
    symbols: Symbols = UNICODE,
) -> str:
    """Render the ``top_k`` best-ranked chunks, marking the user's answer chunk.

    Each row shows rank (1-based), chunk id, page, cosine score and a bounded,
    escaped, single-line snippet. When ``target_chunk_id`` ranks beyond ``top_k``
    it follows after a gap row with its true rank. With no target, the rows carry
    no marker and a one-line note says why.
    """
    by_id = {chunk.id: chunk for chunk in chunks}
    entries = list(ranking.entries)
    shown = max(top_k, 0)

    lines = [f"Retrieved chunks (top {shown} by cosine similarity):"]
    for rank, entry in enumerate(entries[:shown], start=1):
        row = _row(rank, entry, by_id, target_chunk_id, limit, symbols)
        if row is not None:
            lines.append(row)

    if target_chunk_id is None:
        lines.append(_NO_TARGET_NOTE)
    else:
        position = ranking.position_of(target_chunk_id)
        if position is not None and position > shown:
            row = _row(position, entries[position - 1], by_id, target_chunk_id, limit, symbols)
            if row is not None:
                lines += [f"  {symbols.gap}", row]
    return "\n".join(lines)
