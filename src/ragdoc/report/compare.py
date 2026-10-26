"""Terminal reporter for a loader comparison: one row per loader.

Renders the :class:`~ragdoc.compare.LoaderResult` facts T-004 returned and
nothing else. Terminal text only (spec Exclusions, ADR-004).

Rules this module keeps:

* **No metric.** A ``locate``-only comparison has no precision/recall figure, so
  none is printed — not a placeholder, not a ceiling line. The honest form is
  to show no metric at all (``.ai/constitution.md``).
* **Facts only.** It never recommends or selects a loader.
* **Separate from the engine.** It does not run the comparison and imports no
  CLI layer; the result type is imported for typing only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from ragdoc.report._table import LEFT, RIGHT, render_table
from ragdoc.report.symbols import UNICODE, Symbols

if TYPE_CHECKING:
    from ragdoc.compare import LoaderResult

__all__ = ["render_comparison"]

_HEADERS = ("loader", "result", "page", "match", "chars", "pages")
_ALIGNS = (LEFT, LEFT, LEFT, LEFT, RIGHT, RIGHT)
_UNEVALUATED ="? not evaluated"


def _row(result: "LoaderResult", symbols: Symbols) -> tuple[str, ...]:
    if result.found:
        status = symbols.found
    elif result.evaluated:
        status = symbols.miss
    else:
        status = _UNEVALUATED
    return (
        result.loader,
        status,
        symbols.not_reached if result.page is None else str(result.page),
        result.match or symbols.not_reached,
        str(result.char_count),
        str(result.page_count),
    )


def _attempt(result: "LoaderResult") -> str:
    if result.found:
        return result.match or "found"
    return "not found" if result.evaluated else "not evaluated (empty answer)"


def _escape(text: str) -> str:
    # Escape control characters so the snippet stays on one line.
    return text.replace("\\", "\\\\").replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")


def _snippet(result: "LoaderResult", symbols: Symbols) -> str:
    if result.found and result.found_snippet:
        # The snippet carries its own ellipses where it was clipped.
        return f'"{_escape(result.found_snippet)}"'
    head = result.text_head
    if not head:
        return "(no text extracted)"
    shown = _escape(head)
    text = f'"{shown}"'
    if len(head) < result.char_count:
        text += f" {symbols.ellipsis} ({len(head)} of {result.char_count} chars)"
    return text


def _verbose_block(results: Sequence["LoaderResult"], symbols: Symbols) -> list[str]:
    lines: list[str] = []
    for r in results:
        lines += ["", r.loader, f"  match attempt: {_attempt(r)}"]
        if r.found and r.found_snippet and r.page is not None:
            lines.append(f"  page:          {r.page}")
        lines.append(f"  extracted:     {_snippet(r, symbols)}")
    return lines


def render_comparison(
    results: Sequence["LoaderResult"],
    verbose: bool = False,
    symbols: Symbols = UNICODE,
) -> str:
    """Render one table row per loader; a miss is marked with ``symbols.miss``.

    Pages are each loader's own numbering and are not comparable across rows
    by character position; no figure other than counts of extracted text is shown.

    With ``verbose=True`` the table is followed by, per loader, the match
    attempt and the bounded extraction snippet carried by the result.
    """
    if not results:
        return "No loaders were compared."
    lines = render_table(_HEADERS, [_row(r, symbols) for r in results], _ALIGNS)
    advisory = [
        f"{r.loader}: {len(r.empty_pages)} page(s) had no extractable text: "
        + ", ".join(f"p.{n}" for n in r.empty_pages)
        for r in results
        if r.empty_pages
    ]
    if advisory:
        lines += [""] + advisory
    if verbose:
        lines += _verbose_block(results, symbols)
    return "\n".join(lines)
