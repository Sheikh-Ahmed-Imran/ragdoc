"""Loader comparison: did the expected answer survive each loader's extraction?

Given one PDF, the user's asserted answer and a list of loader names, run each
loader and apply the existing ``locate`` check to its :class:`ExtractedDoc`,
collecting one :class:`LoaderResult` per loader.

Rules this module keeps:

* **``locate`` only.** No chunking, embedding, retrieval or verdict; extraction
  stays offline, so the whole routine is network-free (``.ai/constitution.md``).
* **Ground truth is the supplied answer**, never generated or inferred (ADR-002).
* **Pages come from each loader's own page map.** Character offsets are never
  compared or reconciled across loaders — a position is only meaningful with the
  loader that produced it (``.ai/architecture/data-flow.md``).
* **Pure and headless.** It returns results and prints nothing; formatting lives
  in the reporting layer, which this module does not import.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from ragdoc.attribution.checks import locate
from ragdoc.loader import get, names

__all__ = ["LoaderResult", "TEXT_HEAD_MAX", "compare_loaders"]

#: Maximum length of ``LoaderResult.text_head``.
TEXT_HEAD_MAX = 240

#: Characters of surrounding context kept on each side of a matched span.
_CONTEXT = 40
_ELLIPSIS = "…"


@dataclass(frozen=True)
class LoaderResult:
    """What one loader did with the expected answer.

    Attributes:
        loader: the loader's registry name.
        found: ``locate`` passed — the answer is in this loader's extracted text.
        evaluated: ``False`` when ``locate`` could not run (empty answer); then
            ``found`` is ``False`` but does not mean the loader lost the answer.
        page: 1-based page from this loader's own page map; ``None`` when not
            found or the map does not cover the match.
        match: ``"exact"`` or ``"normalized"`` when found, else ``None``.
        char_count: length of the extracted text.
        page_count: number of pages in this loader's page map.
        text_head: the first ``TEXT_HEAD_MAX`` characters of the extracted text
            — a bounded sample, never the whole document.
        found_snippet: on a find, the matched span of the extraction with about
            40 characters of context each side, at most ``TEXT_HEAD_MAX``
            characters (ellipses included), ``…``-marked where clipped; ``None``
            on a miss, an unevaluated result or an empty extraction.
        empty_pages: 1-based pages this loader extracted no text from
            (``ExtractedDoc.empty_pages``); advisory only.
    """

    loader: str
    found: bool
    evaluated: bool
    page: Optional[int]
    match: Optional[str]
    char_count: int
    page_count: int
    text_head: str = ""
    found_snippet: Optional[str] = None
    empty_pages: tuple[int, ...] = ()


def _matched_window(text: str, start: int, end: int) -> str:
    """Slice ``text`` around ``[start, end)``, bounded to ``TEXT_HEAD_MAX``."""
    n = len(text)
    context = _CONTEXT
    while True:
        lo, hi = max(0, start - context), min(n, end + context)
        marks = (lo > 0) + (hi < n)
        if hi - lo + marks <= TEXT_HEAD_MAX:
            break
        if context == 0:
            # The match alone fills the bound: take the head of the matched region.
            lo = start
            hi = min(n, start + TEXT_HEAD_MAX - (start > 0) - 1)
            if hi >= n:
                hi = n
            break
        context -= 1
    body = text[lo:hi]
    return (_ELLIPSIS if lo > 0 else "") + body + (_ELLIPSIS if hi < n else "")


def compare_loaders(
    pdf_path: str | Path,
    expected_answer: str,
    loader_names: Optional[Sequence[str]] = None,
) -> list[LoaderResult]:
    """Run each named loader on ``pdf_path`` and ``locate`` the answer in it.

    Args:
        pdf_path: the PDF to extract.
        expected_answer: the user's asserted answer.
        loader_names: loaders to run, in the order results are returned; a
            repeated name runs once, at its first occurrence. Defaults to
            every registered loader (sorted by name).

    Raises:
        UnknownLoaderError: a name is not registered (raised before any loader
            runs, so a typo does not cost a partial run).
        FileNotFoundError: ``pdf_path`` does not exist (from the loader).
    """
    # dict.fromkeys drops repeats but keeps first-occurrence order.
    chosen = list(dict.fromkeys(names() if loader_names is None else loader_names))
    loaders = [(name, get(name)) for name in chosen]

    results: list[LoaderResult] = []
    for name, loader in loaders:
        doc = loader.load(pdf_path)
        check = locate(expected_answer, doc)
        found = check.is_pass
        span = check.evidence.get("span") if found else None
        results.append(
            LoaderResult(
                loader=name,
                found=found,
                evaluated=not check.is_unevaluated,
                page=check.evidence.get("page") if found else None,
                match=check.evidence.get("match") if found else None,
                char_count=len(doc.text),
                page_count=len(doc.page_map),
                text_head=doc.text[:TEXT_HEAD_MAX],
                found_snippet=(
                    _matched_window(doc.text, span.start, span.end) if span is not None else None
                ),
                empty_pages=doc.empty_pages,
            )
        )
    return results
