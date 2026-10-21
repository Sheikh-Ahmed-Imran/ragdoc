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

__all__ = ["LoaderResult", "compare_loaders"]


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
    """

    loader: str
    found: bool
    evaluated: bool
    page: Optional[int]
    match: Optional[str]
    char_count: int
    page_count: int


def compare_loaders(
    pdf_path: str | Path,
    expected_answer: str,
    loader_names: Optional[Sequence[str]] = None,
) -> list[LoaderResult]:
    """Run each named loader on ``pdf_path`` and ``locate`` the answer in it.

    Args:
        pdf_path: the PDF to extract.
        expected_answer: the user's asserted answer.
        loader_names: loaders to run, in the order results are returned;
            defaults to every registered loader (sorted by name).

    Raises:
        UnknownLoaderError: a name is not registered (raised before any loader
            runs, so a typo does not cost a partial run).
        FileNotFoundError: ``pdf_path`` does not exist (from the loader).
    """
    chosen = list(names() if loader_names is None else loader_names)
    loaders = [(name, get(name)) for name in chosen]

    results: list[LoaderResult] = []
    for name, loader in loaders:
        doc = loader.load(pdf_path)
        check = locate(expected_answer, doc)
        found = check.is_pass
        results.append(
            LoaderResult(
                loader=name,
                found=found,
                evaluated=not check.is_unevaluated,
                page=check.evidence.get("page") if found else None,
                match=check.evidence.get("match") if found else None,
                char_count=len(doc.text),
                page_count=len(doc.page_map),
            )
        )
    return results
