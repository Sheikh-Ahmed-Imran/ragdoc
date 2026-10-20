"""The offline attribution checks: ``locate`` and ``survive``.

These are the first two of the backward checks the attribution engine runs
against the artefacts the forward pipeline produced. They consume only
extraction and chunking output (an :class:`~ragdoc.artifacts.ExtractedDoc` and a
list of :class:`~ragdoc.artifacts.Chunk`), so they are *offline*: no embedding
model, no network, no cost (``.ai/constitution.md`` — no network calls in the
extraction and chunking stages; spec Scope). ``rank`` (retrieval) and the final
verdict composition are a later task (T-006).

Two rules from the project's decisions shape what these checks are allowed to
do:

* **Ground truth is the user's asserted answer, never inferred.** ``locate``
  searches the extracted text for the *exact* string the user supplied; it does
  not generate a golden answer and does not read retrieval scores to decide what
  "should" have been found (``.ai/constitution.md``; ADR-002). Matching is
  verbatim, modulo whitespace — paraphrase is explicitly out of scope for v0
  (spec Exclusions).
* **The chunk's source span is authoritative for survival, not its text.**
  ``survive`` decides whether an answer survived chunking by comparing *spans*
  (``CharSpan.contains``), never by fuzzy-matching chunk text, because the whole
  point is to catch the case where a boundary cut the answer in half
  (``.ai/architecture/data-flow.md``).

Each check returns a :class:`~ragdoc.artifacts.CheckResult`. A check that cannot
be run to a conclusion returns ``unevaluated`` (not ``fail``): the engine then
withholds the verdict rather than letting it fall through to the next stage
(``.ai/constitution.md``).
"""

from __future__ import annotations

from typing import Optional, Sequence

from ragdoc.artifacts import CharSpan, CheckResult, Chunk, ExtractedDoc

__all__ = ["locate", "survive"]


# --------------------------------------------------------------------------- #
# locate
# --------------------------------------------------------------------------- #

def _normalize_with_map(text: str) -> tuple[str, list[int]]:
    """Whitespace-normalize ``text``, keeping a map back to original offsets.

    Every run of whitespace (spaces, tabs, newlines) collapses to a single
    space. The returned ``offsets`` list is parallel to the normalized string:
    ``offsets[i]`` is the index in the *original* ``text`` of the character that
    produced normalized character ``i``. For a collapsed-whitespace space, that
    is the offset of the first whitespace character in the run.

    This is what lets ``locate`` report a span that indexes the original
    extracted text even when the match only held after normalizing whitespace.
    """
    norm_chars: list[str] = []
    offsets: list[int] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i].isspace():
            run_start = i
            while i < n and text[i].isspace():
                i += 1
            norm_chars.append(" ")
            offsets.append(run_start)
        else:
            norm_chars.append(text[i])
            offsets.append(i)
            i += 1
    return "".join(norm_chars), offsets


def _normalize(s: str) -> str:
    """Collapse all whitespace runs to single spaces and strip the ends."""
    return " ".join(s.split())


def _find_span(text: str, answer: str) -> Optional[tuple[CharSpan, str]]:
    """Find ``answer`` in ``text``; return its span and the match kind, or None.

    Tries an exact substring match first (so the reported span is precise), then
    falls back to a whitespace-normalized match, mapping the normalized match
    position back onto the original text. ``answer`` is assumed non-empty.

    The match kind is ``"exact"`` or ``"normalized"``, carried in evidence so a
    reader can tell a verbatim hit from one that only held after normalizing.
    """
    exact = text.find(answer)
    if exact >= 0:
        return CharSpan(exact, exact + len(answer)), "exact"

    norm_answer = _normalize(answer)
    if not norm_answer:
        return None
    norm_text, offsets = _normalize_with_map(text)
    at = norm_text.find(norm_answer)
    if at < 0:
        return None
    start = offsets[at]
    end = offsets[at + len(norm_answer) - 1] + 1
    return CharSpan(start, end), "normalized"


def locate(expected_answer: str, doc: ExtractedDoc) -> CheckResult:
    """Find the user's expected answer in the extracted text.

    This is the ``extraction`` check. The expected answer is ground truth the
    user asserted (ADR-002); ``locate`` searches ``doc.text`` for it verbatim
    (modulo whitespace) and does not generate or infer an answer of its own.

    Returns:
        * ``unevaluated`` if ``expected_answer`` is empty or all whitespace —
          there is nothing to locate, so the verdict is withheld rather than
          reported as an extraction failure.
        * ``fail`` if the answer is not present in the extracted text. This is
          the verdict "the loader lost it": the text the user says is in the
          document did not survive extraction. Evidence carries
          ``expected_answer``.
        * ``pass`` if found. Evidence carries ``span`` (a :class:`CharSpan` into
          ``doc.text``), ``page`` (1-based, or ``None`` if the page map does not
          cover the span start), ``snippet`` (the matched original text), and
          ``match`` (``"exact"`` or ``"normalized"``). ``survive`` consumes
          ``span``.
    """
    if not _normalize(expected_answer):
        return CheckResult.unevaluated("expected answer is empty; nothing to locate")

    found = _find_span(doc.text, expected_answer)
    if found is None:
        return CheckResult.failed(
            "expected answer not found in the extracted text; the loader did not "
            "preserve it",
            evidence={"expected_answer": expected_answer},
        )

    span, match = found
    return CheckResult.passed(
        evidence={
            "span": span,
            "page": doc.page_at(span.start),
            "snippet": doc.text[span.start : span.end],
            "match": match,
        }
    )


# --------------------------------------------------------------------------- #
# survive
# --------------------------------------------------------------------------- #

def _lacks_context(chunk: Chunk) -> bool:
    """True if ``chunk`` has a detected heading that is not in its embedded text.

    "Context missing" is deliberately narrow: it fires only when a section path
    *was* detected for the chunk but its heading labels are absent from
    ``chunk.text`` — i.e. the heading exists but will not be embedded, so it
    cannot help the chunk rank (the unembedded-heading failure;
    ``.ai/product/product.md``). When ``prepend_section_context`` is on the
    chunker writes the labels into ``chunk.text``, so they are present and
    context is *not* missing.

    A chunk with no detected section path is **not** treated as context-missing:
    there is no heading to prepend, and flagging it would misattribute a genuine
    retrieval failure to chunking (T-006's verdict rule keys off this flag).
    """
    if not chunk.section_path:
        return False
    return not all(label in chunk.text for label in chunk.section_path)


def survive(located_span: CharSpan, chunks: Sequence[Chunk]) -> CheckResult:
    """Map a located answer span onto the chunk list.

    This is the ``chunking`` check. Survival is decided by spans, never by chunk
    text: the answer survives if some chunk's source span wholly contains the
    located span (``.ai/architecture/data-flow.md``).

    Args:
        located_span: the span ``locate`` reported, indexing the same extracted
            text the chunk spans index.
        chunks: the chunker's output, in document order.

    Returns:
        * ``unevaluated`` if there are no chunks, or if the located span falls
          outside every chunk span (inconsistent inputs — the verdict is
          withheld rather than guessed).
        * ``pass`` if one chunk wholly contains the span. Evidence carries
          ``context_missing`` (a bool flag — see :func:`_lacks_context`),
          ``target_chunk_id``, ``target_span``, ``section_path``, ``page`` and
          the ``located_span``. The flag lets T-006 distinguish an answer that
          survived cleanly from one that survived into a context-poor chunk.
        * ``fail`` if a chunk boundary cuts the span — it overlaps chunks but no
          single chunk contains it. This is the boundary-cut chunking failure;
          evidence carries ``located_span`` and the ``crossing_chunk_ids``.

    When chunks overlap, a span in the shared region may be contained by more
    than one chunk; the earliest in document order is taken as the target, so
    the result is deterministic.
    """
    if not chunks:
        return CheckResult.unevaluated(
            "no chunks to map the located answer onto"
        )

    containing = [c for c in chunks if c.span.contains(located_span)]
    overlapping = [c for c in chunks if c.span.overlaps(located_span)]

    if not overlapping:
        return CheckResult.unevaluated(
            "located span falls outside every chunk span"
        )

    if containing:
        target = containing[0]
        context_missing = _lacks_context(target)
        return CheckResult.passed(
            evidence={
                "context_missing": context_missing,
                "target_chunk_id": target.id,
                "target_span": target.span,
                "section_path": target.section_path,
                "page": target.page,
                "located_span": located_span,
            }
        )

    return CheckResult.failed(
        "answer is split across a chunk boundary; no single chunk contains it",
        evidence={
            "located_span": located_span,
            "crossing_chunk_ids": [c.id for c in overlapping],
            "crossing_chunk_spans": [c.span for c in overlapping],
        },
    )
