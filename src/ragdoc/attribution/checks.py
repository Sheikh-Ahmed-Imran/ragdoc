"""The attribution checks: ``locate``, ``survive`` and ``rank``.

These are the backward checks the attribution engine runs against the artefacts
the forward pipeline produced, one per attribution stage:

* ``locate`` (extraction) and ``survive`` (chunking) consume only extraction and
  chunking output (an :class:`~ragdoc.artifacts.ExtractedDoc` and a list of
  :class:`~ragdoc.artifacts.Chunk`), so they are *offline*: no embedding model,
  no network, no cost (``.ai/constitution.md`` — no network calls in the
  extraction and chunking stages; spec Scope).
* ``rank`` (retrieval) consumes the retrieval artefact — a precomputed
  :class:`~ragdoc.artifacts.Ranking` produced by the retriever — and does no
  embedding or network work itself: it only *reads* the ranking to find the
  target chunk's position and compares it to ``top_k``. The scoring already
  happened upstream in the retriever (T-005).

The rule that turns these three check results into a single verdict naming one
stage lives in :mod:`ragdoc.attribution.verdict`, not here; a check only ever
reports on its own stage.

Two rules from the project's decisions shape what these checks are allowed to
do:

* **Ground truth is the user's asserted answer, never inferred.** ``locate``
  searches the extracted text for the *exact* string the user supplied; it does
  not generate a golden answer and does not read retrieval scores to decide what
  "should" have been found (``.ai/constitution.md``; ADR-002). Matching is
  verbatim, modulo whitespace and letter case — paraphrase is explicitly out of
  scope for v0 (spec Exclusions, narrowed only to case by the 2026-10-22
  addendum).
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

import unicodedata
from typing import Any, Optional, Sequence, Union

from ragdoc.artifacts import CharSpan, CheckResult, Chunk, ExtractedDoc, Ranking

__all__ = ["locate", "survive", "rank", "retarget"]


# --------------------------------------------------------------------------- #
# locate
# --------------------------------------------------------------------------- #

def _normalize_with_map(text: str) -> tuple[str, list[int]]:
    """Whitespace-normalize and case-fold ``text``, keeping a map to original offsets.

    Every run of whitespace (spaces, tabs, newlines) collapses to a single
    space, and letters are lowercased. The returned ``offsets`` list is parallel to the normalized string:
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
            # A character whose lowercase is several characters maps each of
            # them back to the same original offset, so offsets stay valid.
            for folded in _fold(text[i]):
                norm_chars.append(folded)
                offsets.append(i)
            i += 1
    return "".join(norm_chars), offsets


def _fold(c: str) -> str:
    """NFKC-normalize then casefold one character; may yield several (``"ﬁ"``→``"fi"``, ``"ß"``→``"ss"``)."""
    return unicodedata.normalize("NFKC", c).casefold()


def _normalize(s: str) -> str:
    """Collapse whitespace runs to single spaces, strip the ends, fold case.

    Case is folded per character (not ``str.lower()`` on the whole string) so the
    answer folds exactly as ``_normalize_with_map`` folds the text.
    """
    return "".join(_fold(c) for c in " ".join(s.split()))


def _find_spans(text: str, answer: str) -> tuple[list[CharSpan], str]:
    """Find every occurrence of ``answer`` in ``text``; return spans and match kind.

    Works at the best available tier and never mixes tiers: if the answer occurs
    verbatim, every exact-substring occurrence is returned (kind ``"exact"``);
    only when there is none does it fall back to a whitespace-normalized,
    case-insensitive match, returning every normalized occurrence (kind
    ``"normalized"``) mapped back onto the original text. Spans are in document
    order; an empty list means no occurrence. ``answer`` is assumed non-empty.

    The match kind is carried in evidence so a reader can tell a verbatim hit
    from one that only held after normalizing.
    """
    spans: list[CharSpan] = []
    at = text.find(answer)
    while at >= 0:
        spans.append(CharSpan(at, at + len(answer)))
        at = text.find(answer, at + 1)
    if spans:
        return spans, "exact"

    norm_answer = _normalize(answer)
    if not norm_answer:
        return [], "normalized"
    norm_text, offsets = _normalize_with_map(text)
    at = norm_text.find(norm_answer)
    while at >= 0:
        start = offsets[at]
        end = offsets[at + len(norm_answer) - 1] + 1
        spans.append(CharSpan(start, end))
        at = norm_text.find(norm_answer, at + 1)
    return spans, "normalized"


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
        * ``pass`` if found. Evidence carries ``spans`` (every occurrence in
          document order, at the best tier — all exact occurrences, or if there
          are none all normalized ones, never a mix) and, for display, a primary
          ``span`` (the first occurrence, a :class:`CharSpan` into ``doc.text``),
          ``page`` (1-based, or ``None`` if the page map does not cover the span
          start), ``snippet`` (the matched original text) and ``match``
          (``"exact"`` or ``"normalized"``). ``survive`` consumes ``spans``;
          :func:`retarget` later repoints the display fields at the copy that
          was actually judged.
    """
    if not _normalize(expected_answer):
        return CheckResult.unevaluated("expected answer is empty; nothing to locate")

    spans, match = _find_spans(doc.text, expected_answer)
    if not spans:
        return CheckResult.failed(
            "expected answer not found in the extracted text; the loader did not "
            "preserve it",
            evidence={"expected_answer": expected_answer},
        )

    span = spans[0]
    return CheckResult.passed(
        evidence={
            "span": span,
            "spans": tuple(spans),
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


def survive(
    located_spans: Union[CharSpan, Sequence[CharSpan]], chunks: Sequence[Chunk]
) -> CheckResult:
    """Map the located answer span(s) onto the chunk list.

    This is the ``chunking`` check. Survival is decided by spans, never by chunk
    text: the answer survives if some chunk's source span wholly contains *any*
    located copy (``.ai/architecture/data-flow.md``).

    Args:
        located_spans: the span(s) ``locate`` reported (a single
            :class:`CharSpan` is accepted as a one-copy set), indexing the same
            extracted text the chunk spans index.
        chunks: the chunker's output, in document order.

    Returns:
        * ``unevaluated`` if there are no chunks, or if no located span overlaps
          any chunk span (inconsistent inputs — the verdict is withheld rather
          than guessed).
        * ``pass`` if any span is wholly contained by some chunk. Evidence
          carries, for the primary target (earliest copy, then earliest
          containing chunk): ``context_missing`` (a bool flag — see
          :func:`_lacks_context`), ``target_chunk_id``, ``target_span``,
          ``section_path``, ``page`` and ``located_span``. It also carries
          ``candidates`` — one such record per distinct containing chunk, in
          document order of copy then chunk — and ``target_chunk_ids``, from
          which ``rank`` picks the best-ranking chunk.
        * ``fail`` if no span is wholly contained but at least one overlaps a
          chunk — the boundary-cut chunking failure; evidence carries
          ``located_span`` (first overlapping copy), ``located_spans`` and the
          ``crossing_chunk_ids``.

    Candidate order is deterministic (copies in document order, chunks in chunk
    order), so ties downstream resolve to the earliest.
    """
    if isinstance(located_spans, CharSpan):
        spans: list[CharSpan] = [located_spans]
    else:
        spans = sorted(located_spans, key=lambda sp: (sp.start, sp.end))

    if not chunks:
        return CheckResult.unevaluated(
            "no chunks to map the located answer onto"
        )

    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for span in spans:
        for c in chunks:
            if c.span.contains(span) and c.id not in seen:
                seen.add(c.id)
                candidates.append(
                    {
                        "context_missing": _lacks_context(c),
                        "target_chunk_id": c.id,
                        "target_span": c.span,
                        "section_path": c.section_path,
                        "page": c.page,
                        "located_span": span,
                    }
                )

    if candidates:
        evidence = dict(candidates[0])
        evidence["candidates"] = tuple(candidates)
        evidence["target_chunk_ids"] = tuple(c["target_chunk_id"] for c in candidates)
        evidence["located_spans"] = tuple(spans)
        return CheckResult.passed(evidence=evidence)

    overlapped = [sp for sp in spans if any(c.span.overlaps(sp) for c in chunks)]
    if not overlapped:
        return CheckResult.unevaluated(
            "located span falls outside every chunk span"
        )

    overlapping = [c for c in chunks if any(c.span.overlaps(sp) for sp in overlapped)]
    return CheckResult.failed(
        "answer is split across a chunk boundary; no single chunk contains it",
        evidence={
            "located_span": overlapped[0],
            "located_spans": tuple(overlapped),
            "crossing_chunk_ids": [c.id for c in overlapping],
            "crossing_chunk_spans": [c.span for c in overlapping],
        },
    )


def retarget(
    located: CheckResult, survived: CheckResult, target_chunk_id: str, doc: ExtractedDoc
) -> tuple[CheckResult, CheckResult]:
    """Repoint ``locate``/``survive`` evidence at the copy in ``target_chunk_id``.

    ``survive`` names the earliest copy as primary, but ``rank`` may judge a
    later one (the best-ranking). The reported span/page/snippet, the
    ``context_missing`` flag and the target must describe the chunk the verdict
    was made on, so this returns copies of both results rewritten for it. Inputs
    that lack the candidate, or already target it, come back unchanged.
    """
    if not (located.is_pass and survived.is_pass):
        return located, survived
    if survived.evidence.get("target_chunk_id") == target_chunk_id:
        return located, survived
    chosen = next(
        (
            c
            for c in survived.evidence.get("candidates", ())
            if c["target_chunk_id"] == target_chunk_id
        ),
        None,
    )
    if chosen is None:
        return located, survived

    span = chosen["located_span"]
    new_located = CheckResult.passed(
        evidence={
            **located.evidence,
            "span": span,
            "page": doc.page_at(span.start),
            "snippet": doc.text[span.start : span.end],
        }
    )
    new_survived = CheckResult.passed(evidence={**survived.evidence, **chosen})
    return new_located, new_survived


# --------------------------------------------------------------------------- #
# rank
# --------------------------------------------------------------------------- #

def rank(
    target_chunk_id: Union[str, Sequence[str]], ranking: Ranking, top_k: int
) -> CheckResult:
    """Find the target chunk's position in the full ranking and compare to ``top_k``.

    This is the ``retrieval`` check. It does no embedding and no scoring of its
    own — the retriever (T-005) already produced the full :class:`Ranking` over
    *every* chunk; ``rank`` only reads the target chunk's absolute position out of
    it and asks whether that position is inside ``top_k``.

    The target chunk is the one ``survive`` found the answer in
    (``evidence["target_chunk_id"]``). When the answer survived in several chunks
    (a repeated answer) a sequence of candidate ids
    (``evidence["target_chunk_ids"]``) may be passed instead, and the
    **best-ranking** one is measured; ties resolve to the earliest candidate in
    the order given (document order), so the result is deterministic. Working
    from chunks the answer actually survived into — rather than re-deciding
    relevance — is what keeps the diagnosis grounded in the user's asserted
    answer and not in an inferred golden chunk (``.ai/constitution.md``; ADR-002).

    A ``fail`` here is only the *observed symptom* — "the right chunk ranked below
    the cutoff". Whether that symptom is attributed to retrieval or re-attributed
    upstream to chunking (the unembedded-heading case) is the verdict
    composition's job, which reads ``survive``'s ``context_missing`` flag
    alongside this result (``.ai/domain/domain.md``;
    :mod:`ragdoc.attribution.verdict`).

    Args:
        target_chunk_id: id of the chunk the answer survived into (from
            ``survive``'s evidence).
        ranking: the full, best-first ranking over every chunk.
        top_k: the cutoff the normal retrieval path would apply (from
            ``RunConfig.top_k``).

    Returns:
        * ``unevaluated`` if the ranking is empty, or if the target chunk is
          absent from it (an inconsistency between the chunk list and the
          ranking — the verdict is withheld rather than guessed).
        * ``fail`` if the target's 1-based position is greater than ``top_k`` —
          the normal retrieval path would not have returned it. Evidence carries
          ``position``, ``total`` (``len(ranking)``), ``top_k`` and
          ``target_chunk_id``, which the reporter renders as "ranked 23 of 400,
          top_k=10".
        * ``pass`` if the position is within ``top_k``. Evidence carries the same
          keys, so a verdict can still cite the exact rank even when it passed.
    """
    candidates = (
        [target_chunk_id] if isinstance(target_chunk_id, str) else list(target_chunk_id)
    )
    total = len(ranking)
    if total == 0:
        return CheckResult.unevaluated(
            "ranking is empty; nothing to locate the target chunk in"
        )
    if not candidates:
        return CheckResult.unevaluated("no target chunk to look up in the ranking")

    position: Optional[int] = None
    for cid in candidates:
        pos = ranking.position_of(cid)
        if pos is None:
            return CheckResult.unevaluated(
                f"target chunk {cid!r} is absent from the ranking; the "
                "chunk list and the ranking are inconsistent"
            )
        if position is None or pos < position:
            position, target_chunk_id = pos, cid

    evidence = {
        "position": position,
        "total": total,
        "top_k": top_k,
        "target_chunk_id": target_chunk_id,
    }

    if position > top_k:
        return CheckResult.failed(
            f"target chunk ranked {position} of {total}, outside top_k={top_k}",
            evidence=evidence,
        )
    return CheckResult.passed(evidence=evidence)
