"""The artifacts every pipeline stage and attribution check pass to one another.

These are the stable vocabulary the forward pipeline (load → chunk → index →
retrieve) emits and the attribution engine reads. They are deliberately *data
only*: no stage logic lives here, so the engine can depend on these shapes
without depending on any stage implementation or on the interface layer
(``.ai/constitution.md`` — stage boundaries are real module boundaries; the
engine never depends on the UI).

Artifacts live in memory and are rebuilt each run. There is no persistence,
caching, or on-disk form of these types by design (spec Exclusions;
``.ai/architecture/system.md``).

Two things carried here are what make diagnosis possible at all:

* A ``Chunk`` records the *character span* of the source text it came from, not
  just its text. The chunker may alter chunk text (prepending a section path),
  so the span — not the text — is the stable link back to the document, and the
  thing ``survive`` maps a located answer onto.
* A ``Ranking`` covers *all* chunks, not just the ``top_k`` slice, so a verdict
  can state "ranked 23 of 400" even when the target chunk is far outside the
  cutoff.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Sequence

__all__ = [
    "CharSpan",
    "PageSpan",
    "ExtractedDoc",
    "Chunk",
    "RankedChunk",
    "Ranking",
    "CheckStatus",
    "CheckResult",
]


@dataclass(frozen=True)
class CharSpan:
    """A half-open ``[start, end)`` character span into an extracted text.

    Offsets index :attr:`ExtractedDoc.text`. Used both for a ``Chunk``'s source
    span and for the span ``locate`` reports for a found answer, so the two can
    be compared directly by ``survive``.
    """

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"invalid span: start={self.start}, end={self.end}")

    @property
    def length(self) -> int:
        return self.end - self.start

    def contains(self, other: "CharSpan") -> bool:
        """True if ``other`` lies wholly within this span.

        This is the test ``survive`` makes: an answer survives chunking if its
        located span is wholly contained in one chunk's source span.
        """
        return self.start <= other.start and other.end <= self.end

    def overlaps(self, other: "CharSpan") -> bool:
        """True if the two spans share any character (half-open)."""
        return self.start < other.end and other.start < self.end


@dataclass(frozen=True)
class PageSpan:
    """The character span one page occupies in the extracted text.

    ``page`` is 1-based, matching how page numbers are reported to the user.
    """

    page: int
    span: CharSpan


@dataclass
class ExtractedDoc:
    """Loader output: the full extracted text plus a page map.

    The extracted text is authoritative for position; page numbers in a verdict
    come from this page map, never from re-reading the PDF later
    (``.ai/architecture/data-flow.md``).

    The page map is expressed as an ordered, non-overlapping list of
    :class:`PageSpan`, covering the text start to finish. :meth:`page_at`
    resolves any character offset to its page number.
    """

    text: str
    page_map: Sequence[PageSpan] = field(default_factory=list)

    @property
    def empty_pages(self) -> tuple[int, ...]:
        """1-based numbers of pages that yielded no text, in page-map order.

        A page with a zero-length span produced nothing extractable (e.g. a
        scanned image page). Derived from :attr:`page_map`; extracts nothing new.
        """
        return tuple(p.page for p in self.page_map if p.span.length == 0)

    def page_at(self, offset: int) -> Optional[int]:
        """Return the 1-based page number containing ``offset``, or ``None``.

        ``None`` means the offset falls outside every recorded page span (an
        empty or incomplete map, or an out-of-range offset).
        """
        for page_span in self.page_map:
            s = page_span.span
            if s.start <= offset < s.end:
                return page_span.page
        return None


@dataclass(frozen=True)
class Chunk:
    """A unit the chunker produced from the extracted text.

    Attributes:
        id: Stable identifier for this chunk within a run. :class:`Ranking`
            refers to chunks by this id.
        text: The chunk's text as it will be embedded. May differ from the
            source slice when ``prepend_section_context`` is on.
        span: The source character span in :attr:`ExtractedDoc.text` this chunk
            came from. Authoritative for survival; unaffected by section-context
            prepending.
        page: 1-based page the chunk's source span starts on.
        section_path: The heading/section path the chunk sits under, outermost
            first (e.g. ``("2. Methods", "2.1 Sampling")``). Empty when no
            structure was detected.
    """

    id: str
    text: str
    span: CharSpan
    page: int
    section_path: tuple[str, ...] = ()


@dataclass(frozen=True)
class RankedChunk:
    """One entry in a :class:`Ranking`: a chunk id and its question score."""

    chunk_id: str
    score: float


@dataclass
class Ranking:
    """A full ordering of every chunk by score against the question.

    ``entries`` is ordered best-first and must cover *all* chunks, not just the
    ``top_k`` slice — the retriever's contract is that the target chunk's exact
    position is knowable even when it is far outside ``top_k``
    (``.ai/architecture/components.md``).
    """

    entries: Sequence[RankedChunk] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.entries)

    def position_of(self, chunk_id: str) -> Optional[int]:
        """Return the 1-based rank of ``chunk_id``, or ``None`` if absent.

        A return of ``1`` is the top-ranked chunk. Combined with :meth:`__len__`
        this supports a verdict phrased as "ranked 23 of 400".
        """
        for index, entry in enumerate(self.entries, start=1):
            if entry.chunk_id == chunk_id:
                return index
        return None


class CheckStatus(Enum):
    """The four distinct outcomes an attribution check can have.

    ``FAIL`` and ``UNEVALUATED`` are separate values on purpose: a stage that
    *failed* is the verdict, while a stage that *could not be evaluated* must
    withhold the verdict rather than fall through to the next plausible stage
    (``.ai/constitution.md`` — "A verdict is never produced without evidence
    from every stage it rules out"; spec AC#3).
    """

    PASS = "pass"
    FAIL = "fail"
    NOT_REACHED = "not_reached"
    UNEVALUATED = "unevaluated"


@dataclass(frozen=True)
class CheckResult:
    """The outcome of one attribution check.

    Construct via the classmethods rather than the raw initializer so the
    ``reason``/``evidence`` invariants are respected:

    * :meth:`passed` — the check passed. May carry supporting ``evidence``.
    * :meth:`failed` — the check failed; this is a verdict. Carries a ``reason``
      (which a fix suggestion is later attached to) and ``evidence``.
    * :meth:`not_reached` — an earlier check already failed, so this one was
      never run.
    * :meth:`unevaluated` — this check could not be run to a conclusion (e.g. a
      missing input); the verdict is withheld. Distinct from ``failed``.

    ``evidence`` is a free-form mapping (spans, pages, ranks, snippets) that the
    reporter renders; its exact keys are each check's concern, not this type's.
    """

    status: CheckStatus
    reason: Optional[str] = None
    evidence: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def passed(cls, evidence: Optional[Mapping[str, Any]] = None) -> "CheckResult":
        return cls(CheckStatus.PASS, evidence=dict(evidence or {}))

    @classmethod
    def failed(
        cls, reason: str, evidence: Optional[Mapping[str, Any]] = None
    ) -> "CheckResult":
        return cls(CheckStatus.FAIL, reason=reason, evidence=dict(evidence or {}))

    @classmethod
    def not_reached(cls) -> "CheckResult":
        return cls(CheckStatus.NOT_REACHED)

    @classmethod
    def unevaluated(cls, reason: Optional[str] = None) -> "CheckResult":
        return cls(CheckStatus.UNEVALUATED, reason=reason)

    @property
    def is_pass(self) -> bool:
        return self.status is CheckStatus.PASS

    @property
    def is_fail(self) -> bool:
        return self.status is CheckStatus.FAIL

    @property
    def is_not_reached(self) -> bool:
        return self.status is CheckStatus.NOT_REACHED

    @property
    def is_unevaluated(self) -> bool:
        return self.status is CheckStatus.UNEVALUATED
