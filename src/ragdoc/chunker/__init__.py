"""The chunker stage: extracted text becomes chunks with source spans.

The chunker is the second forward-pipeline stage. It splits the loader's
:class:`~ragdoc.artifacts.ExtractedDoc` into a list of
:class:`~ragdoc.artifacts.Chunk`\\ s and — the point of the whole stage — records
for each chunk the *character span* of the extracted text it came from
(``.ai/architecture/components.md``).

Three rules from the architecture shape this stage:

* **The source span is authoritative for survival, not the chunk text.** A later
  check (``survive``, T-004) asks whether a located answer lies wholly within one
  chunk by comparing *spans*, never by fuzzy-matching chunk text. So the span a
  chunk carries must index :attr:`ExtractedDoc.text` exactly and must not move
  when the chunk's text is altered (``.ai/architecture/data-flow.md``).
* **Prepending section context is a configuration option, not a policy.** Whether
  a chunk's detected section path is prepended to its *text* before embedding is
  one of the knobs the tool exists to let people test, so it is a toggle on
  :class:`~ragdoc.config.RunConfig` (``prepend_section_context``), not a fixed
  behaviour (``.ai/architecture/components.md``). Toggling it changes chunk text
  but never chunk spans.
* **Chunking is offline and deterministic.** It depends only on the loader's
  output format and makes no network calls, so the same ``ExtractedDoc`` and
  ``RunConfig`` always yield the same chunks and the stage is testable without
  API keys or cost (``.ai/constitution.md``).

Section-path detection from raw extracted text is genuinely hard, and v0 only
needs it to be deterministic and good enough that prepending context can change a
chunk's embedded text. This module detects headings by a small, documented set of
line-shape rules (see :func:`_detect_heading`); it is deliberately conservative
rather than clever, and the recognised forms are part of this stage's contract so
fixtures (T-009) can be built to match.
"""

from __future__ import annotations

import re

from ragdoc.artifacts import CharSpan, Chunk, ExtractedDoc
from ragdoc.config import RunConfig

__all__ = ["Chunker"]


#: How the labels of a section path are joined when prepended to chunk text.
SECTION_PATH_SEPARATOR = " > "
#: What separates the prepended section context from the chunk's source text.
SECTION_CONTEXT_SEPARATOR = "\n\n"

#: A line longer than this (stripped) is treated as body text, never a heading.
#: Keeps multi-line list items and running prose — including the five-item list
#: of the split-list fixture, whose items are long enough that five of them split
#: a 500-char chunk — from being mistaken for headings.
_HEADING_MAX_LEN = 64

#: A numbered-section prefix: ``2``, ``2.``, ``2.1``, ``2.1.3)`` ... The nesting
#: depth is the count of dotted numeric components (``2`` → 1, ``2.1`` → 2).
_NUMBER_PREFIX_RE = re.compile(r"^(\d+(?:\.\d+)*)[.)]?\s+\S")

#: Trailing characters that mark a line as running prose rather than a heading.
_TERMINAL_PUNCTUATION = (".", "?", "!", ",", ";", ":")


def _is_title_like(s: str) -> bool:
    """True if ``s`` reads like an unnumbered heading (ALL CAPS or Title Case).

    Conservative on purpose: at most eight words, and either entirely uppercase
    or with every alphabetic word beginning with a capital. This is the rule that
    separates a short standalone heading ("Methods", "DATA SOURCES") from an
    ordinary short body line.
    """
    words = s.split()
    if not words or len(words) > 8:
        return False
    lettered = [w for w in words if any(c.isalpha() for c in w)]
    if not lettered:
        return False
    if s.isupper():
        return True
    return all(w[0].isupper() for w in lettered)


def _detect_heading(raw_line: str) -> tuple[int, str] | None:
    """Classify one physical line, returning ``(level, label)`` or ``None``.

    A line is a heading when, stripped, it is non-empty and at most
    :data:`_HEADING_MAX_LEN` characters, contains at least one letter, does not
    end in sentence punctuation, and is *either* numbered
    (:data:`_NUMBER_PREFIX_RE`) *or* title-like (:func:`_is_title_like`). The
    level is the numbered depth, or ``1`` for an unnumbered heading. The label is
    the stripped line as written (e.g. ``"2. Methods"``), so it carries the
    section number a reader would cite.
    """
    s = raw_line.strip()
    if not (0 < len(s) <= _HEADING_MAX_LEN):
        return None
    if not any(c.isalpha() for c in s):
        return None
    if s.endswith(_TERMINAL_PUNCTUATION):
        return None
    numbered = _NUMBER_PREFIX_RE.match(s)
    if numbered:
        return (numbered.group(1).count(".") + 1, s)
    if _is_title_like(s):
        return (1, s)
    return None


def _section_markers(text: str) -> list[tuple[int, tuple[str, ...]]]:
    """Scan ``text`` for headings, returning ``(offset, section_path)`` markers.

    Each marker records the character offset of a heading line's start and the
    full section path in effect from that offset onward, outermost heading first.
    Markers are returned in increasing-offset order. Nesting is maintained with a
    level stack: a heading pops every stack entry at its level or deeper before it
    is pushed, so ``2`` then ``2.1`` then ``3`` yields paths ``("2 …",)``,
    ``("2 …", "2.1 …")``, ``("3 …",)``.
    """
    markers: list[tuple[int, tuple[str, ...]]] = []
    stack: list[tuple[int, str]] = []
    offset = 0
    for raw_line in text.split("\n"):
        detected = _detect_heading(raw_line)
        if detected is not None:
            level, label = detected
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, label))
            markers.append((offset, tuple(label for _level, label in stack)))
        offset += len(raw_line) + 1  # + 1 for the newline ``split`` consumed
    return markers


def _path_at(markers: list[tuple[int, tuple[str, ...]]], offset: int) -> tuple[str, ...]:
    """Return the section path in effect at ``offset`` (``()`` before any heading).

    The governing heading is the last one whose line starts at or before
    ``offset``; ``markers`` is ordered, so this walks until it passes ``offset``.
    """
    path: tuple[str, ...] = ()
    for marker_offset, marker_path in markers:
        if marker_offset <= offset:
            path = marker_path
        else:
            break
    return path


class Chunker:
    """Splits an :class:`ExtractedDoc` into :class:`Chunk`\\ s with source spans.

    Owns the three settings that change how a document is chunked, read from the
    :class:`~ragdoc.config.RunConfig` it is constructed with so a run stays
    reproducible from its recorded configuration:

    * ``chunk_size`` — the target chunk length, in characters.
    * ``overlap`` — how many characters adjacent chunks share.
    * ``prepend_section_context`` — when true, a chunk's detected section path is
      prepended to its *text* (not its span) before it would be embedded.

    The chunker is offline and deterministic: the same ``ExtractedDoc`` and
    ``RunConfig`` always produce the same chunks.
    """

    def __init__(self, config: RunConfig) -> None:
        """Configure the chunker from ``config``.

        Raises:
            ValueError: if ``chunk_size`` is not positive, ``overlap`` is
                negative, or ``overlap`` is not smaller than ``chunk_size`` (which
                would leave a non-positive stride and fail to advance).
        """
        if config.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {config.chunk_size}")
        if config.overlap < 0:
            raise ValueError(f"overlap must not be negative, got {config.overlap}")
        if config.overlap >= config.chunk_size:
            raise ValueError(
                "overlap must be smaller than chunk_size "
                f"(got overlap={config.overlap}, chunk_size={config.chunk_size})"
            )
        self._chunk_size = config.chunk_size
        self._overlap = config.overlap
        self._prepend_section_context = config.prepend_section_context

    def chunk(self, doc: ExtractedDoc) -> list[Chunk]:
        """Split ``doc`` into chunks whose source spans tile its extracted text.

        Spans are a sliding window of width ``chunk_size`` advancing by
        ``chunk_size - overlap``; the first starts at offset 0 and the last ends
        at ``len(doc.text)``, so together they cover the whole text (with the
        configured overlap between neighbours). Each chunk also carries the
        1-based page its span starts on (via :meth:`ExtractedDoc.page_at`) and the
        section path detected at its start.

        When ``prepend_section_context`` is on and a chunk has a non-empty section
        path, that path is prepended to the chunk's *text*; the span is unchanged
        either way. An empty document yields an empty list.
        """
        text = doc.text
        if not text:
            return []

        markers = _section_markers(text)
        stride = self._chunk_size - self._overlap
        length = len(text)

        chunks: list[Chunk] = []
        start = 0
        index = 0
        while start < length:
            end = min(start + self._chunk_size, length)
            span = CharSpan(start, end)
            section_path = _path_at(markers, start)

            source = text[start:end]
            if self._prepend_section_context and section_path:
                chunk_text = (
                    SECTION_PATH_SEPARATOR.join(section_path)
                    + SECTION_CONTEXT_SEPARATOR
                    + source
                )
            else:
                chunk_text = source

            page = doc.page_at(start)
            if page is None:
                # Every offset in [0, len(text)) resolves against a complete page
                # map; this only triggers if the map is absent or incomplete.
                # Carry the previous chunk's page (or 1) so the field stays a
                # concrete, deterministic page number.
                page = chunks[-1].page if chunks else 1

            chunks.append(
                Chunk(
                    id=f"chunk-{index}",
                    text=chunk_text,
                    span=span,
                    page=page,
                    section_path=section_path,
                )
            )
            index += 1
            if end >= length:
                break
            start += stride

        return chunks
