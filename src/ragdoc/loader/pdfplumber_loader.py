"""pdfplumber implementation of the :class:`~ragdoc.loader.Loader` interface.

pdfplumber is the second extraction library, added so extraction can be compared
across libraries. This module is the only place that imports it, so the choice
stays owned by the loader (``.ai/architecture/components.md``).

The page map is built exactly as in the PyMuPDF loader: each page's extracted
text is appended to one growing string with no separators, and the half-open
``[start, end)`` span it occupies is recorded as a :class:`PageSpan`. The
recorded offsets therefore index the returned text exactly, and are built from
the concatenated output, never from a separate re-read of the PDF
(``.ai/architecture/data-flow.md``).

Extraction is local and deterministic: no network calls, and the same PDF yields
the same text and map every run (``.ai/constitution.md``). No OCR is attempted;
a page with no text layer contributes an empty span.
"""

from __future__ import annotations

from pathlib import Path

import pdfplumber

from ragdoc.artifacts import CharSpan, ExtractedDoc, PageSpan
from ragdoc.loader import Loader

__all__ = ["PdfPlumberLoader"]


class PdfPlumberLoader(Loader):
    """Extracts a PDF into text plus a page map using pdfplumber."""

    name = "pdfplumber"

    def load(self, path: str | Path) -> ExtractedDoc:
        """Extract ``path`` into an :class:`ExtractedDoc` with a page map.

        Raises:
            FileNotFoundError: if ``path`` does not point at an existing file.
        """
        pdf_path = Path(path)
        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        parts: list[str] = []
        page_map: list[PageSpan] = []
        offset = 0

        with pdfplumber.open(pdf_path) as pdf:
            for index, page in enumerate(pdf.pages, start=1):
                # extract_text() returns None for a page with no text layer.
                page_text = page.extract_text() or ""
                start = offset
                end = start + len(page_text)
                parts.append(page_text)
                page_map.append(PageSpan(page=index, span=CharSpan(start, end)))
                offset = end

        return ExtractedDoc(text="".join(parts), page_map=page_map)
