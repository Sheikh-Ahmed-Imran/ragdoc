"""pypdf implementation of the :class:`~ragdoc.loader.Loader` interface.

pypdf is a lightweight, pure-Python extractor included as a contrast to PyMuPDF
in the loader comparison. This module is the only place that imports it
(``.ai/architecture/components.md``). A weak or empty extraction on some PDFs is
expected and deliberately not worked around: surfacing a weak loader is the
point, so there are no fallbacks to other libraries.

The page map is built exactly as in the PyMuPDF loader: each page's extracted
text is appended to one growing string with no separators, and the half-open
``[start, end)`` span it occupies is recorded as a :class:`PageSpan`. Offsets
come from the concatenated output, so they index the returned text exactly
(``.ai/architecture/data-flow.md``).

Extraction is local and deterministic: pypdf reads the file from disk and makes
no network calls (``.ai/constitution.md``).
"""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import EmptyFileError, PdfReadError

from ragdoc.artifacts import CharSpan, ExtractedDoc, PageSpan
from ragdoc.loader import Loader, LoaderError

__all__ = ["PyPDFLoader"]


class PyPDFLoader(Loader):
    """Extracts a PDF into text plus a page map using pypdf."""

    name = "pypdf"

    def load(self, path: str | Path) -> ExtractedDoc:
        """Extract ``path`` into an :class:`ExtractedDoc` with a page map.

        Raises:
            FileNotFoundError: if ``path`` does not point at an existing file.
            LoaderError: if the file exists but cannot be read or parsed as a PDF.
        """
        pdf_path = Path(path)
        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        parts: list[str] = []
        page_map: list[PageSpan] = []
        offset = 0

        try:
            reader = PdfReader(pdf_path)
            if reader.is_encrypted:
                raise LoaderError(f"encrypted / password-protected PDF: {pdf_path}")
            for index, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                start = offset
                end = start + len(page_text)
                parts.append(page_text)
                page_map.append(PageSpan(page=index, span=CharSpan(start, end)))
                offset = end
        except (PdfReadError, EmptyFileError) as exc:
            raise LoaderError(f"{pdf_path} is not a readable PDF: {exc}") from exc

        return ExtractedDoc(text="".join(parts), page_map=page_map)
