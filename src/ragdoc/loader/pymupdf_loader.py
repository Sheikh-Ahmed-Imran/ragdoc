"""PyMuPDF implementation of the :class:`~ragdoc.loader.Loader` interface.

PyMuPDF is the v0 extraction library (plan step 2; it is the library from the
author's prior pipeline). This module is the only place that imports it, so the
choice stays owned by the loader and nothing else in the system depends on it
(``.ai/architecture/components.md``).

How the page map is built: each page's extracted text is appended to one growing
string, and the half-open ``[start, end)`` character span it occupies in that
string is recorded as a :class:`PageSpan`. Pages are concatenated in document
order with no inserted separators, so the recorded offsets index the returned
text exactly — the text is authoritative for position
(``.ai/architecture/data-flow.md``). The resulting spans are ordered,
contiguous and non-overlapping, which is what :meth:`ExtractedDoc.page_at`
relies on to resolve any offset to the right page.

Extraction is local and deterministic: PyMuPDF reads the file from disk and
makes no network calls, and the same PDF yields the same text and map every run
(``.ai/constitution.md``).
"""

from __future__ import annotations

from pathlib import Path

try:  # PyMuPDF >= 1.24 exposes the top-level ``pymupdf`` name.
    import pymupdf
except ImportError:  # pragma: no cover - older PyMuPDF only ships ``fitz``.
    import fitz as pymupdf

from ragdoc.artifacts import CharSpan, ExtractedDoc, PageSpan
from ragdoc.loader import Loader, LoaderError

__all__ = ["PyMuPDFLoader"]


class PyMuPDFLoader(Loader):
    """Extracts a PDF into text plus a page map using PyMuPDF.

    The loader reads text in the document's natural reading order (PyMuPDF's
    default ``"text"`` extraction) and records, for each page, the character
    span its text occupies in the concatenated output.
    """

    name = "pymupdf"

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
            with pymupdf.open(pdf_path) as doc:
                # PyMuPDF sniffs content and also opens images, XPS, EPUB, ...
                # as "documents"; only a real PDF is accepted here.
                if not doc.is_pdf:
                    raise LoaderError(f"{pdf_path} is not a readable PDF: not in PDF format")
                if doc.needs_pass:
                    raise LoaderError(f"encrypted / password-protected PDF: {pdf_path}")
                for index, page in enumerate(doc, start=1):
                    page_text = page.get_text()
                    start = offset
                    end = start + len(page_text)
                    parts.append(page_text)
                    page_map.append(PageSpan(page=index, span=CharSpan(start, end)))
                    offset = end
        except (
            pymupdf.FileDataError,
            pymupdf.EmptyFileError,
            pymupdf.mupdf.FzErrorFormat,  # content sniffed as a format but malformed
        ) as exc:
            raise LoaderError(f"{pdf_path} is not a readable PDF: {exc}") from exc

        return ExtractedDoc(text="".join(parts), page_map=page_map)
