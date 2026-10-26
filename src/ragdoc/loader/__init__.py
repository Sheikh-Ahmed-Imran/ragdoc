"""The loader stage: a PDF becomes extracted text plus a page map.

The loader is the first forward-pipeline stage. It owns the choice of extraction
library and the per-page character offsets that let any position in the
extracted text be traced back to a page number
(``.ai/architecture/components.md``).

Two rules from the architecture shape this interface:

* **The extracted text is authoritative for position.** Page numbers in a later
  verdict come from the :class:`~ragdoc.artifacts.ExtractedDoc` page map this
  stage produces, never from re-reading the PDF
  (``.ai/architecture/data-flow.md``). So a loader must return text and map
  together, as one artifact, and the offsets in the map must index exactly the
  text it returns.
* **Loading is offline and deterministic.** It makes no network calls, and the
  same PDF yields the same text and the same map every time
  (``.ai/constitution.md`` — extraction is testable without API keys or cost).

Several loader implementations are expected behind this one interface, because
comparing loaders is a v1 feature; v0 ships exactly one
(:class:`~ragdoc.loader.pymupdf_loader.PyMuPDFLoader`). The interface exists so
that comparison stays open without the rest of the system depending on any
concrete loader.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ragdoc.artifacts import ExtractedDoc

__all__ = ["Loader", "LoaderError", "UnknownLoaderError", "register", "get", "names"]


class LoaderError(ValueError):
    """The file exists but could not be read or parsed as a PDF.

    Raised by every loader in place of its extraction library's own exception
    (corrupt PDF, empty file, another format renamed to ``.pdf``). It subclasses
    :class:`ValueError`, so the CLI reports it as a one-line ``ragdoc: error:``
    message with exit code 2 and no traceback.
    """


class Loader(ABC):
    """Abstract loader: turns a PDF path into an :class:`ExtractedDoc`.

    Subclasses own the extraction library and how per-page offsets are computed.
    The contract every subclass must honour:

    * :meth:`load` returns an :class:`ExtractedDoc` whose ``page_map`` covers its
      ``text`` with ordered, non-overlapping, 1-based :class:`PageSpan`\\ s, so
      :meth:`ExtractedDoc.page_at` resolves any in-range offset to the page it
      came from.
    * No network access, and deterministic: the same PDF produces the same text
      and the same map.
    * :meth:`load` raises :class:`LoaderError` when the file exists but cannot be
      read or parsed as a PDF, and :class:`FileNotFoundError` when it does not
      exist. An encrypted / password-protected PDF is a distinct sub-case: the
      :class:`LoaderError` message says ``encrypted / password-protected PDF``
      rather than the generic ``not a readable PDF``. Loaders never attempt a
      password or decryption.

    :attr:`name` identifies the implementation and is the value recorded in
    :attr:`RunConfig.loader`; a page position is only meaningful paired with the
    loader that produced it (``.ai/architecture/data-flow.md``).
    """

    #: Stable identifier for this loader, recorded in ``RunConfig.loader``.
    name: str = ""

    @abstractmethod
    def load(self, path: str | Path) -> ExtractedDoc:
        """Extract ``path`` into an :class:`ExtractedDoc` with a page map.

        Args:
            path: Filesystem path to a PDF file.

        Returns:
            An :class:`ExtractedDoc` whose ``text`` is the full extracted text
            and whose ``page_map`` maps every character offset in that text back
            to its 1-based page number.

        Raises:
            FileNotFoundError: if ``path`` does not point at an existing file.
            LoaderError: if the file exists but cannot be read or parsed as a PDF.
        """
        raise NotImplementedError


# Imported last: the registry (and the loaders it registers) import ``Loader``
# from this module, so it must be defined before they load.
from ragdoc.loader.registry import UnknownLoaderError, get, names, register  # noqa: E402
