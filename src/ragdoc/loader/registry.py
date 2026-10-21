"""Name-based registry of :class:`~ragdoc.loader.Loader` implementations.

A loader's ``name`` is the value recorded in ``RunConfig.loader``, and a page
position is only meaningful paired with the loader that produced it
(``.ai/architecture/data-flow.md``). So the registry key is exactly the loader's
own ``name`` — there is no separate alias — and a recorded name always resolves
back to the implementation that produced it.

New loaders register themselves with :func:`register` (usable as a class
decorator); nothing outside the loader package needs editing. This module
imports only from the loader package, keeping the stage boundary intact
(``.ai/constitution.md``).
"""

from __future__ import annotations

from ragdoc.loader import Loader
from ragdoc.loader.pdfplumber_loader import PdfPlumberLoader
from ragdoc.loader.pymupdf_loader import PyMuPDFLoader
from ragdoc.loader.pypdf_loader import PyPDFLoader

__all__ = ["UnknownLoaderError", "register", "get", "names"]

_REGISTRY: dict[str, type[Loader]] = {}


class UnknownLoaderError(ValueError):
    """Raised when a loader name is not registered."""


def register(loader_cls: type[Loader]) -> type[Loader]:
    """Register ``loader_cls`` under its ``name``; returns the class unchanged.

    Raises:
        ValueError: if the class has no non-empty ``name``, or the name is
            already registered to a different class.
    """
    name = loader_cls.name
    if not isinstance(name, str) or not name:
        raise ValueError(f"{loader_cls.__name__} must define a non-empty `name`")
    existing = _REGISTRY.get(name)
    if existing is not None and existing is not loader_cls:
        raise ValueError(
            f"loader name {name!r} is already registered to {existing.__name__}"
        )
    _REGISTRY[name] = loader_cls
    return loader_cls


def get(name: str) -> Loader:
    """Return a new instance of the loader registered as ``name``.

    Raises:
        UnknownLoaderError: if ``name`` is not registered; the message names the
            bad value and lists the registered names.
    """
    try:
        loader_cls = _REGISTRY[name]
    except KeyError:
        raise UnknownLoaderError(
            f"unknown loader {name!r}; registered loaders: {', '.join(names())}"
        ) from None
    return loader_cls()


def names() -> list[str]:
    """Return the registered loader names, sorted."""
    return sorted(_REGISTRY)


register(PyMuPDFLoader)
register(PdfPlumberLoader)
register(PyPDFLoader)
