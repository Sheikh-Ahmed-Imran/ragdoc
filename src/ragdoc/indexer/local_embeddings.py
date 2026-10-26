"""A local sentence-transformers embedding backend.

This is v0's one :class:`~ragdoc.indexer.Indexer` implementation: it embeds
chunks with a sentence-transformers model that runs on the local machine
(default ``all-MiniLM-L6-v2``) and holds the resulting vectors in memory for the
run. It is the only component that touches the network, and only to *download*
the model the first time it is used — once the model is cached, a run needs no
network at all (``.ai/architecture/components.md``).

The model is a real, heavy dependency (sentence-transformers pulls in
``torch``), so it is imported and loaded lazily — not when this module is
imported, but the first time embedding actually happens. That keeps the
:class:`~ragdoc.indexer.Indexer` interface and the retriever importable, and
their behaviour testable with a fake backend, without ``torch`` present. It also
means the model download is deferred to first use rather than import.

Vectors are held only in this object, in memory, and are gone when it is
discarded; there is no persistence between runs (spec Exclusions).
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

from ragdoc.artifacts import Chunk
from ragdoc.config import RunConfig
from ragdoc.indexer import Indexer, ModelLoadError

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime import
    from sentence_transformers import SentenceTransformer

__all__ = ["LocalEmbeddingsIndexer"]


# Quiet third-party model-load chatter *before* huggingface_hub is imported.
# huggingface_hub and transformers read these settings at their own import time,
# so the runtime calls in ``_quiet_hf_once`` (below) fire too late to catch the
# "Loading weights" progress bar or the "unauthenticated requests to the HF Hub"
# warning. Setting them here — at this module's import, which runs before the lazy
# ``from sentence_transformers import ...`` in ``_get_model`` pulls huggingface_hub
# in — is what actually keeps that chatter out of ragdoc's output. ``setdefault``
# leaves a value the user set deliberately untouched; the runtime calls remain as
# a best-effort fallback.
for _var, _val in (
    ("HF_HUB_DISABLE_PROGRESS_BARS", "1"),
    ("HF_HUB_VERBOSITY", "error"),
    ("TRANSFORMERS_VERBOSITY", "error"),
    ("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1"),
):
    os.environ.setdefault(_var, _val)
del _var, _val

#: The default local model: small, fast, offline after first download, and the
#: value :class:`~ragdoc.config.RunConfig` records as ``embedding_model`` by
#: default. Chosen to satisfy the spec's local-embeddings decision.
DEFAULT_MODEL = "all-MiniLM-L6-v2"


_hf_quieted = False


def _quiet_hf_once() -> None:
    """Silence huggingface_hub progress bars and warnings, once per process.

    Best effort: a missing or renamed API is ignored and never breaks a run.
    """
    global _hf_quieted
    if _hf_quieted:
        return
    _hf_quieted = True
    try:
        import huggingface_hub
        from huggingface_hub import logging as hf_logging

        huggingface_hub.disable_progress_bars()
        hf_logging.set_verbosity_error()
    except Exception:  # cosmetic only
        pass


def _model_cached(name: str) -> bool:
    """Best-effort local-cache probe (no network); True when unsure."""
    try:
        from huggingface_hub import try_to_load_from_cache

        repo_id = name if "/" in name else f"sentence-transformers/{name}"
        # ``None`` means the file is not in the cache; a path or the
        # "known missing" sentinel both mean the repo has been looked up before.
        return try_to_load_from_cache(repo_id, "modules.json") is not None
    except Exception:  # cosmetic only
        return True


class LocalEmbeddingsIndexer(Indexer):
    """Embeds chunks with a local sentence-transformers model, in memory.

    Construct from a :class:`~ragdoc.config.RunConfig` (so the model identifier
    is the one recorded with the run) or with an explicit model name. The model
    itself is not loaded until the first :meth:`index` or :meth:`embed_query`
    call, so constructing the indexer is cheap and makes no network call.

    The embeddings are L2-normalised at source (``normalize_embeddings=True``),
    which means a dot product between two of them already is their cosine
    similarity; the retriever still normalises defensively, so a backend that
    does not pre-normalise stays correct.
    """

    def __init__(
        self, config: Optional[RunConfig] = None, *, model_name: Optional[str] = None
    ) -> None:
        """Choose the model without loading it yet.

        Args:
            config: Run configuration; its ``embedding_model`` names the model.
            model_name: Explicit model name, overriding ``config`` when given.

        If neither is supplied, :data:`DEFAULT_MODEL` is used. The resolved name
        is exposed as :attr:`name` and is what should be recorded in
        ``RunConfig.embedding_model``.
        """
        if model_name is not None:
            resolved = model_name
        elif config is not None:
            resolved = config.embedding_model
        else:
            resolved = DEFAULT_MODEL
        self.name = resolved

        self._model: Optional["SentenceTransformer"] = None
        self._chunk_ids: list[str] = []
        self._vectors: np.ndarray = np.empty((0, 0), dtype=np.float32)

    def _get_model(self) -> "SentenceTransformer":
        """Load the model on first use (lazy import + lazy download).

        Importing sentence-transformers and loading the model happen here, not at
        module import, so nothing that merely imports this module needs ``torch``
        or a network connection. The model is loaded once and reused.
        """
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            _quiet_hf_once()
            if not _model_cached(self.name):
                sys.stderr.write(
                    f"Downloading embedding model '{self.name}' (first run, ~90 MB); "
                    f"cached offline after this.\n"
                )

            try:
                self._model = SentenceTransformer(self.name)
            except Exception as exc:  # library error classes vary; translate at this one boundary
                raise ModelLoadError(
                    f"embedding model {self.name!r} is not cached and could not be "
                    f"downloaded (no network access, or it may not be a valid model "
                    f"name): {type(exc).__name__}: {' '.join(str(exc).split())}"
                ) from exc
        return self._model

    def index(self, chunks: Sequence[Chunk]) -> None:
        """Embed ``chunks`` and hold their vectors in memory, keyed by id.

        Replaces any previously indexed chunks. An empty sequence clears the
        index and loads no model.
        """
        self._chunk_ids = [chunk.id for chunk in chunks]
        if not chunks:
            self._vectors = np.empty((0, 0), dtype=np.float32)
            return

        model = self._get_model()
        vectors = model.encode(
            [chunk.text for chunk in chunks],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        self._vectors = np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed one string into the chunk vector space (1-D vector)."""
        model = self._get_model()
        vector = model.encode(
            [text], normalize_embeddings=True, convert_to_numpy=True
        )
        return np.asarray(vector, dtype=np.float32)[0]

    @property
    def chunk_ids(self) -> Sequence[str]:
        """Indexed chunk ids, in the order their vectors are stored."""
        return list(self._chunk_ids)

    @property
    def vectors(self) -> np.ndarray:
        """The ``(n_chunks, dim)`` matrix of chunk vectors (row i ↔ chunk_ids[i])."""
        return self._vectors
