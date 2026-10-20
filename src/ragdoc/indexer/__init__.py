"""The indexer stage: chunks become vectors held in memory for the run.

The indexer is the third forward-pipeline stage. It embeds each
:class:`~ragdoc.artifacts.Chunk` the chunker produced into a dense vector and
holds those vectors in memory, keyed by chunk id, for the duration of the run.
It also embeds an arbitrary query string into the *same* vector space, which is
how the retriever (T-005) turns a question into something comparable to the
chunk vectors.

Three rules from the architecture shape this interface:

* **The indexer is the only component that makes network calls, and only at
  model download/install time.** Embedding runs locally; a model is fetched from
  the hub the first time it is needed and cached, after which a run needs no
  network at all. Extraction and chunking stay offline precisely because the
  networked behaviour is isolated here (``.ai/architecture/components.md``). To
  keep that isolation real, this interface module imports no embedding library;
  a concrete backend (see :mod:`ragdoc.indexer.local_embeddings`) owns the model
  and imports its dependencies lazily, so importing the interface — or a test
  double of it — never drags in ``torch``.
* **The embedding model is part of the recorded configuration.** The ranking a
  verdict cites depends on which model produced the vectors, so the model
  identifier travels on :attr:`RunConfig.embedding_model` and a backend records
  it as its :attr:`name` (``.ai/architecture/data-flow.md``).
* **Vectors live in memory per run.** There is no persistence, cache, or on-disk
  index between runs by design (spec Exclusions).

Several embedding backends are expected behind this one interface — an
API-based provider is a conceivable v1 option — so the interface exists to keep
that swap open. v0 ships exactly one backend, the local sentence-transformers
model in :mod:`ragdoc.indexer.local_embeddings`; an API-based provider is
explicitly *not* the default (spec Decisions).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Sequence

from ragdoc.artifacts import Chunk

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime import
    import numpy as np

__all__ = ["Indexer"]


class Indexer(ABC):
    """Abstract indexer: embeds chunks and holds their vectors for the run.

    A backend owns the embedding model and whatever it keeps vectors in; this
    interface fixes only the contract the retriever depends on. The contract
    every backend must honour:

    * :meth:`index` embeds the given chunks and holds one vector per chunk in
      memory, keyed by chunk id, for the rest of the run. Calling it replaces any
      previously indexed chunks (one forward pipeline, one index).
    * After :meth:`index`, :attr:`chunk_ids` lists the indexed chunk ids in a
      fixed order and :attr:`vectors` is the matrix of their vectors, row ``i``
      being the vector for ``chunk_ids[i]``. The retriever scores against this
      matrix directly.
    * :meth:`embed_query` embeds an arbitrary string into the *same* vector
      space, so a question's vector is comparable to the chunk vectors. It does
      not require :meth:`index` to have been called.
    * Embedding may download a model the first time it runs, but a run must not
      otherwise depend on the network.

    Vectors are returned as :class:`numpy.ndarray` so the retriever can score by
    brute-force cosine over the whole matrix at once; backends built on other
    array libraries must convert.
    """

    #: Stable identifier for this backend's model, recorded in
    #: ``RunConfig.embedding_model``. A ranking is only meaningful paired with the
    #: model that produced it (``.ai/architecture/data-flow.md``).
    name: str = ""

    @abstractmethod
    def index(self, chunks: Sequence[Chunk]) -> None:
        """Embed ``chunks`` and hold one vector per chunk in memory.

        Args:
            chunks: The chunks to embed, in document order. Each chunk's
                :attr:`Chunk.text` (which already reflects the
                ``prepend_section_context`` toggle) is what gets embedded; its
                :attr:`Chunk.id` is the key the vector is held under.

        Replaces any previously indexed chunks. An empty sequence leaves the
        index empty.
        """
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, text: str) -> "np.ndarray":
        """Embed an arbitrary string into the same vector space as the chunks.

        Returns a 1-D vector of the backend's embedding dimension. Used by the
        retriever to turn a question into something comparable to the indexed
        chunk vectors; does not depend on :meth:`index` having run.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def chunk_ids(self) -> Sequence[str]:
        """Ids of the indexed chunks, in the order their vectors are stored.

        Empty before :meth:`index` is called (or after it is called with no
        chunks). ``chunk_ids[i]`` is the chunk whose vector is row ``i`` of
        :attr:`vectors`.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def vectors(self) -> "np.ndarray":
        """The indexed chunk vectors as a 2-D ``(n_chunks, dim)`` matrix.

        Row ``i`` is the vector for ``chunk_ids[i]``. Empty (shape ``(0, …)`` or
        ``(0,)``) before :meth:`index` is called.
        """
        raise NotImplementedError
