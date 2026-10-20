"""The retriever stage: a question becomes a full ranking of every chunk.

The retriever is the fourth forward-pipeline stage and the first to see the
question. It embeds the question through the :class:`~ragdoc.indexer.Indexer`
(so the question lands in the same vector space as the chunks), scores it
against *every* indexed chunk by brute-force cosine, and returns a complete
:class:`~ragdoc.artifacts.Ranking` ordered best-first. It owns ``top_k`` — the
cutoff the normal retrieval path would apply — but the ranking it returns is the
whole ordering, never the ``top_k`` slice.

Two rules from the architecture shape this stage:

* **The full ranking is authoritative for retrieval, not the ``top_k`` slice.** A
  verdict needs to state "ranked 23 of 400" even when the target chunk is far
  below the cutoff, so the retriever must expose any chunk's absolute position
  independent of ``top_k`` (``.ai/architecture/components.md``,
  ``.ai/architecture/data-flow.md``). :meth:`rank` returns every chunk, and
  :meth:`position_of` reports an absolute 1-based rank regardless of where the
  cutoff falls.
* **Scoring is exact, not approximate.** Cosine is computed by brute force over
  the whole vector matrix — no approximate-nearest-neighbour store — so a
  reported rank is the true rank, which is the entire point of exposing it
  (plan Decisions).

The retriever does **not** decide a verdict or compare a rank to ``top_k`` for
attribution. It reports the ranking and the position; the ``rank`` check (T-006)
interprets them.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ragdoc.artifacts import Ranking, RankedChunk
from ragdoc.config import RunConfig
from ragdoc.indexer import Indexer

__all__ = ["Retriever"]


class Retriever:
    """Scores a question against an index and returns the full ranking.

    Construct from an :class:`~ragdoc.indexer.Indexer` (already holding chunk
    vectors, or ready to embed the question) and a
    :class:`~ragdoc.config.RunConfig` (from which it reads ``top_k``). The
    indexer must have been given its chunks — via :meth:`Indexer.index` — before
    :meth:`rank` is called.
    """

    def __init__(self, indexer: Indexer, config: Optional[RunConfig] = None) -> None:
        """Wire the retriever to an index and read ``top_k`` from ``config``.

        Args:
            indexer: The index holding (or ready to hold) chunk vectors and able
                to embed the question into the same space.
            config: Run configuration; ``top_k`` is read from it. When omitted,
                ``RunConfig``'s default ``top_k`` applies.
        """
        self._indexer = indexer
        self._top_k = (config or RunConfig()).top_k

    @property
    def top_k(self) -> int:
        """The cutoff the normal retrieval path would apply.

        The retriever owns this; it does not change the full ranking :meth:`rank`
        returns, it is only the point a rank is later compared against (T-006).
        """
        return self._top_k

    def rank(self, question: str) -> Ranking:
        """Score ``question`` against every indexed chunk, best-first.

        Embeds the question through the indexer, computes cosine similarity
        against the whole chunk-vector matrix by brute force, and returns a
        :class:`~ragdoc.artifacts.Ranking` whose ``entries`` cover *all* indexed
        chunks ordered by descending score. Ties are broken by index order so the
        ranking is deterministic for a given index.

        An empty index yields an empty ``Ranking``. The returned ranking is the
        full ordering; applying ``top_k`` is the caller's concern
        (see :meth:`top_ids`), and the ranking stands on its own for
        :meth:`position_of`.
        """
        chunk_ids = list(self._indexer.chunk_ids)
        if not chunk_ids:
            return Ranking(entries=[])

        matrix = np.asarray(self._indexer.vectors, dtype=np.float32)
        query = np.asarray(self._indexer.embed_query(question), dtype=np.float32)

        scores = _cosine_scores(matrix, query)

        # Descending score, ties broken by original index order. ``argsort`` is
        # ascending; negating the scores and using a stable sort gives
        # descending-by-score, then index-ascending on ties.
        order = np.argsort(-scores, kind="stable")
        entries = [
            RankedChunk(chunk_id=chunk_ids[i], score=float(scores[i])) for i in order
        ]
        return Ranking(entries=entries)

    def top_ids(self, ranking: Ranking) -> list[str]:
        """The chunk ids the normal retrieval path would return: the top ``top_k``.

        A convenience over the full ``ranking`` — it does not re-score anything.
        Returns fewer than ``top_k`` ids only when the index has fewer chunks.
        """
        return [entry.chunk_id for entry in ranking.entries[: self._top_k]]

    def position_of(self, ranking: Ranking, chunk_id: str) -> Optional[int]:
        """The absolute 1-based rank of ``chunk_id``, or ``None`` if absent.

        Independent of ``top_k``: a chunk ranked far below the cutoff still
        reports its true position (e.g. ``23``), and ``len(ranking)`` gives the
        total (the "of 400"), so the pair expresses "ranked 23 of 400". This is
        what the ``rank`` check (T-006) reads before comparing against
        :attr:`top_k`.
        """
        return ranking.position_of(chunk_id)


def _cosine_scores(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Cosine similarity of ``query`` against every row of ``matrix``.

    Normalises defensively — the local backend already returns unit vectors, but
    a backend that does not must still score correctly. A zero-norm vector (no
    direction to compare) scores ``0`` rather than producing ``NaN`` from a
    divide-by-zero.
    """
    row_norms = np.linalg.norm(matrix, axis=1)
    query_norm = np.linalg.norm(query)

    safe_rows = np.where(row_norms == 0.0, 1.0, row_norms)
    normalised = matrix / safe_rows[:, None]
    normalised[row_norms == 0.0] = 0.0

    if query_norm == 0.0:
        return np.zeros(matrix.shape[0], dtype=np.float32)
    unit_query = query / query_norm

    return normalised @ unit_query
