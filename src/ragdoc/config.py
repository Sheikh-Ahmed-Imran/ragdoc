"""Run configuration.

A diagnosis must be reproducible from the configuration recorded with it: two
runs over the same PDF with the same ``RunConfig`` must produce the same verdict
(``.ai/constitution.md`` — "Every run is reproducible from its recorded
configuration"). So ``RunConfig`` carries *every* value that can change a
verdict — loader, chunk size, overlap, whether section context is prepended
before embedding, embedding model, and ``top_k`` — and nothing that cannot.

It is frozen because a run's configuration is a record of what happened, not a
mutable scratchpad; freezing it keeps it safe to attach to a verdict and
compare across runs.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["RunConfig"]


@dataclass(frozen=True)
class RunConfig:
    """Everything that affects a verdict, recorded so a run can be reproduced.

    Attributes:
        loader: Name of the loader implementation used for extraction. The page
            positions in a verdict are only meaningful paired with the loader
            that produced them, so the loader is part of the recorded config.
        chunk_size: Target chunk length the chunker splits to. Default 500.
        overlap: Overlap between adjacent chunks. Default 50.
        prepend_section_context: When true, the chunker prepends a chunk's
            section path/heading to its text before it is handed to embedding.
            This changes chunk *text* but not chunk *spans*; it is a toggle the
            tool exists to let people test, not a fixed policy.
        embedding_model: Name/identifier of the embedding model the indexer
            uses. Recorded because the ranking a verdict cites depends on it.
        top_k: How many chunks the retriever's normal path would return. The
            full ranking is always computed; ``top_k`` is the cutoff a rank is
            compared against. Default 10.
    """

    loader: str = "pymupdf"
    chunk_size: int = 500
    overlap: int = 50
    prepend_section_context: bool = False
    embedding_model: str = "all-MiniLM-L6-v2"
    top_k: int = 10
