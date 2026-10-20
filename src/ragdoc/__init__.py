"""ragdoc — headless attribution engine for PDF RAG pipelines.

This top level re-exports the run configuration and the artifact/result types
that form the shared vocabulary of the pipeline and the attribution engine, so
callers can ``from ragdoc import RunConfig, ExtractedDoc, Chunk, Ranking,
CheckResult`` without reaching into submodules.
"""

from __future__ import annotations

from ragdoc.artifacts import (
    CharSpan,
    Chunk,
    CheckResult,
    CheckStatus,
    ExtractedDoc,
    PageSpan,
    Ranking,
    RankedChunk,
)
from ragdoc.config import RunConfig

__all__ = [
    "RunConfig",
    "CharSpan",
    "PageSpan",
    "ExtractedDoc",
    "Chunk",
    "RankedChunk",
    "Ranking",
    "CheckStatus",
    "CheckResult",
]
