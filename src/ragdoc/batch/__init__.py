"""Batch mode: read a user-supplied question set.

The input is a JSON array of objects, each with a non-empty string ``question``
and a non-empty string ``expected_answer``. Order is preserved.

Rules this module keeps:

* **Ground truth is user-supplied.** Pairs are only read, never generated or
  inferred (``.ai/constitution.md``, ADR-002).
* **JSON only.** No CSV or JSONL.
* **Headless.** It imports neither the CLI nor any view layer.
* **The forward pass runs once.** :func:`run_batch` builds the loader, chunker and
  index a single time; only the question-dependent work repeats per pair.
* **No new attribution logic.** Each pair goes through the v0 engine and
  ``compose_verdict`` unchanged; this module only buckets the resulting verdicts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence, Union

from ragdoc.attribution.verdict import Verdict, build_engine, compose_verdict
from ragdoc.chunker import Chunker
from ragdoc.config import RunConfig
from ragdoc.indexer.local_embeddings import LocalEmbeddingsIndexer
from ragdoc.loader import get as get_loader
from ragdoc.retriever import Retriever

__all__ = [
    "BUCKETS",
    "BatchInputError",
    "BatchResult",
    "PreparedBatch",
    "QuestionPair",
    "QuestionResult",
    "load_question_set",
    "prepare_batch",
    "run_batch",
    "score_batch",
]

#: Every bucket a verdict can land in, in report order. There is deliberately no
#: ``generation`` bucket: generation is unbuilt and never a verdict stage.
BUCKETS = ("extraction", "chunking", "retrieval", "withheld", "no-failure")


class BatchInputError(ValueError):
    """The question-set file is malformed."""


@dataclass(frozen=True)
class QuestionPair:
    """One asserted question and the answer the user says it should return."""

    question: str
    expected_answer: str


def load_question_set(path: Union[str, Path]) -> list[QuestionPair]:
    """Read and validate a JSON question set, preserving input order."""
    path = Path(path)
    try:
        # utf-8-sig strips a leading BOM and reads BOM-less files identically.
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise BatchInputError(f"cannot read question set {path}: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise BatchInputError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, list):
        raise BatchInputError(
            f"{path}: question set must be a JSON array, got {type(data).__name__}"
        )

    pairs: list[QuestionPair] = []
    for index, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise BatchInputError(
                f"entry {index}: must be an object, got {type(entry).__name__}"
            )
        values = {}
        for field in ("question", "expected_answer"):
            value = entry.get(field)
            if field not in entry:
                raise BatchInputError(f"entry {index}: missing '{field}'")
            if not isinstance(value, str):
                raise BatchInputError(
                    f"entry {index}: '{field}' must be a string, got {type(value).__name__}"
                )
            if not value.strip():
                raise BatchInputError(f"entry {index}: '{field}' is empty")
            values[field] = value
        pairs.append(QuestionPair(**values))
    return pairs


@dataclass(frozen=True)
class QuestionResult:
    """The outcome for one pair, in input order.

    ``bucket`` is one of :data:`BUCKETS`. ``stage`` is the verdict's stage
    (``None`` for ``withheld`` / ``no-failure``). ``evidence`` merges the evidence
    of every evaluated stage (``answer_page`` from extraction, ``chunk_page`` from
    chunking, ``snippet``, ``position``, ``total``, ``top_k``, ...) so the reporter needs nothing else; ``verdict`` is the full
    :class:`Verdict` for anything further (``fix``, ``reason_code``, ``stages``).
    """

    question: str
    expected_answer: str
    bucket: str
    stage: Any
    reason: Any
    evidence: Mapping[str, Any]
    verdict: Verdict


@dataclass(frozen=True)
class BatchResult:
    """Per-question results in input order plus the count per bucket."""

    results: tuple[QuestionResult, ...]
    counts: Mapping[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return len(self.results)


def _bucket(verdict: Verdict) -> str:
    if verdict.withheld:
        return "withheld"
    if verdict.stage is None:
        return "no-failure"
    if verdict.stage in ("extraction", "chunking", "retrieval"):
        return verdict.stage
    # An unknown stage is never guessed into a bucket (constitution: no verdict
    # without evidence).
    return "withheld"


def _evidence(verdict: Verdict) -> dict[str, Any]:
    # Extraction and chunking both record ``page`` (the answer's page vs. the target
    # chunk's start page); keep them under distinct keys so the later stage cannot
    # overwrite the earlier one.
    page_keys = {"extraction": "answer_page", "chunking": "chunk_page"}
    merged: dict[str, Any] = {}
    for report in verdict.stages:
        if report.result is None:
            continue
        evidence = dict(report.result.evidence)
        key = page_keys.get(report.stage)
        if key is not None and "page" in evidence:
            evidence[key] = evidence.pop("page")
        merged.update(evidence)
    return merged


@dataclass(frozen=True)
class PreparedBatch:
    """The ``top_k``-independent forward state for one PDF and one question list.

    ``rankings[i]`` is the full ranking (every chunk) for ``pairs[i]``. ``config``
    is the config the forward pass was built from; :func:`score_batch` overrides
    only its ``top_k``.
    """

    doc: Any
    chunks: Any
    rankings: tuple[Any, ...]
    config: RunConfig


def prepare_batch(
    pdf: Union[str, Path],
    config: RunConfig,
    pairs: Sequence[QuestionPair],
) -> PreparedBatch:
    """Build the forward pass once: load, chunk, embed and rank every pair.

    An empty question set builds nothing (no loader, chunker, indexer or ranking).
    """
    if not pairs:
        return PreparedBatch(doc=None, chunks=(), rankings=(), config=config)
    doc = get_loader(config.loader).load(pdf)
    chunks = Chunker(config).chunk(doc)
    indexer = LocalEmbeddingsIndexer(config)
    indexer.index(chunks)
    retriever = Retriever(indexer, config)
    rankings = tuple(retriever.rank(pair.question) for pair in pairs)
    return PreparedBatch(doc=doc, chunks=chunks, rankings=rankings, config=config)


def score_batch(
    prepared: PreparedBatch,
    pairs: Sequence[QuestionPair],
    top_k: int,
) -> BatchResult:
    """Score prepared forward state at ``top_k``; no loading, embedding or ranking."""
    if len(pairs) != len(prepared.rankings):
        raise ValueError(
            f"prepared batch holds {len(prepared.rankings)} rankings "
            f"but {len(pairs)} pairs were given"
        )
    config = replace(prepared.config, top_k=top_k)
    engine = build_engine()

    counts = {name: 0 for name in BUCKETS}
    results: list[QuestionResult] = []
    for pair, ranking in zip(pairs, prepared.rankings):
        verdict = compose_verdict(
            engine.run(
                expected_answer=pair.expected_answer,
                doc=prepared.doc,
                chunks=prepared.chunks,
                ranking=ranking,
                config=config,
            )
        )
        bucket = _bucket(verdict)
        counts[bucket] += 1
        results.append(
            QuestionResult(
                question=pair.question,
                expected_answer=pair.expected_answer,
                bucket=bucket,
                stage=verdict.stage,
                reason=verdict.reason,
                evidence=_evidence(verdict),
                verdict=verdict,
            )
        )
    return BatchResult(results=tuple(results), counts=counts)


def run_batch(
    pdf: Union[str, Path],
    config: RunConfig,
    pairs: Sequence[QuestionPair],
) -> BatchResult:
    """Attribute every pair against one PDF, building the forward pass once."""
    return score_batch(prepare_batch(pdf, config, pairs), pairs, config.top_k)
