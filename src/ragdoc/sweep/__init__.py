"""Config sweep: run a full batch for every config in a grid.

The grid is the cartesian product of per-knob value lists; a knob given no list
uses its :class:`~ragdoc.config.RunConfig` default (one value).

Rules this module keeps:

* **One forward pass per base.** Configs that differ only in ``top_k`` share a
  base (every other knob), so :func:`~ragdoc.batch.prepare_batch` runs once per
  base and :func:`~ragdoc.batch.score_batch` once per ``top_k`` in it.
* **No new attribution logic.** Scoring is the batch seam, unchanged.
* **Headless.** It prints nothing and imports no CLI or view layer; progress goes
  to an optional callback.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Optional, Sequence, Union

from ragdoc.batch import BatchResult, QuestionPair, prepare_batch, score_batch
from ragdoc.config import RunConfig

__all__ = ["SweepEntry", "SweepResult", "build_grid", "count_forward_passes", "run_sweep"]

_DEFAULTS = RunConfig()


@dataclass(frozen=True)
class SweepEntry:
    """One grid config and the batch it produced."""

    config: RunConfig
    batch: BatchResult


@dataclass(frozen=True)
class SweepResult:
    """Entries in grid order, plus the config and forward-pass counts."""

    entries: tuple[SweepEntry, ...]
    config_count: int
    forward_passes: int


def _values(given: Optional[Sequence], default) -> list:
    if given is None:
        return [default]
    # Drop repeats, keep first-seen order, so the grid has no duplicate configs.
    unique = list(dict.fromkeys(given))
    if not unique:
        raise ValueError("a sweep knob was given an empty list of values")
    return unique


def _check_chunking(chunk_size: int, overlap: int) -> None:
    # Same rule as Chunker's checks, applied up front so a bad pair rejects the
    # whole sweep before any forward pass runs.
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            f"invalid sweep pair (chunk_size={chunk_size}, overlap={overlap}): "
            "chunk_size must be positive, overlap must not be negative, "
            "and overlap must be smaller than chunk_size"
        )


def build_grid(
    *,
    chunk_sizes: Optional[Sequence[int]] = None,
    overlaps: Optional[Sequence[int]] = None,
    top_ks: Optional[Sequence[int]] = None,
    embedding_models: Optional[Sequence[str]] = None,
    prepend_section_context: Optional[Sequence[bool]] = None,
) -> list[RunConfig]:
    """Cartesian product of the value lists, ``top_k`` varying fastest.

    Raises ``ValueError`` naming the first invalid ``(chunk_size, overlap)`` pair.
    """
    sizes = _values(chunk_sizes, _DEFAULTS.chunk_size)
    overlap_values = _values(overlaps, _DEFAULTS.overlap)
    for chunk_size, overlap in itertools.product(sizes, overlap_values):
        _check_chunking(chunk_size, overlap)
    product = itertools.product(
        sizes,
        overlap_values,
        _values(prepend_section_context, _DEFAULTS.prepend_section_context),
        _values(embedding_models, _DEFAULTS.embedding_model),
        _values(top_ks, _DEFAULTS.top_k),
    )
    return [
        RunConfig(
            chunk_size=chunk_size,
            overlap=overlap,
            prepend_section_context=prepend,
            embedding_model=model,
            top_k=top_k,
        )
        for chunk_size, overlap, prepend, model, top_k in product
    ]


def _base(config: RunConfig) -> RunConfig:
    # Every knob except top_k, normalised so configs sharing a base compare equal.
    return replace(config, top_k=_DEFAULTS.top_k)


def count_forward_passes(grid: Sequence[RunConfig]) -> int:
    """Number of distinct bases (configs ignoring ``top_k``) in the grid."""
    return len({_base(config) for config in grid})


def run_sweep(
    pdf: Union[str, Path],
    pairs: Sequence[QuestionPair],
    *,
    chunk_sizes: Optional[Sequence[int]] = None,
    overlaps: Optional[Sequence[int]] = None,
    top_ks: Optional[Sequence[int]] = None,
    embedding_models: Optional[Sequence[str]] = None,
    prepend_section_context: Optional[Sequence[bool]] = None,
    progress: Optional[Callable[[int, int, RunConfig], None]] = None,
) -> SweepResult:
    """Run every grid config; ``progress(done, total, config)`` fires after each."""
    grid = build_grid(
        chunk_sizes=chunk_sizes,
        overlaps=overlaps,
        top_ks=top_ks,
        embedding_models=embedding_models,
        prepend_section_context=prepend_section_context,
    )
    groups: dict[RunConfig, list[RunConfig]] = {}
    for config in grid:
        groups.setdefault(_base(config), []).append(config)

    scored: dict[RunConfig, BatchResult] = {}
    done = 0
    for configs in groups.values():
        prepared = prepare_batch(pdf, configs[0], pairs)
        for config in configs:
            scored[config] = score_batch(prepared, pairs, config.top_k)
            done += 1
            if progress is not None:
                progress(done, len(grid), config)

    return SweepResult(
        entries=tuple(SweepEntry(config, scored[config]) for config in grid),
        config_count=len(grid),
        forward_passes=len(groups),
    )
