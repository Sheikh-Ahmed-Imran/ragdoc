"""The attribution engine: decide which pipeline stage lost the answer.

This package is the diagnostic half of ``ragdoc``. It reads the artefacts the
forward pipeline produced (an :class:`~ragdoc.artifacts.ExtractedDoc`, a list of
:class:`~ragdoc.artifacts.Chunk`, and later a :class:`~ragdoc.artifacts.Ranking`)
and runs the backward checks in order, stopping at the first failure and
withholding the verdict when a check cannot be evaluated.

It depends only on the artefact shapes and on nothing in the interface layer
(``.ai/constitution.md`` — the headless engine never depends on the UI). Import
the engine and the offline checks from here:

    from ragdoc.attribution import AttributionEngine, locate, survive

T-004 ships ``locate`` and ``survive`` and the runner; ``rank`` and the final
verdict composition are T-006, which extends the runner's check sequence.
"""

from __future__ import annotations

from ragdoc.attribution.checks import locate, survive
from ragdoc.attribution.engine import (
    DEFAULT_CHECKS,
    AttributionEngine,
    Check,
    CheckOutcome,
    EngineResult,
    RunState,
)

__all__ = [
    "locate",
    "survive",
    "AttributionEngine",
    "EngineResult",
    "CheckOutcome",
    "Check",
    "RunState",
    "DEFAULT_CHECKS",
]
