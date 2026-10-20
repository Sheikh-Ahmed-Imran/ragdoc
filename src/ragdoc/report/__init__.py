"""Terminal reporter: render a composed :class:`Verdict` as plain text.

Terminal text only (spec Exclusions, ADR-004). The reporter renders what
``compose_verdict`` produced; it never computes a verdict, applies a fix, or
imports the engine's runner, so the engine stays scriptable without its output
format being what callers parse (``.ai/architecture/components.md``).

Every metric is printed with its theoretical maximum (``.ai/constitution.md``):
with one labelled chunk per question ``precision@k`` cannot exceed ``1/k``, so a
precision figure only ever appears through :func:`format_metric`, which has no
way to omit the ceiling.
"""

from __future__ import annotations

from typing import Optional

from ragdoc.artifacts import CheckStatus
from ragdoc.attribution.verdict import StageReport, Verdict
from ragdoc.config import RunConfig

__all__ = ["format_metric", "precision_at_k", "render_report"]

_NOT_REACHED = "— not reached"
_PASS = "✓"
_FAIL = "✗"
_FLAGGED = "⚑"
_UNEVALUATED = "? could not be evaluated"


def format_metric(name: str, value: float, ceiling: float) -> str:
    """``name = value (max ceiling)`` — the only way a metric is printed."""
    return f"{name} = {value:.2f} (max {ceiling:.2f})"


def precision_at_k(position: int, top_k: int) -> tuple[float, float]:
    """``(precision, ceiling)`` for one labelled relevant chunk.

    The single target is either inside the top ``top_k`` (precision ``1/k``,
    which equals the ceiling) or not (``0``). The ceiling is always ``1/k``.
    """
    ceiling = 1.0 / top_k
    return (ceiling if position <= top_k else 0.0), ceiling


def _stage_detail(report: StageReport, top_k: Optional[int]) -> str:
    result = report.result
    evidence = result.evidence if result is not None else {}
    if report.stage == "retrieval" and "position" in evidence:
        k = evidence.get("top_k", top_k)
        detail = f"target chunk ranked {evidence['position']} of {evidence['total']}"
        if k:
            value, ceiling = precision_at_k(evidence["position"], k)
            detail += f"; {format_metric(f'precision@{k}', value, ceiling)}"
        return detail
    if report.stage == "chunking" and evidence.get("context_missing"):
        return "section heading not embedded in chunk text"
    if report.status is CheckStatus.FAIL and result is not None and result.reason:
        return result.reason
    return ""


def _marker(report: StageReport) -> str:
    if report.status is CheckStatus.PASS:
        flagged = (
            report.result is not None
            and report.result.evidence.get("context_missing")
        )
        return _FLAGGED if flagged else _PASS
    if report.status is CheckStatus.FAIL:
        return _FAIL
    if report.status is CheckStatus.UNEVALUATED:
        return _UNEVALUATED
    return _NOT_REACHED


def render_report(verdict: Verdict, config: RunConfig) -> str:
    """Render the four-row table, VERDICT, FIX and the run configuration."""
    top_k = config.top_k
    lines = []
    for report in verdict.stages:
        if report.stage == "generation":
            marker, detail = _NOT_REACHED, ""  # always, in v0
        else:
            marker, detail = _marker(report), _stage_detail(report, top_k)
        row = f"  {report.stage:<11} {marker}"
        lines.append(f"{row}  {detail}" if detail else row)

    lines.append("")
    if verdict.withheld:
        lines.append("VERDICT: withheld — a stage could not be evaluated")
    elif verdict.stage is None:
        lines.append("VERDICT: no failing stage found")
    else:
        lines.append(f"VERDICT: {verdict.stage} — {verdict.reason}")
        lines.append(f"FIX: {verdict.fix}")

    lines += [
        "",
        "Run configuration:",
        f"  loader: {config.loader}",
        f"  chunk size: {config.chunk_size}",
        f"  overlap: {config.overlap}",
        f"  embedding model: {config.embedding_model}",
        f"  top_k: {config.top_k}",
    ]
    return "\n".join(lines)
