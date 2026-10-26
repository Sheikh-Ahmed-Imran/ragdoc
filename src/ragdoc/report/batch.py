"""Terminal and JSON reporter for a batch run: distribution + per-question list.

Renders the :class:`~ragdoc.batch.BatchResult` T-002 returned and nothing else.
Terminal text and JSON only (spec Exclusions, ADR-004).

Rules this module keeps:

* **Honest single-relevant aggregates only.** Over the questions that reached
  retrieval (those with a ``position``) it computes MRR, hit-rate@``top_k`` (the
  run's configured cutoff) and the rank median / p90, and states the M-of-N
  denominator. Questions lost upstream are not counted as RR = 0. No precision
  average, recall curve or other metric is computed, and nothing spans runs.
* **Ceilings.** MRR / RR / hit-rate are not precision and max at 1.0, so they
  carry no ceiling. The only precision figure that can appear is a per-question
  ``precision@k`` built from that question's own rank evidence, and it goes
  through :func:`ragdoc.report.format_metric`, which always prints the ceiling.
* **Separate from the engine.** It does not run anything, writes nothing to
  disk, and imports no CLI layer; the result type is imported for typing only.
"""

from __future__ import annotations

import statistics
from typing import TYPE_CHECKING, Any, Mapping, Optional

from ragdoc.report import format_metric, precision_at_k, reciprocal_rank
from ragdoc.report._table import LEFT, RIGHT, render_table
from ragdoc.report.symbols import UNICODE, Symbols

if TYPE_CHECKING:
    from ragdoc.batch import BatchResult, QuestionResult

__all__ = ["batch_to_json", "render_batch_report"]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _distribution(batch: "BatchResult") -> dict[str, int]:
    # Tallied from the results so the counts always sum to the question total.
    tallied: dict[str, int] = {name: 0 for name in batch.counts}
    for item in batch.results:
        tallied[item.bucket] = tallied.get(item.bucket, 0) + 1
    return tallied


def retrieval_metrics(batch: "BatchResult") -> dict[str, Any]:
    """MRR, hit-rate@top_k and rank median/p90 over questions that reached retrieval.

    p90 uses ``statistics.quantiles(n=10, method="inclusive")``; with a single
    rank it is that rank. Figures are ``None`` when ``reached`` is 0.
    """
    positions: list[int] = []
    top_k: Optional[int] = None
    for item in batch.results:
        evidence = item.evidence or {}
        position = evidence.get("position")
        if not isinstance(position, int) or position < 1:
            continue
        positions.append(position)
        k = evidence.get("top_k")
        if top_k is None and isinstance(k, int) and k > 0:
            top_k = k
    metrics: dict[str, Any] = {
        "reached": len(positions),
        "total": len(batch.results),
        "top_k": top_k,
        "mrr": None,
        "hit_rate": None,
        "rank_median": None,
        "rank_p90": None,
    }
    if not positions:
        return metrics
    metrics["mrr"] = sum(reciprocal_rank(p) for p in positions) / len(positions)
    if top_k is not None:
        metrics["hit_rate"] = sum(1 for p in positions if p <= top_k) / len(positions)
    metrics["rank_median"] = statistics.median(positions)
    metrics["rank_p90"] = (
        statistics.quantiles(positions, n=10, method="inclusive")[8]
        if len(positions) > 1
        else float(positions[0])
    )
    return metrics


def _metrics_lines(metrics: Mapping[str, Any]) -> list[str]:
    m, n = metrics["reached"], metrics["total"]
    if not m:
        return [f"Retrieval metrics: no questions reached retrieval (0 of {n})."]
    parts = [f"MRR = {metrics['mrr']:.2f}"]
    if metrics["hit_rate"] is not None:
        parts.append(f"hit-rate@{metrics['top_k']} = {metrics['hit_rate']:.2f}")
    parts.append(f"rank median = {metrics['rank_median']:g}")
    parts.append(f"p90 = {metrics['rank_p90']:g}")
    return [
        f"Retrieval metrics (over {m} of {n} questions that reached retrieval): "
        + ", ".join(parts)
    ]


def _evidence_detail(item: "QuestionResult") -> str:
    evidence = item.evidence or {}
    parts = []
    if "position" in evidence:
        total = evidence.get("total")
        parts.append(
            f"rank {evidence['position']}" + (f" of {total}" if total else "")
        )
    if "answer_page" in evidence:
        parts.append(f"page {evidence['answer_page']}")
    k = evidence.get("top_k")
    if isinstance(evidence.get("position"), int) and isinstance(k, int) and k > 0:
        value, ceiling = precision_at_k(evidence["position"], k)
        parts.append(format_metric(f"precision@{k}", value, ceiling))
    return "; ".join(parts)


def render_batch_report(batch: "BatchResult", symbols: Symbols = UNICODE) -> str:
    """Render the distribution (headline) then one line per question."""
    total = len(batch.results)
    lines = [f"Failure distribution ({total} question{'s' if total != 1 else ''}):"]
    dist_rows = []
    for bucket, count in _distribution(batch).items():
        pct = (100.0 * count / total) if total else 0.0
        dist_rows.append((bucket, str(count), f"{pct:.1f}%"))
    table = render_table(("stage", "count", "share"), dist_rows, (LEFT, RIGHT, RIGHT))
    lines += ["  " + text for text in table]

    lines += [""] + _metrics_lines(retrieval_metrics(batch))
    lines += ["", "Questions:"]
    if not total:
        lines.append("  (no questions in the set)")
    for index, item in enumerate(batch.results, start=1):
        stage = item.stage if item.stage else symbols.not_reached
        row = f"  {index}. [{item.bucket}] {item.question}"
        if item.stage and item.stage != item.bucket:
            row += f"  (stage: {stage})"
        detail = _evidence_detail(item)
        if detail:
            row += f"\n       {detail}"
        if item.reason:
            row += f"\n       {item.reason}"
        lines.append(row)
    return "\n".join(lines)


def batch_to_json(batch: "BatchResult") -> dict[str, Any]:
    """Return distribution, retrieval_metrics and questions (JSON-able)."""
    return {
        "distribution": _distribution(batch),
        "retrieval_metrics": retrieval_metrics(batch),
        "questions": [
            {
                "question": item.question,
                "bucket": item.bucket,
                "stage": item.stage,
                "reason": item.reason,
                "evidence": _jsonable(item.evidence or {}),
            }
            for item in batch.results
        ],
    }
