"""Terminal and JSON reporter for a batch run: distribution + per-question list.

Renders the :class:`~ragdoc.batch.BatchResult` T-002 returned and nothing else.
Terminal text and JSON only (spec Exclusions, ADR-004).

Rules this module keeps:

* **No aggregate metric.** A bucket count is a count, not a retrieval metric, so
  no MRR / recall / precision average is computed or printed.
* **Ceilings.** The only precision figure that can appear is a per-question
  ``precision@k`` built from that question's own rank evidence, and it goes
  through :func:`ragdoc.report.format_metric`, which always prints the ceiling.
* **Separate from the engine.** It does not run anything, writes nothing to
  disk, and imports no CLI layer; the result type is imported for typing only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

from ragdoc.report import format_metric, precision_at_k

if TYPE_CHECKING:
    from ragdoc.batch import BatchResult, QuestionResult

__all__ = ["batch_to_json", "render_batch_report"]

_NONE = "—"


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


def _evidence_detail(item: "QuestionResult") -> str:
    evidence = item.evidence or {}
    parts = []
    if "position" in evidence:
        total = evidence.get("total")
        parts.append(
            f"rank {evidence['position']}" + (f" of {total}" if total else "")
        )
    if "page" in evidence:
        parts.append(f"page {evidence['page']}")
    k = evidence.get("top_k")
    if isinstance(evidence.get("position"), int) and isinstance(k, int) and k > 0:
        value, ceiling = precision_at_k(evidence["position"], k)
        parts.append(format_metric(f"precision@{k}", value, ceiling))
    return "; ".join(parts)


def render_batch_report(batch: "BatchResult") -> str:
    """Render the distribution (headline) then one line per question."""
    total = len(batch.results)
    lines = [f"Failure distribution ({total} question{'s' if total != 1 else ''}):"]
    for bucket, count in _distribution(batch).items():
        pct = (100.0 * count / total) if total else 0.0
        lines.append(f"  {bucket:<11} {count:>4}  {pct:5.1f}%")

    lines += ["", "Questions:"]
    for index, item in enumerate(batch.results, start=1):
        stage = item.stage if item.stage else _NONE
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
    """Return ``{"distribution": {...}, "questions": [...]}`` (JSON-able)."""
    return {
        "distribution": _distribution(batch),
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
