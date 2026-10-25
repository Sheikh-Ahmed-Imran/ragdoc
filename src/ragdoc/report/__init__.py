"""Terminal reporter: render a composed :class:`Verdict` as plain text.

Terminal text only (spec Exclusions, ADR-004). The reporter renders what
``compose_verdict`` produced; it never computes a verdict, applies a fix, or
imports the engine's runner, so the engine stays scriptable without its output
format being what callers parse (``.ai/architecture/components.md``).

Every precision metric is printed with its theoretical maximum (``.ai/constitution.md``):
with one labelled chunk per question ``precision@k`` cannot exceed ``1/k``, so a
precision figure only ever appears through :func:`format_metric`, which has no
way to omit the ceiling. Reciprocal rank is not precision and maxes at 1.0, so
it is printed plainly.
"""

from __future__ import annotations

from typing import Optional, Sequence

from ragdoc.artifacts import CheckStatus
from ragdoc.attribution.verdict import StageReport, Verdict
from ragdoc.config import RunConfig
from ragdoc.report.symbols import UNICODE, Symbols

__all__ = ["format_metric", "precision_at_k", "reciprocal_rank", "render_report"]

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


def reciprocal_rank(position: int) -> float:
    """``1 / position`` for the single relevant chunk (1-based absolute rank).

    Not a precision figure and it tops out at 1.0, so it is printed without a
    ``(max …)`` ceiling.
    """
    return 1.0 / position


def _stage_detail(report: StageReport, top_k: Optional[int]) -> str:
    result = report.result
    evidence = result.evidence if result is not None else {}
    if report.stage == "retrieval" and "position" in evidence:
        k = evidence.get("top_k", top_k)
        detail = f"target chunk ranked {evidence['position']} of {evidence['total']}"
        if k:
            value, ceiling = precision_at_k(evidence["position"], k)
            detail += f"; {format_metric(f'precision@{k}', value, ceiling)}"
        detail += f"; RR = {reciprocal_rank(evidence['position']):.2f}"
        return detail
    if report.stage == "chunking" and evidence.get("context_missing"):
        return "section heading not embedded in chunk text"
    if report.status is CheckStatus.FAIL and result is not None and result.reason:
        return result.reason
    return ""


def _marker(report: StageReport, symbols: Symbols) -> str:
    if report.status is CheckStatus.PASS:
        flagged = (
            report.result is not None
            and report.result.evidence.get("context_missing")
        )
        return symbols.flagged if flagged else symbols.passed
    if report.status is CheckStatus.FAIL:
        return symbols.fail
    if report.status is CheckStatus.UNEVALUATED:
        return _UNEVALUATED
    return f"{symbols.not_reached} not reached"


def _span_text(span: object) -> str:
    start, end = getattr(span, "start", None), getattr(span, "end", None)
    if start is None or end is None:
        return str(span)
    return f"[{start}, {end})"


def _evidence_lines(report: StageReport) -> list[str]:
    """Curated, labelled evidence beneath a stage row (verbose only).

    A stage that was not reached or has no result adds nothing. Only a fixed
    subset of keys is shown — never the raw evidence dict.
    """
    result = report.result
    if result is None or report.status in (
        CheckStatus.NOT_REACHED,
        CheckStatus.UNEVALUATED,
    ):
        return []
    ev = result.evidence
    rows: list[tuple[str, str]] = []
    if report.stage == "extraction":
        if "span" in ev:
            rows.append(("located span", _span_text(ev["span"])))
        if "page" in ev:
            rows.append(("page", str(ev["page"])))
        if "snippet" in ev:
            rows.append(("matched snippet", repr(ev["snippet"])))
        if "match" in ev:
            rows.append(("match kind", str(ev["match"])))
    elif report.stage == "chunking":
        if "context_missing" in ev:
            rows.append(("context missing", "yes" if ev["context_missing"] else "no"))
        if "target_chunk_id" in ev:
            rows.append(("target chunk", str(ev["target_chunk_id"])))
        if "section_path" in ev:
            path = ev["section_path"]
            rows.append(("section path", " > ".join(path) if path else "(none)"))
        elif "crossing_chunk_ids" in ev:
            rows.append(("crossing chunks", ", ".join(map(str, ev["crossing_chunk_ids"]))))
    elif report.stage == "retrieval":
        for key, label in (("position", "position"), ("total", "total"), ("top_k", "top_k")):
            if key in ev:
                rows.append((label, str(ev[key])))
    return [f"      {label}: {value}" for label, value in rows]


def render_report(
    verdict: Verdict,
    config: RunConfig,
    verbose: bool = False,
    symbols: Symbols = UNICODE,
    empty_pages: Sequence[int] = (),
    total_pages: Optional[int] = None,
) -> str:
    """Render the four-row table, VERDICT, FIX and the run configuration.

    With ``verbose=True`` each evaluated stage row is followed by a curated
    subset of its check evidence; the default output is unchanged.

    A non-empty ``empty_pages`` adds one advisory line (whatever the verdict,
    verbose or not) naming pages that yielded no extractable text. It never
    alters a verdict. ``total_pages`` is the document's page count.
    """
    top_k = config.top_k
    lines = []
    for report in verdict.stages:
        if report.stage == "generation":
            marker, detail = f"{symbols.not_reached} not reached", ""  # always, in v0
        else:
            marker, detail = _marker(report, symbols), _stage_detail(report, top_k)
        row = f"  {report.stage:<11} {marker}"
        lines.append(f"{row}  {detail}" if detail else row)
        if verbose and report.stage != "generation":
            lines += _evidence_lines(report)

    lines.append("")
    if verdict.withheld:
        lines.append(f"VERDICT: withheld {symbols.not_reached} a stage could not be evaluated")
    elif verdict.stage is None:
        lines.append("VERDICT: no failing stage found")
    else:
        lines.append(f"VERDICT: {verdict.stage} {symbols.not_reached} {verdict.reason}")
        lines.append(f"FIX: {verdict.fix}")

    if empty_pages:
        pages = ", ".join(f"p.{n}" for n in empty_pages)
        of_total = f" of {total_pages}" if total_pages else ""
        lines.append("")
        lines.append(f"{len(empty_pages)}{of_total} pages had no extractable text: {pages}")

    lines += [
        "",
        "Run configuration:",
        f"  loader: {config.loader}",
        f"  chunk size: {config.chunk_size}",
        f"  overlap: {config.overlap}",
        f"  embedding model: {config.embedding_model}",
        f"  top_k: {config.top_k}",
        f"  prepend_section_context: {config.prepend_section_context}",
    ]
    return "\n".join(lines)
