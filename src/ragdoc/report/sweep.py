"""Terminal and JSON reporter for a config sweep: table + winner.

Renders the :class:`~ragdoc.sweep.SweepResult` the sweep engine returned and
nothing else; it runs nothing and loads no PDF.

Rules this module keeps:

* **Default objective is the no-failure count**, measured over all N questions and
  so comparable across configs. Hit-rate and MRR are over the per-config subset
  that reached retrieval (ADR-009); they are offered as objectives, and the other
  of the pair is the graded tie-break (MRR after no-failure).
* **Exactly one winner.** Objective, then MRR, then smaller ``top_k``; remaining
  ties fall back to grid order (stable sort). No other knob is preferred.
* **No new metrics.** Bucket counts and :func:`retrieval_metrics` only.
* **Pure view.** No CLI import; the result type is imported for typing only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Sequence

from ragdoc.report.batch import retrieval_metrics

if TYPE_CHECKING:
    from ragdoc.sweep import SweepEntry, SweepResult

__all__ = ["OBJECTIVES", "rank_entries", "render_sweep_table", "sweep_to_json"]

OBJECTIVES = ("no-failure", "hit-rate", "mrr")

# (RunConfig attribute, column header) in display order.
_KNOBS = (
    ("chunk_size", "chunk"),
    ("overlap", "overlap"),
    ("top_k", "top-k"),
    ("embedding_model", "model"),
    ("prepend_section_context", "prepend"),
)

_CONFIG_FIELDS = (
    "loader", "chunk_size", "overlap", "prepend_section_context", "embedding_model", "top_k",
)


def _score(value: Optional[float]) -> float:
    # A config with nothing in retrieval has no metric; it ranks below any real one.
    return -1.0 if value is None else value


def _key(entry: "SweepEntry", objective: str) -> tuple:
    metrics = retrieval_metrics(entry.batch)
    mrr, hit = _score(metrics["mrr"]), _score(metrics["hit_rate"])
    # Last key: smaller top_k wins a tie (negated, as the sort is descending).
    smaller_top_k = -entry.config.top_k
    if objective == "no-failure":
        return (entry.batch.counts.get("no-failure", 0), mrr, smaller_top_k)
    if objective == "hit-rate":
        return (hit, mrr, smaller_top_k)
    return (mrr, hit, smaller_top_k)


def rank_entries(result: "SweepResult", objective: str = "no-failure") -> list[int]:
    """Entry indices, best first: objective, MRR, smaller top_k, then grid order."""
    if objective not in OBJECTIVES:
        raise ValueError(
            f"unknown objective {objective!r}; choose one of: {', '.join(OBJECTIVES)}"
        )
    # reverse=True keeps equal keys in their original (grid) order.
    return sorted(
        range(len(result.entries)),
        key=lambda i: _key(result.entries[i], objective),
        reverse=True,
    )


def _varied(entries: Sequence["SweepEntry"]) -> list[tuple[str, str]]:
    varied = [
        knob for knob in _KNOBS
        if len({getattr(e.config, knob[0]) for e in entries}) > 1
    ]
    # With nothing varied (a single config) show every knob so the row is not blank.
    return varied or list(_KNOBS)


def _cell(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _fmt(value: Optional[float]) -> str:
    return "—" if value is None else f"{value:.2f}"


def render_sweep_table(result: "SweepResult", objective: str = "no-failure") -> str:
    """One row per config in grid order, the winner marked with ``*``."""
    ranked = rank_entries(result, objective)
    if not ranked:
        return "Sweep: no configs."
    winner = ranked[0]
    knobs = _varied(result.entries)
    headers = [header for _, header in knobs] + ["no-failure", "MRR", "hit-rate@k"]
    rows = []
    for entry in result.entries:
        metrics = retrieval_metrics(entry.batch)
        rows.append(
            [_cell(getattr(entry.config, attr)) for attr, _ in knobs]
            + [
                f"{entry.batch.counts.get('no-failure', 0)}/{entry.batch.total}",
                _fmt(metrics["mrr"]),
                _fmt(metrics["hit_rate"]),
            ]
        )
    widths = [max(len(headers[c]), *(len(r[c]) for r in rows)) for c in range(len(headers))]

    def line(mark: str, cells: Sequence[str]) -> str:
        return f"{mark} " + "  ".join(c.ljust(w) for c, w in zip(cells, widths)).rstrip()

    lines = [line(" ", headers)]
    lines += [line("*" if i == winner else " ", row) for i, row in enumerate(rows)]
    best = result.entries[winner].config
    described = ", ".join(f"{header} {_cell(getattr(best, attr))}" for attr, header in knobs)
    survived, mrr = rows[winner][len(knobs)], rows[winner][len(knobs) + 1]
    lines += ["", f"* Winner by {objective}: {described} ({survived} no-failure, MRR {mrr})"]
    return "\n".join(lines)


def sweep_to_json(result: "SweepResult", objective: str = "no-failure") -> dict[str, Any]:
    """Every config's knobs, bucket counts and retrieval metrics, plus the winner."""
    ranked = rank_entries(result, objective)
    winner = ranked[0] if ranked else None
    return {
        "objective": objective,
        "config_count": result.config_count,
        "forward_passes": result.forward_passes,
        "winner": winner,
        "configs": [
            {
                "config": {name: getattr(entry.config, name) for name in _CONFIG_FIELDS},
                "counts": dict(entry.batch.counts),
                "total": entry.batch.total,
                "retrieval_metrics": retrieval_metrics(entry.batch),
                "winner": index == winner,
            }
            for index, entry in enumerate(result.entries)
        ],
    }
