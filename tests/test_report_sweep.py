"""Sweep reporter: ranking, table and JSON (pure; hand-built results)."""

import json

import pytest

from ragdoc.batch import BatchResult, QuestionResult
from ragdoc.config import RunConfig
from ragdoc.report.sweep import rank_entries, render_sweep_table, sweep_to_json
from ragdoc.sweep import SweepEntry, SweepResult


def _batch(no_failure, positions, top_k=5):
    """``no_failure`` survivors plus one retrieval-bucket result per position."""
    results = [
        QuestionResult("q", "a", "retrieval", "retrieval", None,
                       {"position": p, "total": 20, "top_k": top_k}, None)
        for p in positions
    ]
    results += [QuestionResult("q", "a", "no-failure", None, None, {}, None)] * no_failure
    counts = {"extraction": 0, "chunking": 0, "retrieval": len(positions),
              "withheld": 0, "no-failure": no_failure}
    return BatchResult(results=tuple(results), counts=counts)


def _sweep(*specs):
    entries = tuple(
        SweepEntry(RunConfig(chunk_size=size), _batch(nf, positions))
        for size, nf, positions in specs
    )
    return SweepResult(entries=entries, config_count=len(entries), forward_passes=len(entries))


# chunk 500: 3 survive, MRR 1.0 | 800: 5 survive, MRR 0.5 | 1200: 5 survive, MRR 1/3
RESULT = _sweep((500, 3, [1]), (800, 5, [2]), (1200, 5, [3]))


def test_default_winner_is_most_no_failure_tie_broken_by_mrr():
    assert rank_entries(RESULT)[0] == 1  # 800 beats 1200 on MRR


def test_mrr_objective_changes_winner():
    assert rank_entries(RESULT, "mrr")[0] == 0


def test_hit_rate_objective_changes_winner():
    # top_k is 5: rank 9 misses, rank 2 hits.
    result = _sweep((500, 5, [9]), (800, 1, [2]))
    assert rank_entries(result, "no-failure")[0] == 0
    assert rank_entries(result, "hit-rate")[0] == 1


def test_full_tie_keeps_grid_order_single_winner():
    result = _sweep((500, 2, [1]), (800, 2, [1]))
    assert rank_entries(result)[0] == 0


def test_unknown_objective_rejected():
    with pytest.raises(ValueError):
        rank_entries(RESULT, "bogus")


def test_table_shows_varied_knobs_counts_and_marks_winner():
    lines = render_sweep_table(RESULT).splitlines()
    assert "chunk" in lines[0] and "overlap" not in lines[0]
    assert "no-failure" in lines[0] and "MRR" in lines[0] and "hit-rate@k" in lines[0]
    rows = lines[1:4]
    assert [r.startswith("*") for r in rows] == [False, True, False]
    assert "5/6" in rows[1]  # 5 survivors of 6 results (1 retrieval + 5)
    assert "Winner by no-failure" in lines[-1] and "chunk 800" in lines[-1]


def test_json_has_every_config():
    data = sweep_to_json(RESULT, "mrr")
    assert data["objective"] == "mrr" and data["winner"] == 0
    assert len(data["configs"]) == 3
    first = data["configs"][0]
    assert first["config"]["chunk_size"] == 500
    assert first["counts"]["no-failure"] == 3
    assert first["retrieval_metrics"]["mrr"] == 1.0
    json.dumps(data)
