"""Sweep engine: grid construction and forward-pass reuse (no model needed)."""

from dataclasses import replace

import pytest

from ragdoc import sweep as sweep_mod
from ragdoc.batch import BatchResult, PreparedBatch, QuestionPair
from ragdoc.config import RunConfig

PAIRS = [QuestionPair("Q?", "A")]


def _fake(monkeypatch):
    calls = {"prepare": [], "score": []}

    def prepare(pdf, config, pairs):
        calls["prepare"].append(config)
        return PreparedBatch(doc=None, chunks=None, rankings=(None,), config=config)

    def score(prepared, pairs, top_k):
        calls["score"].append((prepared.config, top_k))
        return BatchResult(results=(), counts={"no-failure": top_k})

    monkeypatch.setattr(sweep_mod, "prepare_batch", prepare)
    monkeypatch.setattr(sweep_mod, "score_batch", score)
    return calls


def test_cartesian_product():
    grid = sweep_mod.build_grid(chunk_sizes=[500, 800], overlaps=[0, 50])
    assert len(grid) == 4
    assert {(c.chunk_size, c.overlap) for c in grid} == {(500, 0), (500, 50), (800, 0), (800, 50)}


def test_unlisted_knob_uses_default():
    (config,) = sweep_mod.build_grid()
    assert config == RunConfig()


def test_top_k_only_grid_is_one_forward_pass(monkeypatch):
    calls = _fake(monkeypatch)
    result = sweep_mod.run_sweep("d.pdf", PAIRS, top_ks=[1, 3, 5])
    assert (result.config_count, result.forward_passes) == (3, 1)
    assert len(calls["prepare"]) == 1
    assert [k for _, k in calls["score"]] == [1, 3, 5]


def test_forward_pass_per_base(monkeypatch):
    calls = _fake(monkeypatch)
    result = sweep_mod.run_sweep("d.pdf", PAIRS, chunk_sizes=[500, 800], top_ks=[3, 5])
    assert (result.config_count, result.forward_passes) == (4, 2)
    assert len(calls["prepare"]) == 2


def test_results_carry_config_and_batch_in_stable_order(monkeypatch):
    _fake(monkeypatch)
    result = sweep_mod.run_sweep("d.pdf", PAIRS, chunk_sizes=[800, 500], top_ks=[2, 4])
    assert [(e.config.chunk_size, e.config.top_k) for e in result.entries] == [
        (800, 2), (800, 4), (500, 2), (500, 4)]
    assert all(e.batch.counts["no-failure"] == e.config.top_k for e in result.entries)


def test_progress_called_per_config(monkeypatch):
    _fake(monkeypatch)
    seen = []
    sweep_mod.run_sweep("d.pdf", PAIRS, top_ks=[1, 2],
                        progress=lambda done, total, config: seen.append((done, total)))
    assert seen == [(1, 2), (2, 2)]


def test_duplicate_values_collapse():
    assert len(sweep_mod.build_grid(chunk_sizes=[500, 500])) == 1
    assert sweep_mod.count_forward_passes(
        [RunConfig(top_k=1), replace(RunConfig(), top_k=2)]) == 1


def test_invalid_pair_rejected_before_any_forward_pass(monkeypatch):
    calls = _fake(monkeypatch)
    with pytest.raises(ValueError, match=r"chunk_size=100, overlap=200"):
        sweep_mod.run_sweep("d.pdf", PAIRS, chunk_sizes=[500, 100], overlaps=[200])
    assert calls == {"prepare": [], "score": []}


@pytest.mark.parametrize(
    "chunk_sizes, overlaps, pair",
    [
        ([100], [200], "chunk_size=100, overlap=200"),
        ([100], [100], "chunk_size=100, overlap=100"),
        ([500], [0, -5], "chunk_size=500, overlap=-5"),
        ([500, 0], [0], "chunk_size=0, overlap=0"),
        ([-10], [0], "chunk_size=-10, overlap=0"),
    ],
)
def test_build_grid_rejects_invalid_pair(chunk_sizes, overlaps, pair):
    with pytest.raises(ValueError, match=pair):
        sweep_mod.build_grid(chunk_sizes=chunk_sizes, overlaps=overlaps)


def test_valid_grid_unaffected():
    assert len(sweep_mod.build_grid(chunk_sizes=[500, 800], overlaps=[0, 50])) == 4
