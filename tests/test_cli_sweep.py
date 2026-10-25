"""CLI wiring for ``ragdoc sweep`` (the sweep engine is stubbed)."""

import json

import pytest

from ragdoc import cli
from ragdoc.batch import BatchResult, QuestionResult
from ragdoc.sweep import SweepEntry, SweepResult


def _batch(no_failure, position):
    results = [QuestionResult("q", "a", "retrieval", "retrieval", None,
                              {"position": position, "total": 9, "top_k": 5}, None)]
    results += [QuestionResult("q", "a", "no-failure", None, None, {}, None)] * no_failure
    return BatchResult(results=tuple(results), counts={"retrieval": 1, "no-failure": no_failure})


def _stub(monkeypatch, seen):
    def fake(pdf, pairs, *, progress=None, **knobs):
        seen.update(pdf=pdf, pairs=pairs, knobs=knobs)
        grid = cli.build_grid(**knobs)
        # chunk 500 survives most; chunk 800 has the better MRR.
        entries = tuple(
            SweepEntry(c, _batch(3, 4) if c.chunk_size == 500 else _batch(1, 1)) for c in grid
        )
        for i, c in enumerate(grid, 1):
            progress(i, len(grid), c)
        return SweepResult(entries, len(grid), cli.count_forward_passes(grid))
    monkeypatch.setattr(cli, "execute_sweep", fake)


def _questions(tmp_path):
    path = tmp_path / "q.json"
    path.write_text(json.dumps([{"question": "Q?", "expected_answer": "A"}]), encoding="utf-8")
    return str(path)


def test_parses_lists_and_prints_grid_size(monkeypatch, tmp_path, capfd):
    seen = {}
    _stub(monkeypatch, seen)
    code = cli.main(["sweep", "d.pdf", _questions(tmp_path), "--chunk-size", "500,800",
                     "--overlap", "0,50", "--prepend-section-context", "true,false"])
    captured = capfd.readouterr()
    assert code == 0
    assert seen["knobs"]["chunk_sizes"] == [500, 800]
    assert seen["knobs"]["overlaps"] == [0, 50]
    assert seen["knobs"]["prepend_section_context"] == [True, False]
    assert seen["knobs"]["top_ks"] is None  # unset knob -> engine default
    assert "Sweep: 8 configs, 8 forward passes" in captured.out
    assert "[8/8]" in captured.err


def test_top_k_only_prints_one_forward_pass(monkeypatch, tmp_path, capfd):
    _stub(monkeypatch, {})
    cli.main(["sweep", "d.pdf", _questions(tmp_path), "--top-k", "1,3,5"])
    assert "Sweep: 3 configs, 1 forward pass\n" in capfd.readouterr().out


def test_objective_changes_named_winner(monkeypatch, tmp_path, capfd):
    _stub(monkeypatch, {})
    qs = _questions(tmp_path)
    cli.main(["sweep", "d.pdf", qs, "--chunk-size", "500,800"])
    assert "Winner by no-failure: chunk 500" in capfd.readouterr().out
    cli.main(["sweep", "d.pdf", qs, "--chunk-size", "500,800", "--objective", "mrr"])
    assert "Winner by mrr: chunk 800" in capfd.readouterr().out


def test_json_file(monkeypatch, tmp_path, capfd):
    _stub(monkeypatch, {})
    out = tmp_path / "out.json"
    code = cli.main(["sweep", "d.pdf", _questions(tmp_path), "--chunk-size", "500,800",
                     "--json", str(out)])
    capfd.readouterr()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert code == 0 and len(data["configs"]) == 2 and data["objective"] == "no-failure"


def test_unknown_objective_exits_nonzero(tmp_path, capfd):
    with pytest.raises(SystemExit) as exc:
        cli.main(["sweep", "d.pdf", _questions(tmp_path), "--objective", "bogus"])
    assert exc.value.code != 0


def test_bad_list_value_exits_nonzero(tmp_path, capfd):
    with pytest.raises(SystemExit) as exc:
        cli.main(["sweep", "d.pdf", _questions(tmp_path), "--chunk-size", "a,b"])
    assert exc.value.code != 0


def test_other_subcommands_still_parse():
    parser = cli.build_parser()
    assert parser.parse_args(["batch", "d.pdf", "q.json"]).top_k == 10
    assert parser.parse_args(["diagnose", "d.pdf", "Q", "A"]).chunk_size == 500
    assert parser.parse_args(["compare", "d.pdf", "A"]).loaders is None


@pytest.mark.parametrize("sizes", ["100", "500,100"])
def test_invalid_pair_exits_2_without_progress(monkeypatch, tmp_path, capfd, sizes):
    seen = {}
    _stub(monkeypatch, seen)
    code = cli.main(["sweep", "d.pdf", _questions(tmp_path), "--chunk-size", sizes,
                     "--overlap", "200"])
    captured = capfd.readouterr()
    assert code == 2
    assert "chunk_size=100, overlap=200" in captured.err
    assert "Sweep:" not in captured.out and "[1/" not in captured.err
    assert "Traceback" not in captured.err
    assert seen == {}
