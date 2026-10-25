"""ASCII/Unicode symbol threading through the reporters and the CLI (T-002)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ragdoc import cli
from ragdoc.artifacts import CharSpan, CheckResult, CheckStatus, Chunk, RankedChunk, Ranking
from ragdoc.attribution.verdict import StageReport, Verdict
from ragdoc.batch import BatchResult, QuestionResult
from ragdoc.compare import LoaderResult
from ragdoc.config import RunConfig
from ragdoc.report import render_report
from ragdoc.report.batch import render_batch_report
from ragdoc.report.compare import render_comparison
from ragdoc.report.retrieved import render_retrieved
from ragdoc.report.sweep import render_sweep_table
from ragdoc.report.symbols import ASCII, UNICODE
from ragdoc.sweep import SweepEntry, SweepResult

_FLAG = CheckResult.passed({"context_missing": True})
_VERDICT = Verdict(
    stage="retrieval", reason_code="x", reason="ranked too low", fix="raise top_k",
    stages=(
        StageReport("extraction", CheckStatus.PASS, CheckResult.passed()),
        StageReport("chunking", CheckStatus.PASS, _FLAG),
        StageReport("retrieval", CheckStatus.FAIL, CheckResult.failed("ranked too low")),
        StageReport("generation", CheckStatus.NOT_REACHED, None),
    ),
)
_WITHHELD = Verdict(
    stage=None, reason_code=None, reason=None, fix=None, withheld=True,
    stages=(StageReport("extraction", CheckStatus.NOT_REACHED, None),),
)
_LOADERS = [
    LoaderResult("a", True, True, 2, "exact", 10, 3),
    LoaderResult("b", False, True, None, None, 9, 3, text_head="abc"),
    LoaderResult("c", False, False, None, None, 0, 0),
]
_CHUNKS = [Chunk(id=f"c{i}", text="x" * 50, span=CharSpan(i, i + 1), page=1) for i in range(5)]
_RANKING = Ranking([RankedChunk(c.id, 0.9 - 0.1 * i) for i, c in enumerate(_CHUNKS)])
_BATCH = BatchResult(
    results=(QuestionResult("q", "a", "extraction", "extraction", "gone", {}, None),),
    counts={"extraction": 1, "no-failure": 0},
)
_SWEEP = SweepResult(
    entries=(SweepEntry(RunConfig(chunk_size=500), _BATCH),
             SweepEntry(RunConfig(chunk_size=800), _BATCH)),
    config_count=2, forward_passes=2,
)

_CASES = {
    "report": lambda s: render_report(_VERDICT, RunConfig(), verbose=True, symbols=s),
    "withheld": lambda s: render_report(_WITHHELD, RunConfig(), symbols=s),
    "compare": lambda s: render_comparison(_LOADERS, verbose=True, symbols=s),
    "batch": lambda s: render_batch_report(_BATCH, s),
    "retrieved": lambda s: render_retrieved(_RANKING, _CHUNKS, "c4", 2, limit=10, symbols=s),
    "sweep": lambda s: render_sweep_table(_SWEEP, "mrr", s),
}


@pytest.mark.parametrize("name", _CASES)
def test_ascii_output_has_no_unicode(name):
    out = _CASES[name](ASCII)
    assert out.isascii(), out


@pytest.mark.parametrize("name", _CASES)
def test_default_is_the_unicode_set(name):
    render = _CASES[name]
    assert render(UNICODE) == {
        "report": lambda: render_report(_VERDICT, RunConfig(), verbose=True),
        "withheld": lambda: render_report(_WITHHELD, RunConfig()),
        "compare": lambda: render_comparison(_LOADERS, verbose=True),
        "batch": lambda: render_batch_report(_BATCH),
        "retrieved": lambda: render_retrieved(_RANKING, _CHUNKS, "c4", 2, limit=10),
        "sweep": lambda: render_sweep_table(_SWEEP, "mrr"),
    }[name]()


def test_unicode_output_still_has_glyphs():
    assert "✓" in _CASES["report"](UNICODE) and "⚑" in _CASES["report"](UNICODE)
    assert "✗ MISS" in _CASES["compare"](UNICODE)
    assert "⋮" in _CASES["retrieved"](UNICODE)


def test_ascii_forms_and_same_marker_choice():
    report = _CASES["report"](ASCII)
    assert "[OK]" in report and "[!]" in report and "[X]" in report
    assert "- not reached" in report
    assert "[OK] found" in _CASES["compare"](ASCII)
    assert "[X] MISS" in _CASES["compare"](ASCII)
    assert "..." in _CASES["retrieved"](ASCII)
    # Same row layout: one marker per stage row in either mode.
    for mode in (UNICODE, ASCII):
        rows = render_report(_VERDICT, RunConfig(), symbols=mode).splitlines()[:4]
        assert [r.split()[0] for r in rows] == ["extraction", "chunking", "retrieval", "generation"]


def test_compare_columns_align_with_wide_ascii_token():
    lines = render_comparison(_LOADERS[:2], symbols=ASCII).splitlines()
    col = lines[0].index("page")
    assert lines[2][col] == "2"  # the page cell sits under its header after "[OK] found"
    assert lines[3][col:col + 1] == "-"


@pytest.mark.parametrize("cmd", [
    ["diagnose", "d.pdf", "Q?", "A"], ["compare", "d.pdf", "A"],
    ["batch", "d.pdf", "q.json"], ["sweep", "d.pdf", "q.json"],
])
def test_ascii_flag_accepted(cmd):
    parser = cli.build_parser()
    assert parser.parse_args(cmd).ascii is False
    assert parser.parse_args(cmd + ["--ascii"]).ascii is True


def test_cli_picks_ascii_for_incapable_stream_or_flag(monkeypatch):
    args = SimpleNamespace(ascii=False)
    monkeypatch.setattr(cli.sys, "stdout", SimpleNamespace(encoding="cp1252"))
    assert cli._symbols_for(args) is ASCII
    monkeypatch.setattr(cli.sys, "stdout", SimpleNamespace(encoding="utf-8"))
    assert cli._symbols_for(args) is UNICODE
    assert cli._symbols_for(SimpleNamespace(ascii=True)) is ASCII


def test_run_compare_end_to_end(monkeypatch):
    monkeypatch.setattr(cli, "compare_loaders", lambda pdf, ans, loaders: _LOADERS)
    monkeypatch.setattr(cli.sys, "stdout", SimpleNamespace(encoding="utf-8"))
    parse = cli.build_parser().parse_args
    assert "✓ found" in cli.run_compare(parse(["compare", "d.pdf", "A"]))
    out = cli.run_compare(parse(["compare", "d.pdf", "A", "--ascii"]))
    assert out.isascii() and "[OK] found" in out
