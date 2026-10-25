"""CLI wiring for ``diagnose --show-retrieved`` (stages are stubbed)."""

import pytest

from ragdoc import cli
from ragdoc.artifacts import CharSpan, CheckResult, CheckStatus, Chunk, RankedChunk, Ranking
from ragdoc.attribution.verdict import StageReport, Verdict

_DIAG = ["diagnose", "doc.pdf", "Q?", "A"]
_CHUNKS = [
    Chunk(id=f"c{i}", text=f"body {i}", span=CharSpan(i, i + 1), page=1) for i in range(5)
]
_RANKING = Ranking([RankedChunk(c.id, 0.9 - 0.1 * i) for i, c in enumerate(_CHUNKS)])


def _verdict(chunking_result):
    return Verdict(
        stage=None, reason_code=None, reason=None, fix=None,
        stages=(
            StageReport("extraction", CheckStatus.PASS, CheckResult.passed()),
            StageReport("chunking", CheckStatus.PASS, chunking_result),
            StageReport("retrieval", CheckStatus.NOT_REACHED, None),
            StageReport("generation", CheckStatus.NOT_REACHED, None),
        ),
    )


def _stub(monkeypatch, verdict):
    class _Stub:
        def __init__(self, *a, **k): pass
        def load(self, *a): return None
        def chunk(self, *a): return list(_CHUNKS)
        def index(self, *a): return None
        def rank(self, *a): return _RANKING
        def run(self, **k): return None

    monkeypatch.setattr(cli, "get_loader", lambda name: _Stub())
    monkeypatch.setattr(cli, "Chunker", _Stub)
    monkeypatch.setattr(cli, "LocalEmbeddingsIndexer", _Stub)
    monkeypatch.setattr(cli, "Retriever", _Stub)
    monkeypatch.setattr(cli, "build_engine", lambda: _Stub())
    monkeypatch.setattr(cli, "compose_verdict", lambda r: verdict)
    monkeypatch.setattr(cli, "render_report", lambda v, c, verbose=False, symbols=None: "REPORT")


def _run(extra):
    return cli.run(cli.build_parser().parse_args(_DIAG + extra))


def test_flag_default_off():
    assert cli.build_parser().parse_args(_DIAG).show_retrieved is False
    assert cli.build_parser().parse_args(_DIAG + ["--show-retrieved"]).show_retrieved is True


def test_no_flag_output_unchanged(monkeypatch):
    _stub(monkeypatch, _verdict(CheckResult.passed({"target_chunk_id": "c1"})))
    assert _run([]) == "REPORT"


def test_flag_appends_list_with_marker(monkeypatch):
    _stub(monkeypatch, _verdict(CheckResult.passed({"target_chunk_id": "c1"})))
    out = _run(["--show-retrieved", "--top-k", "3"])
    assert out.startswith("REPORT\n\n")
    assert "Retrieved chunks" in out and "c0" in out
    assert [l for l in out.splitlines() if "your answer" in l][0].count("c1") == 1


def test_target_below_top_k_via_cli(monkeypatch):
    _stub(monkeypatch, _verdict(CheckResult.passed({"target_chunk_id": "c4"})))
    out = _run(["--show-retrieved", "--top-k", "2"])
    assert "⋮" in out and "5. c4" in out


def test_marks_target_even_when_chunking_displayed_fail(monkeypatch):
    # Missing-context case: survive passed with a target, chunking displayed FAIL.
    verdict = _verdict(CheckResult.passed({"target_chunk_id": "c1"}))
    stages = list(verdict.stages)
    stages[1] = StageReport("chunking", CheckStatus.FAIL, stages[1].result)
    _stub(monkeypatch, Verdict(None, None, None, None, tuple(stages)))
    assert "your answer" in _run(["--show-retrieved"])


@pytest.mark.parametrize("result", [
    CheckResult.failed("cut", {}),
    CheckResult.passed({}),
    None,
])
def test_no_target_shows_note(monkeypatch, result):
    _stub(monkeypatch, _verdict(result))
    out = _run(["--show-retrieved"])
    assert "your answer" not in out
    assert "not located in a single chunk" in out


@pytest.mark.parametrize("base", [
    ["batch", "doc.pdf", "q.json"],
    ["compare", "doc.pdf", "A"],
])
def test_other_commands_reject_flag(base, capfd):
    with pytest.raises(SystemExit) as exc:
        cli.build_parser().parse_args(base + ["--show-retrieved"])
    assert exc.value.code == 2
    assert "unrecognized arguments" in capfd.readouterr().err
