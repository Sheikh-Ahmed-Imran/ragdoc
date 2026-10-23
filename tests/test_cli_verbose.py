"""CLI wiring for ``--verbose`` / ``-v`` (reporters and engine are stubbed)."""

import pytest

from ragdoc import cli

_DIAG = ["diagnose", "doc.pdf", "Q?", "A"]
_CMP = ["compare", "doc.pdf", "A"]


@pytest.mark.parametrize("base", [_DIAG, _CMP])
@pytest.mark.parametrize("flag", ["--verbose", "-v"])
def test_flag_accepted(base, flag):
    assert cli.build_parser().parse_args(base + [flag]).verbose is True


@pytest.mark.parametrize("base", [_DIAG, _CMP])
def test_default_is_not_verbose(base):
    assert cli.build_parser().parse_args(base).verbose is False


@pytest.mark.parametrize("flag", ["--verbose", "-v"])
def test_batch_rejects_flag(flag, capfd):
    with pytest.raises(SystemExit) as exc:
        cli.build_parser().parse_args(["batch", "doc.pdf", "q.json", flag])
    assert exc.value.code == 2
    assert "unrecognized arguments" in capfd.readouterr().err


def _stub_compare(monkeypatch, seen):
    monkeypatch.setattr(cli, "compare_loaders", lambda pdf, ans, loaders: [])

    def fake(results, verbose=False):
        seen["verbose"] = verbose
        return "report"

    monkeypatch.setattr(cli, "render_comparison", fake)


@pytest.mark.parametrize("extra,expected", [([], False), (["-v"], True), (["--verbose"], True)])
def test_compare_dispatches_verbose(monkeypatch, capfd, extra, expected):
    seen = {}
    _stub_compare(monkeypatch, seen)
    assert cli.main(_CMP + extra) == 0
    capfd.readouterr()
    assert seen["verbose"] is expected


def test_diagnose_dispatches_verbose(monkeypatch):
    seen = {}

    class _Stub:
        def __init__(self, *a, **k): pass
        def load(self, *a): return None
        def chunk(self, *a): return []
        def index(self, *a): return None
        def rank(self, *a): return []
        def run(self, **k): return None

    monkeypatch.setattr(cli, "get_loader", lambda name: _Stub())
    monkeypatch.setattr(cli, "Chunker", _Stub)
    monkeypatch.setattr(cli, "LocalEmbeddingsIndexer", _Stub)
    monkeypatch.setattr(cli, "Retriever", _Stub)
    monkeypatch.setattr(cli, "build_engine", lambda: _Stub())
    monkeypatch.setattr(cli, "compose_verdict", lambda r: "verdict")

    def fake(verdict, config, verbose=False):
        seen["verbose"] = verbose
        return "report"

    monkeypatch.setattr(cli, "render_report", fake)
    for extra, expected in (([], False), (["-v"], True)):
        cli.run(cli.build_parser().parse_args(_DIAG + extra))
        assert seen["verbose"] is expected
