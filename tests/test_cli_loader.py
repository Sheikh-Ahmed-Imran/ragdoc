"""``--loader NAME`` on ``diagnose`` and ``batch`` (spec 006).

Parse, validation and wiring tests need no model. The end-to-end test registers a
throwaway loader that fuses inter-word spaces and runs the real pipeline, so it
needs ``sentence_transformers`` and the cached local model.
"""

import json
from pathlib import Path

import pytest

from ragdoc import cli
from ragdoc.artifacts import CharSpan, ExtractedDoc, PageSpan
from ragdoc.loader import Loader, names, register, registry
from ragdoc.loader.pymupdf_loader import PyMuPDFLoader

FIXTURES = Path(__file__).parent / "fixtures"
_MANIFEST = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))

_DIAG = ["diagnose", "doc.pdf", "Q?", "A"]
_BATCH = ["batch", "doc.pdf", "q.json"]


@pytest.fixture
def isolated_registry():
    snapshot = dict(registry._REGISTRY)
    try:
        yield
    finally:
        registry._REGISTRY.clear()
        registry._REGISTRY.update(snapshot)


# --- parsing ---------------------------------------------------------------

@pytest.mark.parametrize("base", [_DIAG, _BATCH])
def test_default_loader_is_pymupdf(base):
    args = cli.build_parser().parse_args(base)
    assert args.loader == "pymupdf"
    assert cli._config_from(args).loader == "pymupdf"


@pytest.mark.parametrize("base", [_DIAG, _BATCH])
@pytest.mark.parametrize("name", ["pymupdf", "pdfplumber", "pypdf"])
def test_loader_flag_reaches_config(base, name):
    args = cli.build_parser().parse_args(base + ["--loader", name])
    assert cli._config_from(args).loader == name


@pytest.mark.parametrize("base", [_DIAG, _BATCH])
def test_unknown_loader_exits_nonzero_listing_registered(base, capfd):
    with pytest.raises(SystemExit) as exc:
        cli.main(base + ["--loader", "nope"])
    assert exc.value.code == 2
    err = capfd.readouterr().err
    assert "unknown loader 'nope'" in err
    for name in names():
        assert name in err


def test_compare_does_not_gain_loader_flag():
    # (argparse would accept ``--loader`` there only as an abbreviation of ``--loaders``.)
    args = cli.build_parser().parse_args(["compare", "doc.pdf", "A"])
    assert not hasattr(args, "loader")
    assert args.loaders is None


# --- wiring ----------------------------------------------------------------

def test_batch_passes_loader_in_config(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(cli, "execute_batch",
                        lambda pdf, config, pairs: seen.update(config=config) or None)
    monkeypatch.setattr(cli, "render_batch_report", lambda result, symbols=None: "report")
    q = tmp_path / "q.json"
    q.write_text(json.dumps([{"question": "Q?", "expected_answer": "A"}]), encoding="utf-8")
    cli.run_batch(cli.build_parser().parse_args(
        ["batch", "doc.pdf", str(q), "--loader", "pdfplumber"]))
    assert seen["config"].loader == "pdfplumber"


def test_diagnose_resolves_loader_via_registry(monkeypatch):
    requested = []

    class _Stub:
        def __init__(self, *a, **k): pass
        def load(self, *a): return None
        def chunk(self, *a): return []
        def index(self, *a): return None
        def rank(self, *a): return []
        def run(self, **k): return None

    def fake_get(name):
        requested.append(name)
        return _Stub()

    monkeypatch.setattr(cli, "get_loader", fake_get)
    monkeypatch.setattr(cli, "Chunker", _Stub)
    monkeypatch.setattr(cli, "LocalEmbeddingsIndexer", _Stub)
    monkeypatch.setattr(cli, "Retriever", _Stub)
    monkeypatch.setattr(cli, "build_engine", lambda: _Stub())
    monkeypatch.setattr(cli, "compose_verdict", lambda r: "verdict")
    seen = {}
    monkeypatch.setattr(cli, "render_report",
                        lambda v, c, verbose=False, symbols=None: seen.update(config=c) or "report")

    parser = cli.build_parser()
    cli.run(parser.parse_args(_DIAG))
    cli.run(parser.parse_args(_DIAG + ["--loader", "pypdf"]))

    assert requested == ["pymupdf", "pypdf"]
    assert seen["config"].loader == "pypdf"


def test_cli_no_longer_hardcodes_pymupdf():
    assert not hasattr(cli, "PyMuPDFLoader")


# --- end to end (model-backed) ---------------------------------------------

class _SpaceFusingLoader(Loader):
    """PyMuPDF text with every space removed: words run together."""

    name = "space-fusing-test"

    def load(self, path):
        text = PyMuPDFLoader().load(path).text.replace(" ", "")
        return ExtractedDoc(text=text, page_map=[PageSpan(page=1, span=CharSpan(0, len(text)))])


def _control():
    return next(f for f in _MANIFEST["fixtures"] if f["name"] == "control")


def _diagnose(extra):
    entry = _control()
    return cli.run(cli.build_parser().parse_args(
        ["diagnose", str(FIXTURES / entry["pdf"]), entry["question"],
         entry["expected_answer"]] + extra))


def test_space_fusing_loader_yields_extraction_verdict(isolated_registry):
    pytest.importorskip("sentence_transformers")
    register(_SpaceFusingLoader)
    report = _diagnose(["--loader", "space-fusing-test"])
    assert "VERDICT: extraction" in report
    assert "loader: space-fusing-test" in report


def test_loader_in_report_and_rerun_same_verdict():
    pytest.importorskip("sentence_transformers")
    first = _diagnose(["--loader", "pypdf"])
    assert "loader: pypdf" in first
    assert _diagnose(["--loader", "pypdf"]) == first


def test_no_flag_equals_explicit_pymupdf():
    pytest.importorskip("sentence_transformers")
    default = _diagnose([])
    assert "loader: pymupdf" in default
    assert _diagnose(["--loader", "pymupdf"]) == default
