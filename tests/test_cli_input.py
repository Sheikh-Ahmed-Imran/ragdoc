"""CLI answer input: positional vs ``--answer-file``, and ``compare`` taking no question.

Parse-level and error-path tests need no model. The file-vs-positional equivalence
for ``compare`` runs the real offline loader comparison; for ``diagnose`` the
pipeline stages are stubbed and only the answer that reaches the engine is checked.
"""

import json
from pathlib import Path

import pytest

from ragdoc import cli

FIXTURES = Path(__file__).parent / "fixtures"


def _two_column():
    # Hand-asserted ground truth lives in the manifest (ADR-002).
    manifest = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))
    fx = next(f for f in manifest["comparison_fixtures"] if f["name"] == "two_column")
    return str(FIXTURES / fx["pdf"]), fx["expected_answer"]


def _answer_file(tmp_path, text, name="answer.txt"):
    path = tmp_path / name
    path.write_bytes(text.encode("utf-8"))  # bytes: no newline translation on Windows
    return str(path)


# --- compare takes no question -------------------------------------------------

def test_compare_accepts_pdf_and_answer():
    args = cli.build_parser().parse_args(["compare", "doc.pdf", "the answer"])
    assert args.pdf == "doc.pdf"
    assert args.expected_answer == "the answer"
    assert not hasattr(args, "question")


def test_compare_rejects_old_three_positional_form(capfd):
    with pytest.raises(SystemExit) as exc:
        cli.build_parser().parse_args(["compare", "doc.pdf", "Q?", "the answer"])
    assert exc.value.code == 2
    assert "unrecognized arguments" in capfd.readouterr().err


def test_diagnose_still_takes_question():
    args = cli.build_parser().parse_args(["diagnose", "doc.pdf", "Q?", "A"])
    assert (args.question, args.expected_answer) == ("Q?", "A")


# --- error paths (via main, no model) ------------------------------------------

@pytest.mark.parametrize("base", [["compare", "doc.pdf"], ["diagnose", "doc.pdf", "Q?"]])
def test_both_positional_and_file_exits_nonzero(base, tmp_path, capfd):
    path = _answer_file(tmp_path, "A")
    assert cli.main(base + ["A", "--answer-file", path]) != 0
    assert "not both" in capfd.readouterr().err


@pytest.mark.parametrize("base", [["compare", "doc.pdf"], ["diagnose", "doc.pdf", "Q?"]])
def test_neither_positional_nor_file_exits_nonzero(base, capfd):
    assert cli.main(base) != 0
    assert "no expected answer" in capfd.readouterr().err


@pytest.mark.parametrize("base", [["compare", "doc.pdf"], ["diagnose", "doc.pdf", "Q?"]])
def test_missing_answer_file_exits_nonzero(base, tmp_path, capfd):
    missing = str(tmp_path / "nope.txt")
    assert cli.main(base + ["--answer-file", missing]) != 0
    assert "answer file not found" in capfd.readouterr().err


def test_unreadable_answer_file_exits_nonzero(tmp_path, capfd):
    # A directory cannot be read as a file.
    assert cli.main(["compare", "doc.pdf", "--answer-file", str(tmp_path)]) != 0
    assert "cannot read answer file" in capfd.readouterr().err


def test_non_utf8_answer_file_exits_nonzero(tmp_path, capfd):
    path = tmp_path / "bad.txt"
    path.write_bytes(b"\xff\xfe\x00bad")
    assert cli.main(["compare", "doc.pdf", "--answer-file", str(path)]) != 0
    assert "cannot read answer file" in capfd.readouterr().err


# --- file equals positional ----------------------------------------------------

def test_compare_answer_file_matches_positional(tmp_path, capfd):
    pdf, answer = _two_column()
    assert cli.main(["compare", pdf, answer]) == 0
    positional = capfd.readouterr().out

    path = _answer_file(tmp_path, answer + "\n")
    assert cli.main(["compare", pdf, "--answer-file", path]) == 0
    from_file = capfd.readouterr().out

    assert from_file == positional
    assert positional.strip()


def _stub_diagnose(monkeypatch, seen):
    class _Stub:
        def __init__(self, *a, **k): pass
        def load(self, *a): return None
        def chunk(self, *a): return []
        def index(self, *a): return None
        def rank(self, *a): return []

        def run(self, **k):
            seen["answer"] = k["expected_answer"]

    monkeypatch.setattr(cli, "get_loader", lambda name: _Stub())
    monkeypatch.setattr(cli, "Chunker", _Stub)
    monkeypatch.setattr(cli, "LocalEmbeddingsIndexer", _Stub)
    monkeypatch.setattr(cli, "Retriever", _Stub)
    monkeypatch.setattr(cli, "build_engine", lambda: _Stub())
    monkeypatch.setattr(cli, "compose_verdict", lambda r: "verdict")
    monkeypatch.setattr(cli, "render_report", lambda v, c, verbose=False, symbols=None: "report")


def test_diagnose_answer_file_matches_positional(monkeypatch, tmp_path):
    seen = {}
    _stub_diagnose(monkeypatch, seen)
    parser = cli.build_parser()

    cli.run(parser.parse_args(["diagnose", "doc.pdf", "Q?", "the answer"]))
    positional = seen["answer"]

    path = _answer_file(tmp_path, "the answer\n")
    cli.run(parser.parse_args(["diagnose", "doc.pdf", "Q?", "--answer-file", path]))

    assert seen["answer"] == positional == "the answer"


# --- multiline file content ----------------------------------------------------

def test_multiline_answer_file_preserves_interior_and_strips_trailing(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(cli, "compare_loaders", lambda pdf, ans, loaders: seen.update(answer=ans) or [])
    monkeypatch.setattr(cli, "render_comparison", lambda results, verbose=False, symbols=None: "report")

    body = "  indented first line\nsecond line  \n\n  third, after a blank line"
    path = _answer_file(tmp_path, body + "  \n\n\t\n")
    assert cli.main(["compare", "doc.pdf", "--answer-file", path]) == 0

    assert seen["answer"] == body
    assert seen["answer"].count("\n") == 3


def test_answer_file_is_read_as_utf8(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(cli, "compare_loaders", lambda pdf, ans, loaders: seen.update(answer=ans) or [])
    monkeypatch.setattr(cli, "render_comparison", lambda results, verbose=False, symbols=None: "report")

    path = _answer_file(tmp_path, "café — “quoted”\n")
    assert cli.main(["compare", "doc.pdf", "--answer-file", path]) == 0
    assert seen["answer"] == "café — “quoted”"


# --- empty / whitespace-only answer is a usage error ---------------------------

@pytest.fixture
def no_pipeline(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("pipeline work ran for an empty answer")
    monkeypatch.setattr(cli, "get_loader", boom)
    monkeypatch.setattr(cli, "compare_loaders", boom)
    monkeypatch.setattr(cli, "LocalEmbeddingsIndexer", boom)


@pytest.mark.parametrize("base", [["compare", "doc.pdf"], ["diagnose", "doc.pdf", "Q?"]])
@pytest.mark.parametrize("answer", ["", "   ", "\t\n"])
def test_empty_positional_answer_is_usage_error(base, answer, no_pipeline, capfd):
    assert cli.main(base + [answer]) == 2
    captured = capfd.readouterr()
    assert captured.err.startswith("ragdoc: error:")
    assert "empty" in captured.err
    assert captured.err.count("\n") == 1
    assert captured.out == ""


@pytest.mark.parametrize("base", [["compare", "doc.pdf"], ["diagnose", "doc.pdf", "Q?"]])
@pytest.mark.parametrize("content", ["", "  \n\t\n"])
def test_empty_answer_file_is_usage_error(base, content, tmp_path, no_pipeline, capfd):
    path = _answer_file(tmp_path, content)
    assert cli.main(base + ["--answer-file", path]) == 2
    captured = capfd.readouterr()
    assert captured.err.startswith("ragdoc: error:")
    assert "empty" in captured.err
    assert captured.out == ""
