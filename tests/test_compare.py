import json
import socket
from pathlib import Path

import pytest

from ragdoc.compare import TEXT_HEAD_MAX, compare_loaders
from ragdoc.loader import UnknownLoaderError
from ragdoc.report.compare import render_comparison

FIXTURES = Path(__file__).parent / "fixtures"
ORDER = ["pymupdf", "pdfplumber", "pypdf"]


def _manifest():
    return json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))


def _entry(name):
    m = _manifest()
    for f in m["fixtures"] + m["comparison_fixtures"]:
        if f["name"] == name:
            return f


def _case(name):
    f = _entry(name)
    return FIXTURES / f["pdf"], f["expected_answer"]


@pytest.fixture
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("network access attempted in an offline stage")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


def test_one_result_per_loader_in_given_order():
    pdf, answer = _case("control")
    assert [r.loader for r in compare_loaders(pdf, answer, ORDER)] == ORDER
    rev = list(reversed(ORDER))
    assert [r.loader for r in compare_loaders(pdf, answer, rev)] == rev


def test_default_is_all_registered():
    pdf, answer = _case("control")
    assert {r.loader for r in compare_loaders(pdf, answer)} == set(ORDER)


def test_found_has_page_and_match_and_absent_does_not():
    pdf, answer = _case("control")
    for r in compare_loaders(pdf, answer, ORDER):
        assert r.found and r.evaluated
        assert r.page is not None and r.match in ("exact", "normalized")
        assert r.char_count > 0 and r.page_count >= 1
    for r in compare_loaders(pdf, "zzz not in this document qqq", ORDER):
        assert not r.found and r.evaluated and r.page is None and r.match is None


def test_text_head_is_bounded_head_of_extracted_text():
    pdf, answer = _case("control")
    for r in compare_loaders(pdf, answer, ORDER):
        assert len(r.text_head) <= TEXT_HEAD_MAX
        assert r.text_head == r.text_head[: r.char_count]
        assert r.text_head or r.char_count == 0
        assert len(r.text_head) == min(r.char_count, TEXT_HEAD_MAX)


def test_empty_answer_is_unevaluated():
    pdf, _ = _case("control")
    for r in compare_loaders(pdf, "  ", ORDER):
        assert not r.found and not r.evaluated


def test_unknown_loader_raises():
    pdf, answer = _case("control")
    with pytest.raises(UnknownLoaderError):
        compare_loaders(pdf, answer, ["pymupdf", "nope"])


def test_deterministic_and_offline(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network used")

    monkeypatch.setattr(socket, "socket", boom)
    pdf, answer = _case("split_list")
    assert compare_loaders(pdf, answer, ORDER) == compare_loaders(pdf, answer, ORDER)


def test_divergence_fixture_found_vs_miss(no_network):
    """The hand-asserted manifest says which loaders preserve the answer."""
    entry = _entry("two_column")
    pdf, answer = _case("two_column")
    by_name = {r.loader: r for r in compare_loaders(pdf, answer, ORDER)}
    assert entry["preserved_by"] and entry["not_preserved_by"]
    assert set(entry["preserved_by"]) | set(entry["not_preserved_by"]) == set(ORDER)
    for name in entry["preserved_by"]:
        assert by_name[name].found and by_name[name].page == 1
    for name in entry["not_preserved_by"]:
        r = by_name[name]
        assert not r.found and r.evaluated and r.page is None


def test_divergence_shows_in_report(no_network):
    pdf, answer = _case("two_column")
    rows = {
        line.split()[0]: line
        for line in render_comparison(compare_loaders(pdf, answer, ORDER)).splitlines()[2:]
    }
    assert "✗ MISS" in rows["pdfplumber"]
    assert "✓ found" in rows["pymupdf"] and "✓ found" in rows["pypdf"]
    assert "✗" not in rows["pymupdf"] + rows["pypdf"]


def test_diagnose_reproduces_v0_verdict():
    """`diagnose` still blames chunking / boundary_cut on the split-list fixture."""
    pytest.importorskip("sentence_transformers")
    from ragdoc.cli import build_parser, run

    entry = _entry("split_list")
    args = build_parser().parse_args(
        ["diagnose", str(FIXTURES / entry["pdf"]), entry["question"], entry["expected_answer"]]
    )
    report = run(args)
    assert f"VERDICT: {entry['known_cause']}" in report
    assert "loader: pymupdf" in report
