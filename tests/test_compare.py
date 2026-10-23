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


def test_verbose_reveals_pdfplumber_miss_extraction(no_network):
    """On two_column.pdf the verbose block shows the MISS loader's text differs."""
    pdf, answer = _case("two_column")
    results = compare_loaders(pdf, answer, ORDER)
    by_name = {r.loader: r for r in results}
    assert not by_name["pdfplumber"].found and by_name["pymupdf"].found
    assert by_name["pdfplumber"].text_head != by_name["pymupdf"].text_head
    plain = render_comparison(results)
    out = render_comparison(results, verbose=True)
    assert out.startswith(plain) and out != plain
    blocks = {}
    for chunk in out[len(plain):].strip().split("\n\n"):
        lines = chunk.splitlines()
        blocks[lines[0]] = lines[1:]
    assert "not found" in blocks["pdfplumber"][0]
    assert "exact" in blocks["pymupdf"][0] or "normalized" in blocks["pymupdf"][0]
    assert blocks["pdfplumber"][1] != blocks["pymupdf"][1]


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


def _window_result(text, start, end):
    from ragdoc.compare import _matched_window

    return _matched_window(text, start, end)


def test_matched_window_context_and_ellipses():
    text = "A" * 100 + "ANSWER" + "B" * 100
    snip = _window_result(text, 100, 106)
    assert snip == "…" + "A" * 40 + "ANSWER" + "B" * 40 + "…"
    assert _window_result("xx ANSWER yy", 3, 9) == "xx ANSWER yy"


def test_matched_window_long_span_takes_head_and_is_bounded():
    text = "P" * 50 + "M" * 500 + "S" * 50
    snip = _window_result(text, 50, 550)
    assert len(snip) <= TEXT_HEAD_MAX and snip.startswith("…M") and snip.endswith("M…")
    assert len(_window_result(text, 50, 50 + 200)) <= TEXT_HEAD_MAX


def test_found_snippet_only_on_find_and_head_unchanged(no_network):
    pdf, answer = _case("two_column")
    for r in compare_loaders(pdf, answer, ORDER):
        if r.found:
            assert r.found_snippet and len(r.found_snippet) <= TEXT_HEAD_MAX
        else:
            assert r.found_snippet is None
        assert len(r.text_head) == min(r.char_count, TEXT_HEAD_MAX)
    for r in compare_loaders(pdf, "zzz not in this document qqq", ORDER):
        assert r.found_snippet is None


def test_verbose_find_on_two_column_shows_matched_text_not_head(no_network):
    pdf, answer = _case("two_column")
    results = compare_loaders(pdf, answer, ORDER)
    by_name = {r.loader: r for r in results}
    out = render_comparison(results, verbose=True)
    for name in _entry("two_column")["preserved_by"]:
        r = by_name[name]
        assert answer.split()[0] in r.found_snippet
        chunk = next(c for c in out.split("\n\n")[1:] if c.splitlines()[0] == name)
        line = next(l for l in chunk.splitlines() if "extracted:" in l)
        assert r.found_snippet.replace("\n", "\\n") in line
