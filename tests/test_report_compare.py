"""Comparison reporter tests (T-005): pure, hand-built results."""

from __future__ import annotations

from ragdoc.compare import LoaderResult
from ragdoc.report.compare import render_comparison

_RESULTS = [
    LoaderResult("pdfplumber", True, True, 2, "exact", 1200, 3),
    LoaderResult("pypdf", False, True, None, None, 1100, 3),
    LoaderResult("other", False, False, None, None, 0, 0),
]


def test_one_row_per_loader_with_all_columns():
    out = render_comparison(_RESULTS).splitlines()
    assert len(out) == 2 + len(_RESULTS)
    found = out[2]
    for token in ("pdfplumber", "found", "2", "exact", "1200", "3"):
        assert token in found


def test_miss_is_distinct_from_found_and_unevaluated():
    out = render_comparison(_RESULTS).splitlines()
    assert "✗ MISS" in out[3] and "✗" not in out[2]
    assert "MISS" not in out[4] and "not evaluated" in out[4]


def test_no_metric_or_recommendation():
    out = render_comparison(_RESULTS).lower()
    for word in ("precision", "recall", "max ", "best", "recommend"):
        assert word not in out


def test_empty():
    assert "No loaders" in render_comparison([])


def _verbose_results():
    long = "x" * 300
    return [
        LoaderResult("pdfplumber", False, True, None, None, 5000, 2, text_head=long),
        LoaderResult("pymupdf", True, True, 1, "exact", 12, 1, text_head="line one\nthe answer"),
    ]


def test_verbose_false_is_byte_identical_to_default():
    results = _verbose_results()
    assert render_comparison(results, verbose=False) == render_comparison(results)


def test_verbose_includes_match_attempt_and_snippet():
    results = _verbose_results()
    out = render_comparison(results, verbose=True)
    assert out.startswith(render_comparison(results))
    assert "match attempt: not found" in out and "match attempt: exact" in out
    assert "line one\\nthe answer" in out  # control characters escaped onto one line


def test_verbose_snippet_is_bounded_with_truncation_indicator():
    out = render_comparison(_verbose_results(), verbose=True)
    assert "… (300 of 5000 chars)" in out
    # a complete extraction carries no truncation indicator
    assert "of 12 chars" not in out


# --- matched-region snippet on a find (T-006) ---------------------------------

_HEAD = "DOCUMENT HEAD unrelated preamble " * 7


def _find(snippet="…context before THE ANSWER context after…", **kw):
    return LoaderResult("pymupdf", True, True, 4, "exact", 5000, 6, text_head=_HEAD, found_snippet=snippet, **kw)


def _extracted(out: str, loader: str) -> str:
    block = out.split("\n\n")[1:]
    chunk = next(c for c in block if c.splitlines()[0] == loader)
    return next(l for l in chunk.splitlines() if l.strip().startswith("extracted:"))


def test_verbose_find_shows_matched_region_and_page_not_head():
    r = _find()
    out = render_comparison([r], verbose=True)
    line = _extracted(out, "pymupdf")
    assert "THE ANSWER" in line and "context before" in line and "context after" in line
    assert "DOCUMENT HEAD" not in line and line.split(":", 1)[1].strip() != r.text_head
    assert "page:          4" in out


def test_verbose_miss_shows_head_as_before():
    r = LoaderResult("pdfplumber", False, True, None, None, 5000, 2, text_head="head text")
    line = _extracted(render_comparison([r], verbose=True), "pdfplumber")
    assert line == '  extracted:     "head text" … (9 of 5000 chars)'
    assert "page:" not in render_comparison([r], verbose=True).split("\n\n", 1)[1]


def test_verbose_empty_extraction_says_no_text():
    r = LoaderResult("pypdf", False, True, None, None, 0, 0)
    assert _extracted(render_comparison([r], verbose=True), "pypdf").endswith("(no text extracted)")


def test_verbose_found_snippet_bounded_with_indicator_and_escaped():
    from ragdoc.compare import TEXT_HEAD_MAX

    snip = "…" + "a\nb " * 50
    snip = snip[: TEXT_HEAD_MAX - 1] + "…"
    line = _extracted(render_comparison([_find(snip)], verbose=True), "pymupdf")
    assert len(snip) <= TEXT_HEAD_MAX
    assert "…" in line and line.count("\n") == 0 and "\\n" in line


def test_verbose_false_equals_default_with_found_and_missed():
    results = [_find(), LoaderResult("pdfplumber", False, True, None, None, 5000, 2, text_head=_HEAD)]
    assert render_comparison(results, verbose=False) == render_comparison(results)
    assert "THE ANSWER" not in render_comparison(results)


def test_numeric_columns_right_aligned_with_rule():
    out = render_comparison(_RESULTS).splitlines()
    assert set(out[1].replace(" ", "")) == {"-"}
    assert len({len(line) for line in out[:5]}) == 1  # chars/pages end flush


def test_symbol_cells_stay_aligned_in_both_modes():
    from ragdoc.report.symbols import ASCII, UNICODE

    for symbols in (UNICODE, ASCII):
        out = render_comparison(_RESULTS, symbols=symbols).splitlines()
        assert len({len(line) for line in out[:5]}) == 1
