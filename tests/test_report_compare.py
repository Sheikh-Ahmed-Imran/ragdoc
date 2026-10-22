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
