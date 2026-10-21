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
