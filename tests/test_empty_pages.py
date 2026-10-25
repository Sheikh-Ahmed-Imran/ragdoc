"""Empty-page helper and diagnose advisory (T-015)."""

from __future__ import annotations

from ragdoc.artifacts import CharSpan, ExtractedDoc, PageSpan
from ragdoc.config import RunConfig
from ragdoc.report import render_report

from tests.test_report import _missing_context_verdict


def _doc(*lengths: int) -> ExtractedDoc:
    spans, pos = [], 0
    for i, n in enumerate(lengths, start=1):
        spans.append(PageSpan(i, CharSpan(pos, pos + n)))
        pos += n
    return ExtractedDoc(text="x" * pos, page_map=spans)


def test_empty_pages_in_order():
    assert _doc(5, 0, 3, 0).empty_pages == (2, 4)


def test_empty_pages_none():
    assert _doc(5, 3).empty_pages == ()
    assert ExtractedDoc(text="").empty_pages == ()


def test_advisory_present_regardless_of_verbose():
    doc = _doc(5, 0, 3, 0)
    for verbose in (False, True):
        out = render_report(
            _missing_context_verdict(), RunConfig(), verbose=verbose,
            empty_pages=doc.empty_pages, total_pages=len(doc.page_map),
        )
        assert "2 of 4 pages had no extractable text: p.2, p.4" in out


def test_advisory_absent_and_report_unchanged():
    v, c = _missing_context_verdict(), RunConfig()
    base = render_report(v, c)
    assert "no extractable text" not in base
    assert render_report(v, c, empty_pages=()) == base
    with_adv = render_report(v, c, empty_pages=(2,), total_pages=3)
    assert [l for l in with_adv.splitlines() if "no extractable" not in l and l] == [
        l for l in base.splitlines() if l
    ]
