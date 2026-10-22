"""Unit tests for ``locate``: exact vs normalized (whitespace / case) matching."""

from ragdoc.artifacts import CharSpan, ExtractedDoc, PageSpan
from ragdoc.attribution.checks import locate


def _doc(text, pages=None):
    if pages is None:
        pages = [PageSpan(1, CharSpan(0, len(text)))]
    return ExtractedDoc(text=text, page_map=pages)


def test_verbatim_match_is_exact():
    result = locate("Net 30", _doc("Payment terms: Net 30 days."))
    assert result.evidence["match"] == "exact"
    assert result.evidence["span"] == CharSpan(15, 21)


def test_case_only_difference_is_normalized():
    result = locate("ABC", _doc("xx abc yy"))
    assert result.is_pass
    assert result.evidence["match"] == "normalized"
    assert result.evidence["span"] == CharSpan(3, 6)
    assert result.evidence["snippet"] == "abc"


def test_case_and_whitespace_together_is_normalized():
    text = "intro\nNet   THIRTY\ndays"
    result = locate("net thirty", _doc(text))
    assert result.evidence["match"] == "normalized"
    span = result.evidence["span"]
    assert text[span.start : span.end] == "Net   THIRTY"


def test_case_match_span_and_page_index_original_text():
    page1 = "Page one filler text.\n"
    page2 = "The Quick BROWN fox."
    text = page1 + page2
    doc = _doc(
        text,
        [
            PageSpan(1, CharSpan(0, len(page1))),
            PageSpan(2, CharSpan(len(page1), len(text))),
        ],
    )
    result = locate("quick brown", doc)
    span = result.evidence["span"]
    assert text[span.start : span.end] == "Quick BROWN"
    assert result.evidence["page"] == 2


def test_length_changing_lowercase_keeps_offsets_valid():
    # "İ".lower() is two characters; offsets must still index the original.
    text = "aİb cd"
    result = locate("CD", _doc(text))
    span = result.evidence["span"]
    assert text[span.start : span.end] == "cd"


def test_absent_answer_fails():
    result = locate("missing", _doc("nothing relevant here"))
    assert result.is_fail


def test_word_reordering_still_fails():
    assert locate("brown quick", _doc("quick brown")).is_fail
