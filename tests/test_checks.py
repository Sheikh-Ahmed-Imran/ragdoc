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


# --------------------------------------------------------------------------- #
# Repeated answers: every copy is judged; any surviving copy counts (T-013)
# --------------------------------------------------------------------------- #

from ragdoc.artifacts import Chunk, RankedChunk, Ranking
from ragdoc.attribution.checks import rank, survive
from ragdoc.attribution.engine import AttributionEngine
from ragdoc.attribution.verdict import build_engine, compose_verdict
from ragdoc.config import RunConfig

# "xx NET30 yy ... zz NET30 ww": copies at 3-8 and 20-25.
_TEXT = "xx NET30 yy aaaa zz NET30 ww"
_FIRST, _SECOND = CharSpan(3, 8), CharSpan(20, 25)


def _chunks():
    # c0 cuts the first copy; c1 holds the second whole; c2 repeats it (overlap).
    return [
        Chunk("c0", _TEXT[0:6], CharSpan(0, 6), 1),
        Chunk("c1", _TEXT[6:26], CharSpan(6, 26), 1),
        Chunk("c2", _TEXT[18:], CharSpan(18, len(_TEXT)), 1),
    ]


def test_locate_returns_every_exact_occurrence():
    result = locate("NET30", _doc(_TEXT))
    assert result.evidence["spans"] == (_FIRST, _SECOND)
    assert result.evidence["span"] == _FIRST


def test_locate_never_mixes_tiers():
    # One verbatim copy, one differing only in case: only the exact tier counts.
    result = locate("NET30", _doc("a NET30 b net30 c"))
    assert result.evidence["match"] == "exact"
    assert result.evidence["spans"] == (CharSpan(2, 7),)


def test_locate_all_normalized_occurrences():
    result = locate("net30", _doc("a NET30 b Net30 c"))
    assert result.evidence["match"] == "normalized"
    assert len(result.evidence["spans"]) == 2


def test_survive_passes_if_any_copy_is_whole():
    result = survive(locate("NET30", _doc(_TEXT)).evidence["spans"], _chunks())
    assert result.is_pass
    assert result.evidence["target_chunk_ids"] == ("c1", "c2")
    assert result.evidence["target_chunk_id"] == "c1"
    assert result.evidence["located_span"] == _SECOND


def test_survive_fails_only_when_no_copy_is_whole():
    chunks = [Chunk("c0", "", CharSpan(0, 6), 1), Chunk("c1", "", CharSpan(6, 18), 1),
              Chunk("c2", "", CharSpan(18, 22), 1), Chunk("c3", "", CharSpan(22, 28), 1)]
    result = survive([_FIRST, _SECOND], chunks)
    assert result.is_fail
    assert result.evidence["crossing_chunk_ids"] == ["c0", "c1", "c2", "c3"]


def test_survive_single_span_still_accepted():
    assert survive(_SECOND, _chunks()).evidence["target_chunk_id"] == "c1"


def test_rank_takes_best_ranking_candidate_and_breaks_ties_by_order():
    ranking = Ranking([RankedChunk(i, 0.0) for i in ("c2", "c0", "c1")])
    assert rank(("c1", "c2"), ranking, 1).evidence["target_chunk_id"] == "c2"
    assert rank(("c1", "c2"), ranking, 1).is_pass
    assert rank("c1", ranking, 1).is_fail
    # Absent candidate -> withheld.
    assert rank(("c1", "zz"), ranking, 5).is_unevaluated


def test_engine_reports_best_ranking_copy():
    doc = _doc(_TEXT)
    ranking = Ranking([RankedChunk(i, 0.0) for i in ("c2", "c0", "c1")])
    result = build_engine().run(
        expected_answer="NET30", doc=doc, chunks=_chunks(), ranking=ranking,
        config=RunConfig(top_k=1),
    )
    verdict = compose_verdict(result)
    assert verdict.stage is None  # c2 holds the second copy and is rank 1
    survived = result.result_for_check("survive")
    assert survived.evidence["target_chunk_id"] == "c2"
    assert result.result_for_check("rank").evidence["position"] == 1
    located = result.result_for_check("locate")
    assert located.evidence["span"] == _SECOND
    assert located.evidence["snippet"] == "NET30"


def test_engine_repeated_answer_not_a_chunking_failure():
    ranking = Ranking([RankedChunk(i, 0.0) for i in ("c0", "c1", "c2")])
    result = build_engine().run(
        expected_answer="NET30", doc=_doc(_TEXT), chunks=_chunks(), ranking=ranking,
        config=RunConfig(top_k=10),
    )
    assert compose_verdict(result).stage is None
