"""Pure tests for the retrieved-chunks view (no model, no PDF)."""

from ragdoc.artifacts import CharSpan, Chunk, RankedChunk, Ranking
from ragdoc.report.retrieved import render_retrieved


def _chunks(n, text=None):
    return [
        Chunk(id=f"c{i}", text=text or f"text of chunk {i}",
              span=CharSpan(i * 10, i * 10 + 10), page=i + 1)
        for i in range(n)
    ]


def _ranking(chunks):
    # c0 best ... descending scores.
    return Ranking([RankedChunk(c.id, 0.9 - 0.05 * i) for i, c in enumerate(chunks)])


def _rows(out):
    return [l.strip() for l in out.splitlines() if l.strip().startswith(tuple("0123456789"))]


def test_top_k_rows_carry_rank_id_page_score_snippet():
    chunks = _chunks(6)
    out = render_retrieved(_ranking(chunks), chunks, "c1", 3)
    rows = _rows(out)
    assert len(rows) == 3
    assert rows[0].startswith("1. c0") and "p.1" in rows[0]
    assert "0.9000" in rows[0] and '"text of chunk 0"' in rows[0]
    assert "0.8500" in rows[1]


def test_target_within_top_k_is_marked_once():
    chunks = _chunks(6)
    out = render_retrieved(_ranking(chunks), chunks, "c1", 3)
    marked = [l for l in out.splitlines() if "your answer" in l]
    assert len(marked) == 1 and " c1 " in marked[0]
    assert "⋮" not in out


def test_target_below_top_k_follows_gap_with_true_rank():
    chunks = _chunks(8)
    out = render_retrieved(_ranking(chunks), chunks, "c6", 3)
    lines = out.splitlines()
    gap = lines.index("  ⋮")
    last = lines[gap + 1]
    assert gap == len(lines) - 2
    assert last.strip().startswith("7. c6") and "p.7" in last
    assert "0.6000" in last and "your answer" in last
    assert len([l for l in lines if "your answer" in l]) == 1


def test_no_target_shows_note_and_no_marker():
    chunks = _chunks(5)
    out = render_retrieved(_ranking(chunks), chunks, None, 3)
    assert len(_rows(out)) == 3
    assert "your answer" not in out
    assert "not located in a single chunk" in out


def test_snippet_bounded_escaped_single_line():
    chunks = _chunks(2, text="a\nb\tc\r" + "x" * 500 + "\x0b ")
    out = render_retrieved(_ranking(chunks), chunks, None, 2)
    rows = _rows(out)
    assert len(rows) == 2
    assert "\\n" in rows[0] and "\\t" in rows[0] and "\\r" in rows[0]
    assert "x" * 200 not in out
    assert "…" in rows[0]
    assert len(rows[0].split("  ", 4)[-1]) < 140


def test_escaped_snippet_not_cut_inside_escape():
    chunks = _chunks(1, text="x" * 119 + "\n" + "tail")
    out = render_retrieved(_ranking(chunks), chunks, None, 1, limit=120)
    row = _rows(out)[0]
    assert '\\"' not in row and not row.rstrip("…").rstrip('"').endswith("\\")


def test_unknown_chunk_id_row_is_skipped():
    chunks = _chunks(3)
    ranking = Ranking([RankedChunk("ghost", 0.99)] + list(_ranking(chunks).entries))
    out = render_retrieved(ranking, chunks, None, 3)
    assert "ghost" not in out
    assert len(_rows(out)) == 2
