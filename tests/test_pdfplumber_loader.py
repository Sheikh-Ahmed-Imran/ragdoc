import pymupdf
import pytest

from ragdoc.loader import get, names
from ragdoc.loader.pdfplumber_loader import PdfPlumberLoader

PAGE_WORDS = ["alpha", "bravo", "charlie"]


@pytest.fixture
def pdf(tmp_path):
    path = tmp_path / "three.pdf"
    doc = pymupdf.open()
    for word in PAGE_WORDS:
        page = doc.new_page()
        page.insert_text((50, 100), f"{word} marker line", fontsize=12)
    doc.save(path)
    doc.close()
    return path


def test_registered():
    assert "pdfplumber" in names()
    assert isinstance(get("pdfplumber"), PdfPlumberLoader)


def test_contiguous_map_and_page_at(pdf):
    doc = get("pdfplumber").load(pdf)
    spans = doc.page_map
    assert [s.page for s in spans] == [1, 2, 3]
    assert spans[0].span.start == 0
    for a, b in zip(spans, spans[1:]):
        assert a.span.end == b.span.start
    assert spans[-1].span.end == len(doc.text)
    for s, word in zip(spans, PAGE_WORDS):
        offset = doc.text.index(word)
        assert doc.page_at(offset) == s.page


def test_deterministic(pdf):
    loader = get("pdfplumber")
    assert loader.load(pdf) == loader.load(pdf)


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        get("pdfplumber").load(tmp_path / "nope.pdf")
