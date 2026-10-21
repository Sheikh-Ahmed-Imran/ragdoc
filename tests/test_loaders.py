"""Page-map tests for every registered loader, run with the network blocked (T-007).

An offset inside page N's text must resolve to page N for each loader, using that
loader's own page map. PDFs are built here with PyMuPDF and never by ``ragdoc``.
"""

from __future__ import annotations

import socket

import pymupdf
import pytest

from ragdoc.loader import get, names

PAGE_WORDS = ["alphamarker", "bravomarker", "charliemarker"]
LOADERS = ["pymupdf", "pdfplumber", "pypdf"]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("network access attempted in an offline stage")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


@pytest.fixture
def three_page_pdf(tmp_path):
    path = tmp_path / "three.pdf"
    doc = pymupdf.open()
    for word in PAGE_WORDS:
        doc.new_page().insert_text((50, 100), f"{word} line of text", fontsize=12)
    doc.save(path)
    doc.close()
    return path


def test_all_three_registered():
    assert set(LOADERS) <= set(names())


@pytest.mark.parametrize("loader", LOADERS)
def test_offset_on_page_n_resolves_to_n(loader, three_page_pdf):
    doc = get(loader).load(three_page_pdf)
    assert [s.page for s in doc.page_map] == [1, 2, 3]
    for page, word in enumerate(PAGE_WORDS, start=1):
        assert doc.page_at(doc.text.index(word)) == page


@pytest.mark.parametrize("loader", LOADERS)
def test_page_map_covers_text_contiguously(loader, three_page_pdf):
    doc = get(loader).load(three_page_pdf)
    spans = doc.page_map
    assert spans[0].span.start == 0 and spans[-1].span.end == len(doc.text)
    for a, b in zip(spans, spans[1:]):
        assert a.span.end == b.span.start


@pytest.mark.parametrize("loader", LOADERS)
def test_missing_file(loader, tmp_path):
    with pytest.raises(FileNotFoundError):
        get(loader).load(tmp_path / "nope.pdf")
