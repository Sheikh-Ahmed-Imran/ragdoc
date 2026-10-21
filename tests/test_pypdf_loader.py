import socket
from pathlib import Path

import pytest

from ragdoc.loader import get
from ragdoc.loader.pypdf_loader import PyPDFLoader

FIXTURES = Path(__file__).parent / "fixtures"


def test_registry_returns_pypdf_loader():
    loader = get("pypdf")
    assert isinstance(loader, PyPDFLoader)
    assert loader.name == "pypdf"


def test_missing_path_raises():
    with pytest.raises(FileNotFoundError):
        PyPDFLoader().load(FIXTURES / "does-not-exist.pdf")


def test_page_map_indexes_returned_text():
    doc = PyPDFLoader().load(FIXTURES / "unembedded_heading.pdf")
    assert len(doc.page_map) > 1
    assert doc.page_map[0].span.start == 0
    assert doc.page_map[-1].span.end == len(doc.text)
    for prev, nxt in zip(doc.page_map, doc.page_map[1:]):
        assert prev.span.end == nxt.span.start


def test_offset_in_page_n_resolves_to_page_n():
    doc = PyPDFLoader().load(FIXTURES / "unembedded_heading.pdf")
    checked = 0
    for ps in doc.page_map:
        if ps.span.end > ps.span.start:
            assert doc.page_at(ps.span.start) == ps.page
            assert doc.page_at(ps.span.end - 1) == ps.page
            checked += 1
    assert checked > 1


def test_loads_offline_and_deterministically(monkeypatch):
    def no_network(*a, **k):
        raise AssertionError("network used")

    monkeypatch.setattr(socket, "socket", no_network)
    first = PyPDFLoader().load(FIXTURES / "unembedded_heading.pdf")
    second = PyPDFLoader().load(FIXTURES / "unembedded_heading.pdf")
    assert first == second
