import json
import socket
from pathlib import Path

import pytest

from ragdoc.compare import compare_loaders
from ragdoc.loader import UnknownLoaderError

FIXTURES = Path(__file__).parent / "fixtures"
ORDER = ["pymupdf", "pdfplumber", "pypdf"]


def _case(name):
    for f in json.loads((FIXTURES / "manifest.json").read_text())["fixtures"]:
        if f["name"] == name:
            return FIXTURES / f["pdf"], f["expected_answer"]


def test_one_result_per_loader_in_given_order():
    pdf, answer = _case("control")
    assert [r.loader for r in compare_loaders(pdf, answer, ORDER)] == ORDER
    rev = list(reversed(ORDER))
    assert [r.loader for r in compare_loaders(pdf, answer, rev)] == rev


def test_default_is_all_registered():
    pdf, answer = _case("control")
    assert {r.loader for r in compare_loaders(pdf, answer)} == set(ORDER)


def test_found_has_page_and_match_and_absent_does_not():
    pdf, answer = _case("control")
    for r in compare_loaders(pdf, answer, ORDER):
        assert r.found and r.evaluated
        assert r.page is not None and r.match in ("exact", "normalized")
        assert r.char_count > 0 and r.page_count >= 1
    for r in compare_loaders(pdf, "zzz not in this document qqq", ORDER):
        assert not r.found and r.evaluated and r.page is None and r.match is None


def test_empty_answer_is_unevaluated():
    pdf, _ = _case("control")
    for r in compare_loaders(pdf, "  ", ORDER):
        assert not r.found and not r.evaluated


def test_unknown_loader_raises():
    pdf, answer = _case("control")
    with pytest.raises(UnknownLoaderError):
        compare_loaders(pdf, answer, ["pymupdf", "nope"])


def test_deterministic_and_offline(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network used")

    monkeypatch.setattr(socket, "socket", boom)
    pdf, answer = _case("split_list")
    assert compare_loaders(pdf, answer, ORDER) == compare_loaders(pdf, answer, ORDER)
