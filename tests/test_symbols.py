import io
import socket
import sys
from types import SimpleNamespace

from ragdoc.report.symbols import ASCII, UNICODE, Symbols, resolve


def _stream(encoding):
    return SimpleNamespace(encoding=encoding)


def test_unicode_set_exact():
    assert (UNICODE.passed, UNICODE.fail, UNICODE.flagged, UNICODE.gap) == (
        "✓", "✗", "⚑", "⋮")
    assert UNICODE.not_reached == "—"
    assert UNICODE.found == "✓ found"
    assert UNICODE.miss == "✗ MISS"


def test_ascii_set_exact():
    assert (ASCII.passed, ASCII.fail, ASCII.flagged, ASCII.gap,
            ASCII.not_reached) == ("[OK]", "[X]", "[!]", "...", "-")
    assert ASCII.found == "[OK] found"
    assert ASCII.miss == "[X] MISS"


def test_ascii_set_is_pure_ascii():
    assert all(v.isascii() for v in vars(ASCII).values())


def test_symbols_is_frozen():
    import dataclasses
    import pytest
    with pytest.raises(dataclasses.FrozenInstanceError):
        UNICODE.passed = "x"  # type: ignore[misc]
    assert isinstance(UNICODE, Symbols)


def test_force_ascii_wins_over_capable_stream():
    assert resolve(_stream("utf-8"), force_ascii=True) is ASCII


def test_force_ascii_with_incapable_stream():
    assert resolve(_stream("cp1252"), force_ascii=True) is ASCII


def test_utf8_stream_gets_unicode():
    assert resolve(_stream("utf-8")) is UNICODE
    assert resolve(io.TextIOWrapper(io.BytesIO(), encoding="utf-8")) is UNICODE


def test_legacy_codepages_get_ascii():
    for enc in ("cp1252", "cp437", "cp850", "ascii"):
        assert resolve(_stream(enc)) is ASCII


def test_missing_or_unknown_encoding_is_not_capable():
    assert resolve(_stream(None)) is ASCII
    assert resolve(_stream("")) is ASCII
    assert resolve(_stream("no-such-codec")) is ASCII
    assert resolve(object()) is ASCII
    assert resolve(io.BytesIO()) is ASCII


def test_resolve_needs_no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network used")
    monkeypatch.setattr(socket, "socket", boom)
    assert resolve(_stream("utf-8")) is UNICODE


def test_module_imports_no_engine_stage():
    import ragdoc.report.symbols as mod
    src = open(mod.__file__, encoding="utf-8").read()
    assert "ragdoc" not in src.replace("ragdoc.report.symbols", "")
