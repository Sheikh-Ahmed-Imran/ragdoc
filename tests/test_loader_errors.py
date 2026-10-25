"""Unreadable / non-PDF files fail cleanly with ``LoaderError`` (T-011)."""

import json
import struct
import zlib

import pytest

from ragdoc import cli
from ragdoc.loader import LoaderError
from ragdoc.loader.pdfplumber_loader import PdfPlumberLoader
from ragdoc.loader.pymupdf_loader import PyMuPDFLoader
from ragdoc.loader.pypdf_loader import PyPDFLoader

LOADERS = [PyMuPDFLoader, PyPDFLoader, PdfPlumberLoader]
_SIG = b"\x89PNG\r\n\x1a\n"


def _chunk(tag: bytes, data: bytes) -> bytes:
    body = tag + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))


# A valid 1x1 PNG: PyMuPDF opens it as an image "document", so it must be rejected.
_PNG = (
    _SIG
    + _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0))
    + _chunk(b"IDAT", zlib.compress(b"\x00\x00"))
    + _chunk(b"IEND", b"")
)

_BAD = {
    "png": _PNG,
    "bad_png": _SIG + b"\x00" * 64,
    "empty": b"",
    "garbage": b"not a pdf at all\n" * 5,
}


@pytest.fixture(params=sorted(_BAD))
def bad_pdf(request, tmp_path):
    path = tmp_path / "bad.pdf"
    path.write_bytes(_BAD[request.param])
    return path


def test_loader_error_is_value_error():
    assert issubclass(LoaderError, ValueError)


@pytest.mark.parametrize("loader_cls", LOADERS)
def test_bad_file_raises_loader_error(loader_cls, bad_pdf):
    with pytest.raises(LoaderError, match="not a readable PDF") as exc:
        loader_cls().load(bad_pdf)
    assert "bad.pdf" in str(exc.value)


@pytest.mark.parametrize("loader_cls", LOADERS)
def test_missing_file_still_file_not_found(loader_cls, tmp_path):
    with pytest.raises(FileNotFoundError):
        loader_cls().load(tmp_path / "nope.pdf")


@pytest.mark.parametrize("argv", [
    ["diagnose", "{pdf}", "q?", "a"],
    ["compare", "{pdf}", "a"],
    ["batch", "{pdf}", "{q}"],
    ["sweep", "{pdf}", "{q}"],
])
def test_cli_reports_clean_error(argv, bad_pdf, tmp_path, capfd):
    q = tmp_path / "q.json"
    q.write_text(json.dumps([{"question": "q?", "expected_answer": "a"}]), encoding="utf-8")
    argv = [a.format(pdf=bad_pdf, q=q) for a in argv]
    assert cli.main(argv) == 2
    err = capfd.readouterr().err
    assert err.startswith("ragdoc: error: ")
    assert "not a readable PDF" in err
    assert "Traceback" not in err
    assert len(err.strip().splitlines()) == 1
