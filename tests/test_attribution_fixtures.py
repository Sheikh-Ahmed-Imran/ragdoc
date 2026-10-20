"""Run the attribution engine over the constructed fixtures (T-009).

The PDFs in ``tests/fixtures/`` are built by ``build_fixtures.py``, which does not
import ``ragdoc``; the expected answers and known causes come from
``manifest.json`` and are asserted there by hand (ADR-002, ADR-003). The control
fixture exists so this harness is capable of failing: a detector that blamed
every document would fail it.

The extraction and chunking stages are tested with the network blocked. The full
pipeline tests embed with the local model, which must already be cached (the
first-ever run downloads it).
"""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from ragdoc.attribution.checks import locate, survive
from ragdoc.attribution.verdict import build_engine, compose_verdict
from ragdoc.chunker import Chunker
from ragdoc.config import RunConfig
from ragdoc.indexer.local_embeddings import LocalEmbeddingsIndexer
from ragdoc.loader.pymupdf_loader import PyMuPDFLoader
from ragdoc.retriever import Retriever

FIXTURES = Path(__file__).parent / "fixtures"
MANIFEST = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))["fixtures"]
IDS = [f["name"] for f in MANIFEST]


@pytest.fixture
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("network access attempted in an offline stage")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


def _pdf(entry) -> Path:
    return FIXTURES / entry["pdf"]


@pytest.mark.parametrize("entry", MANIFEST, ids=IDS)
def test_extraction_and_chunking_offline(entry, no_network):
    config = RunConfig()
    doc = PyMuPDFLoader().load(_pdf(entry))
    located = locate(entry["expected_answer"], doc)
    assert located.is_pass, "fixture answer must survive extraction"

    chunks = Chunker(config).chunk(doc)
    survived = survive(located.evidence["span"], chunks)
    if entry["name"] == "split_list":
        assert survived.is_fail
    else:
        assert survived.is_pass
        assert survived.evidence["context_missing"] == (
            entry["name"] == "unembedded_heading"
        )


@pytest.mark.parametrize("entry", MANIFEST, ids=IDS)
def test_fixture_verdict(entry):
    config = RunConfig()
    doc = PyMuPDFLoader().load(_pdf(entry))
    chunks = Chunker(config).chunk(doc)
    indexer = LocalEmbeddingsIndexer(config)
    indexer.index(chunks)
    ranking = Retriever(indexer, config).rank(entry["question"])
    verdict = compose_verdict(
        build_engine().run(
            expected_answer=entry["expected_answer"],
            doc=doc,
            chunks=chunks,
            ranking=ranking,
            config=config,
        )
    )

    assert not verdict.withheld
    if entry["known_cause"] == "none":
        assert verdict.stage is None
        assert not verdict.has_failure
    else:
        assert verdict.stage == entry["known_cause"]
        assert verdict.reason_code == entry["reason_code"]
