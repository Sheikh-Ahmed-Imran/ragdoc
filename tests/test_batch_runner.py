"""Batch runner (T-002): bucketing, build-once, parity with the single path."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ragdoc import batch
from ragdoc.attribution.verdict import build_engine, compose_verdict
from ragdoc.batch import QuestionPair, run_batch
from ragdoc.chunker import Chunker
from ragdoc.config import RunConfig
from ragdoc.indexer.local_embeddings import LocalEmbeddingsIndexer
from ragdoc.loader.pymupdf_loader import PyMuPDFLoader
from ragdoc.retriever import Retriever

FIXTURES = Path(__file__).parent / "fixtures"
MANIFEST = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))["fixtures"]
CONTROL = next(f for f in MANIFEST if f["name"] == "control")
CFG = RunConfig()


def _pairs(entry):
    return [QuestionPair(entry["question"], entry["expected_answer"])]


def _single(entry):
    pdf = FIXTURES / entry["pdf"]
    doc = PyMuPDFLoader().load(pdf)
    chunks = Chunker(CFG).chunk(doc)
    idx = LocalEmbeddingsIndexer(CFG)
    idx.index(chunks)
    ranking = Retriever(idx, CFG).rank(entry["question"])
    return compose_verdict(build_engine().run(
        expected_answer=entry["expected_answer"], doc=doc, chunks=chunks,
        ranking=ranking, config=CFG))


def test_counts_sum_and_absent_answer_is_extraction():
    pairs = _pairs(CONTROL) + [QuestionPair("q?", "text that is nowhere in the doc xyzzy")]
    res = run_batch(FIXTURES / CONTROL["pdf"], CFG, pairs)
    assert sum(res.counts.values()) == len(pairs) == res.total
    assert [r.question for r in res.results] == [p.question for p in pairs]
    assert res.results[0].bucket == "no-failure"
    assert res.results[1].bucket == "extraction"


def test_unevaluable_pair_is_withheld():
    res = run_batch(FIXTURES / CONTROL["pdf"], CFG, [QuestionPair("q?", "   ")])
    assert res.results[0].bucket == "withheld"
    assert res.results[0].stage is None


@pytest.mark.parametrize("entry", MANIFEST, ids=[f["name"] for f in MANIFEST])
def test_bucket_matches_single_question_path(entry):
    res = run_batch(FIXTURES / entry["pdf"], CFG, _pairs(entry))
    v = _single(entry)
    assert res.results[0].stage == v.stage
    assert res.results[0].reason == v.reason
    expected = "withheld" if v.withheld else (v.stage or "no-failure")
    assert res.results[0].bucket == expected


def test_index_built_once(monkeypatch):
    built = []
    orig = batch.LocalEmbeddingsIndexer.__init__

    def counting(self, *a, **k):
        built.append(1)
        orig(self, *a, **k)

    monkeypatch.setattr(batch.LocalEmbeddingsIndexer, "__init__", counting)
    pairs = _pairs(CONTROL) * 3
    run_batch(FIXTURES / CONTROL["pdf"], CFG, pairs)
    assert len(built) == 1


def test_reproducible_counts():
    pairs = [QuestionPair(e["question"], e["expected_answer"]) for e in MANIFEST]
    a = run_batch(FIXTURES / CONTROL["pdf"], CFG, pairs)
    b = run_batch(FIXTURES / CONTROL["pdf"], CFG, pairs)
    assert dict(a.counts) == dict(b.counts)
