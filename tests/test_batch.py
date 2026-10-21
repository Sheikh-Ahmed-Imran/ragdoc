"""Batch mode over a known-cause fixture (T-005).

``batch_doc.pdf`` joins the split-list, unembedded-heading and control fixtures
into one document (batch runs one document). The question set is
``batch_questions.json``; the expected buckets are asserted by hand under
``batch_fixtures`` in ``manifest.json`` (ADR-002, ADR-003), never generated.

The parsing and extraction/chunking tests run with the network blocked. The
tests that embed need ``sentence_transformers`` and the cached local model.
"""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from ragdoc import batch, cli
from ragdoc.attribution.checks import locate, survive
from ragdoc.attribution.verdict import build_engine, compose_verdict
from ragdoc.batch import BUCKETS, QuestionPair, load_question_set, run_batch
from ragdoc.chunker import Chunker
from ragdoc.config import RunConfig
from ragdoc.indexer.local_embeddings import LocalEmbeddingsIndexer
from ragdoc.loader.pymupdf_loader import PyMuPDFLoader
from ragdoc.retriever import Retriever

FIXTURES = Path(__file__).parent / "fixtures"
_MANIFEST = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))
SPEC = _MANIFEST["batch_fixtures"][0]
PDF = FIXTURES / SPEC["pdf"]
QUESTIONS = FIXTURES / SPEC["questions"]
EXTRA = [QuestionPair(e["question"], e["expected_answer"]) for e in SPEC["extra_pairs"]]
CFG = RunConfig()

#: Hand-asserted buckets, in question-set order, then the appended extra pairs.
EXPECTED_BUCKETS = [
    "chunking", "chunking", "extraction", "no-failure",
] + [e["known_bucket"] for e in SPEC["extra_pairs"]]


@pytest.fixture
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("network access attempted in an offline stage")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


def _pairs():
    return load_question_set(QUESTIONS) + EXTRA


@pytest.fixture(scope="module")
def batch_result():
    pytest.importorskip("sentence_transformers")
    return run_batch(PDF, CFG, _pairs())


def _single(pair):
    doc = PyMuPDFLoader().load(PDF)
    chunks = Chunker(CFG).chunk(doc)
    idx = LocalEmbeddingsIndexer(CFG)
    idx.index(chunks)
    ranking = Retriever(idx, CFG).rank(pair.question)
    return compose_verdict(build_engine().run(
        expected_answer=pair.expected_answer, doc=doc, chunks=chunks,
        ranking=ranking, config=CFG))


# --- offline ---------------------------------------------------------------


def test_manifest_and_question_set_agree(no_network):
    pairs = load_question_set(QUESTIONS)
    assert len(pairs) == len(SPEC["expected"])
    assert pairs[0].question.startswith("What are the five steps")
    assert sum(SPEC["expected_counts"].values()) == len(pairs) + len(EXTRA)
    assert set(SPEC["expected_counts"]) == set(BUCKETS)
    tally = {b: EXPECTED_BUCKETS.count(b) for b in BUCKETS}
    assert tally == SPEC["expected_counts"]


def test_extraction_and_chunking_offline(no_network):
    """The fixture questions' extraction/chunking facts, with no network."""
    pairs = load_question_set(QUESTIONS)
    doc = PyMuPDFLoader().load(PDF)
    chunks = Chunker(CFG).chunk(doc)

    split = locate(pairs[0].expected_answer, doc)
    assert split.is_pass and survive(split.evidence["span"], chunks).is_fail

    heading = locate(pairs[1].expected_answer, doc)
    assert heading.is_pass and survive(heading.evidence["span"], chunks).is_pass

    assert locate(pairs[2].expected_answer, doc).is_fail
    control = locate(pairs[3].expected_answer, doc)
    assert control.is_pass and survive(control.evidence["span"], chunks).is_pass


def test_blank_answer_is_not_a_parseable_entry(tmp_path, no_network):
    """Why the withheld pair is appended in code: the parser rejects it."""
    path = tmp_path / "q.json"
    path.write_text(json.dumps([{"question": "q?", "expected_answer": "  "}]), encoding="utf-8")
    with pytest.raises(ValueError):
        load_question_set(path)


# --- embedding-dependent ---------------------------------------------------


def test_expected_bucket_counts(batch_result):
    assert dict(batch_result.counts) == SPEC["expected_counts"]
    assert [r.bucket for r in batch_result.results] == EXPECTED_BUCKETS


def test_split_list_and_unembedded_heading_both_chunking(batch_result):
    split, heading = batch_result.results[0], batch_result.results[1]
    assert (split.bucket, split.verdict.reason_code) == ("chunking", "boundary_cut")
    assert (heading.bucket, heading.verdict.reason_code) == ("chunking", "missing_context")
    assert [e["known_cause"] for e in SPEC["expected"][:2]] == ["chunking", "chunking"]


def test_counts_sum_to_number_of_questions(batch_result):
    assert sum(batch_result.counts.values()) == batch_result.total == len(_pairs())


def test_forward_pass_built_once(monkeypatch):
    pytest.importorskip("sentence_transformers")
    built, indexed, loaded = [], [], []
    init, index = LocalEmbeddingsIndexer.__init__, LocalEmbeddingsIndexer.index
    load = PyMuPDFLoader.load

    def counting_init(self, *a, **k):
        built.append(1)
        init(self, *a, **k)

    def counting_index(self, *a, **k):
        indexed.append(1)
        return index(self, *a, **k)

    def counting_load(self, *a, **k):
        loaded.append(1)
        return load(self, *a, **k)

    monkeypatch.setattr(batch.LocalEmbeddingsIndexer, "__init__", counting_init)
    monkeypatch.setattr(batch.LocalEmbeddingsIndexer, "index", counting_index)
    monkeypatch.setattr(PyMuPDFLoader, "load", counting_load)
    pairs = _pairs()
    assert len(pairs) > 1
    run_batch(PDF, CFG, pairs)
    assert (len(built), len(indexed), len(loaded)) == (1, 1, 1)


@pytest.mark.parametrize("position", range(len(EXPECTED_BUCKETS)))
def test_batch_bucket_equals_single_question_verdict(batch_result, position):
    verdict = _single(_pairs()[position])
    item = batch_result.results[position]
    assert item.stage == verdict.stage
    assert item.reason == verdict.reason
    assert item.bucket == ("withheld" if verdict.withheld else (verdict.stage or "no-failure"))


def test_reproducible(batch_result):
    again = run_batch(PDF, CFG, _pairs())
    assert dict(again.counts) == dict(batch_result.counts)
    assert [r.bucket for r in again.results] == [r.bucket for r in batch_result.results]


def test_json_output_shape(monkeypatch, tmp_path, capfd, batch_result):
    monkeypatch.setattr(cli, "execute_batch", lambda pdf, config, pairs: batch_result)
    out_path = tmp_path / "out.json"
    assert cli.main(["batch", str(PDF), str(QUESTIONS), "--json", str(out_path)]) == 0
    capfd.readouterr()
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert set(data) == {"distribution", "questions"}
    assert data["distribution"] == SPEC["expected_counts"]
    assert len(data["questions"]) == batch_result.total
    for row, item in zip(data["questions"], batch_result.results):
        assert {"question", "bucket", "stage", "reason", "evidence"} <= set(row)
        assert row["question"] == item.question and row["bucket"] == item.bucket
        assert isinstance(row["evidence"], dict)
    first = data["questions"][0]
    assert first["stage"] == "chunking" and first["reason"] and first["evidence"]


def test_batch_cli_end_to_end(tmp_path, capfd):
    """Real run, nothing stubbed: the file's JSON has the expected buckets."""
    pytest.importorskip("sentence_transformers")
    out_path = tmp_path / "out.json"
    assert cli.main(["batch", str(PDF), str(QUESTIONS), "--json", str(out_path)]) == 0
    out = capfd.readouterr().out
    assert "Failure distribution (4 questions)" in out
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert [q["bucket"] for q in data["questions"]] == EXPECTED_BUCKETS[:4]


# --- diagnose / compare unchanged ------------------------------------------


def test_diagnose_and_compare_unchanged():
    pytest.importorskip("sentence_transformers")
    manifest = _MANIFEST["fixtures"]
    split = next(f for f in manifest if f["name"] == "split_list")
    control = next(f for f in manifest if f["name"] == "control")

    def diagnose(entry):
        return cli.run(cli.build_parser().parse_args(
            ["diagnose", str(FIXTURES / entry["pdf"]), entry["question"], entry["expected_answer"]]))

    report = diagnose(split)
    assert "VERDICT: chunking" in report and "loader: pymupdf" in report
    assert "VERDICT: chunking" not in diagnose(control)

    two = _MANIFEST["comparison_fixtures"][0]
    out = cli.run_compare(cli.build_parser().parse_args(
        ["compare", str(FIXTURES / two["pdf"]), two["question"], two["expected_answer"]]))
    assert "pymupdf" in out and "pypdf" in out and "pdfplumber" in out
