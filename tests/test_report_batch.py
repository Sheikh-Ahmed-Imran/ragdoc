import json

from ragdoc.batch import BUCKETS, BatchResult, QuestionResult
from ragdoc.report.batch import batch_to_json, render_batch_report


def _q(q, bucket, stage=None, reason=None, **ev):
    return QuestionResult(q, "a", bucket, stage, reason, ev, None)


def _batch():
    rs = (
        _q("q1", "retrieval", "retrieval", "ranked low", position=3, total=10, top_k=5, page=2),
        _q("q2", "no-failure"),
        _q("q3", "withheld"),
        _q("q4", "retrieval", "retrieval", "r"),
    )
    counts = {b: 0 for b in BUCKETS}
    for r in rs:
        counts[r.bucket] += 1
    return BatchResult(results=rs, counts=counts)


def test_render_distribution_and_list():
    out = render_batch_report(_batch())
    assert "retrieval" in out and "50.0%" in out
    assert out.index("Failure distribution") < out.index("Questions:")
    assert out.index("q1") < out.index("q2") < out.index("q3")
    assert "precision@5 = 0.20 (max 0.20)" in out
    assert out.index("Failure distribution") < out.index("Retrieval metrics") < out.index("Questions:")


def test_json_shape_and_sum():
    data = batch_to_json(_batch())
    assert sum(data["distribution"].values()) == len(data["questions"]) == 4
    assert set(data["questions"][0]) == {"question", "bucket", "stage", "reason", "evidence"}
    json.dumps(data)


def _metrics_batch():
    rs = (
        _q("a", "retrieval", "retrieval", "r", position=1, total=10, top_k=5),
        _q("b", "retrieval", "retrieval", "r", position=2, total=10, top_k=5),
        _q("c", "retrieval", "retrieval", "r", position=4, total=10, top_k=5),
        _q("d", "retrieval", "retrieval", "r", position=10, total=10, top_k=5),
        _q("e", "extraction", "extraction", "lost"),
    )
    counts = {b: 0 for b in BUCKETS}
    for r in rs:
        counts[r.bucket] += 1
    return BatchResult(results=rs, counts=counts)


def test_aggregate_block_values_and_denominator():
    out = render_batch_report(_metrics_batch())
    # MRR = (1 + .5 + .25 + .1) / 4 = 0.4625; hit@5 = 3/4; median 3; p90 inclusive = 8.2
    assert "over 4 of 5 questions that reached retrieval" in out
    assert "MRR = 0.46" in out
    assert "hit-rate@5 = 0.75" in out
    assert "rank median = 3" in out
    assert "p90 = 8.2" in out
    block = next(ln for ln in out.splitlines() if ln.startswith("Retrieval metrics"))
    assert "(max" not in block


def test_aggregate_block_when_nothing_reached_retrieval():
    rs = (_q("x", "extraction", "extraction", "lost"), _q("y", "no-failure"))
    counts = {b: 0 for b in BUCKETS}
    for r in rs:
        counts[r.bucket] += 1
    batch = BatchResult(results=rs, counts=counts)
    out = render_batch_report(batch)
    assert "no questions reached retrieval" in out
    m = batch_to_json(batch)["retrieval_metrics"]
    assert m["reached"] == 0 and m["total"] == 2 and m["mrr"] is None


def test_json_retrieval_metrics():
    m = batch_to_json(_metrics_batch())["retrieval_metrics"]
    assert m["reached"] == 4 and m["total"] == 5 and m["top_k"] == 5
    assert abs(m["mrr"] - 0.4625) < 1e-9
    assert m["hit_rate"] == 0.75
    assert m["rank_median"] == 3
    assert abs(m["rank_p90"] - 8.2) < 1e-9
    json.dumps(m)
