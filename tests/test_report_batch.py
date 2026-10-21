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
    assert "MRR" not in out


def test_json_shape_and_sum():
    data = batch_to_json(_batch())
    assert sum(data["distribution"].values()) == len(data["questions"]) == 4
    assert set(data["questions"][0]) == {"question", "bucket", "stage", "reason", "evidence"}
    json.dumps(data)
