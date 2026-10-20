"""Reporter rendering tests (T-007).

Pure standard-library: the reporter renders a :class:`Verdict` that is built here
by hand, so these tests import nothing from the loader/indexer/embedding stack and
run without a network or the local model.
"""

from __future__ import annotations

from ragdoc.artifacts import CheckResult, CheckStatus
from ragdoc.attribution.verdict import (
    MISSING_CONTEXT,
    FIXES,
    StageReport,
    Verdict,
)
from ragdoc.config import RunConfig
from ragdoc.report import format_metric, precision_at_k, render_report


def _missing_context_verdict() -> Verdict:
    """A chunking (unembedded-heading) verdict whose symptom is a bad rank.

    Chunking is displayed FAIL (re-attributed), retrieval is also FAIL (the
    observed symptom) and carries the absolute rank in its evidence.
    """
    survive_r = CheckResult.passed(
        {"context_missing": True, "section_path": ("2. Methods", "2.1 Sampling")}
    )
    rank_r = CheckResult.failed(
        "target chunk ranked below top_k",
        {"position": 23, "total": 400, "top_k": 5},
    )
    stages = (
        StageReport("extraction", CheckStatus.PASS, CheckResult.passed()),
        StageReport("chunking", CheckStatus.FAIL, survive_r),
        StageReport("retrieval", CheckStatus.FAIL, rank_r),
        StageReport("generation", CheckStatus.NOT_REACHED, None),
    )
    return Verdict(
        stage="chunking",
        reason_code=MISSING_CONTEXT,
        reason="the answer's chunk sits under a heading that is not embedded",
        fix=FIXES[MISSING_CONTEXT],
        stages=stages,
    )


def test_renders_four_rows_verdict_and_fix():
    out = render_report(_missing_context_verdict(), RunConfig(top_k=5))
    for stage in ("extraction", "chunking", "retrieval", "generation"):
        assert stage in out
    assert "VERDICT: chunking" in out
    assert "FIX:" in out
    assert FIXES[MISSING_CONTEXT] in out


def test_generation_always_not_reached():
    out = render_report(_missing_context_verdict(), RunConfig())
    gen_line = next(ln for ln in out.splitlines() if ln.strip().startswith("generation"))
    assert "— not reached" in gen_line


def test_config_block_includes_prepend_section_context():
    out = render_report(_missing_context_verdict(), RunConfig(prepend_section_context=True))
    assert "Run configuration:" in out
    for field in ("loader:", "chunk size:", "overlap:", "embedding model:", "top_k:"):
        assert field in out
    assert "prepend_section_context: True" in out


def test_prepend_toggle_distinguishes_otherwise_identical_configs():
    verdict = _missing_context_verdict()
    off = render_report(verdict, RunConfig(prepend_section_context=False))
    on = render_report(verdict, RunConfig(prepend_section_context=True))
    assert off != on
    assert "prepend_section_context: False" in off
    assert "prepend_section_context: True" in on


def test_precision_renders_with_ceiling():
    out = render_report(_missing_context_verdict(), RunConfig(top_k=5))
    assert "precision@5 = 0.00 (max 0.20)" in out
    assert format_metric("precision@5", 0.0, 0.2) == "precision@5 = 0.00 (max 0.20)"


def test_retrieval_position_is_absolute_rank():
    out = render_report(_missing_context_verdict(), RunConfig(top_k=5))
    assert "ranked 23 of 400" in out


def test_precision_at_k_ceiling_is_one_over_k():
    inside, ceiling = precision_at_k(3, 5)
    assert ceiling == 0.2
    assert inside == 0.2
    outside, ceiling2 = precision_at_k(23, 5)
    assert outside == 0.0
    assert ceiling2 == 0.2
