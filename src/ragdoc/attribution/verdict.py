"""Compose the three check results into one verdict naming a single stage.

This is the module that turns ``locate`` / ``survive`` / ``rank`` outcomes into
the product's actual output: a verdict naming *one* stage, with a fix tied to
the specific reason it failed (``.ai/product/product.md``). The checks each
report on their own stage; the rules for collapsing them into one attribution —
including the case where a *chunking* root cause only shows up as a bad
*retrieval* rank — live here and nowhere else.

Three things this module is careful about, each from a project decision:

* **A verdict is never produced without evidence from every stage it rules out**
  (``.ai/constitution.md``). The engine runs the checks in order and withholds
  the verdict if any check could not be evaluated; this module preserves that —
  when a stage is ``unevaluated`` there is no verdict, and later stages read
  ``not_reached``, never "pass".
* **The fix attaches to the failure reason, not the stage** (``.ai/architecture/
  system.md``). The same stage fails in distinguishable ways — chunking fails by
  a boundary cut *or* by an unembedded heading — so the fix is keyed off a reason
  code (:data:`FIXES`), not off the stage name.
* **The unembedded-heading case resolves to chunking even though its symptom is a
  poor retrieval rank** (``.ai/domain/domain.md``; ``.ai/product/product.md``).
  When ``survive`` passed but flagged ``context_missing`` *and* ``rank`` then
  failed, the verdict is **chunking**, attributed upstream to the missing
  context — while retrieval is still shown failing, because that is the symptom
  the user observed. This is the clearest domain case of a failure that looks
  like retrieval and is not.

It imports only the engine result type and the artefact shapes; nothing from the
CLI or any view layer (``.ai/constitution.md`` — the headless engine never
depends on the UI). Rendering the :class:`Verdict` for a terminal is the
reporter's job (T-007), not this module's.

Wiring note: the engine's ``DEFAULT_CHECKS`` are the two offline checks; the full
v0 sequence (adding ``rank``) is assembled here as :data:`V0_CHECKS`, using the
engine's injectable-sequence seam so the runner itself stays untouched. A caller
builds the complete engine with :func:`build_engine` (or
``AttributionEngine(V0_CHECKS)``), runs it, and passes the :class:`EngineResult`
to :func:`compose_verdict`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ragdoc.artifacts import CheckResult, CheckStatus
from ragdoc.attribution import checks
from ragdoc.attribution.engine import (
    DEFAULT_CHECKS,
    AttributionEngine,
    Check,
    EngineResult,
    RunState,
)
from ragdoc.config import RunConfig

__all__ = [
    "EXTRACTION_LOST",
    "BOUNDARY_CUT",
    "MISSING_CONTEXT",
    "RANK_BELOW_TOP_K",
    "FIXES",
    "StageReport",
    "Verdict",
    "compose_verdict",
    "RANK_CHECK",
    "V0_CHECKS",
    "build_engine",
]


# --------------------------------------------------------------------------- #
# Reason codes and the fixes tied to them.
#
# The fix attaches to the *reason*, not the stage (``.ai/architecture/system.md``):
# chunking fails two distinguishable ways (a boundary cut vs. an unembedded
# heading) and each wants a different fix, so the key is the reason, not the
# stage name.
# --------------------------------------------------------------------------- #

#: The loader did not preserve the answer text (``locate`` failed).
EXTRACTION_LOST = "extraction_lost"
#: A chunk boundary cut the answer in half (``survive`` hard-failed).
BOUNDARY_CUT = "boundary_cut"
#: The answer's chunk lacks the heading/section context that would make it
#: retrievable, and it consequently ranked below ``top_k`` (``survive`` passed
#: with ``context_missing`` *and* ``rank`` failed). A chunking cause whose
#: symptom is a retrieval rank.
MISSING_CONTEXT = "missing_context"
#: The target chunk ranked below ``top_k`` with no upstream context flag
#: (``rank`` failed, ``survive`` clean). A genuine retrieval failure.
RANK_BELOW_TOP_K = "rank_below_top_k"

#: Reason code → fix suggestion. The tool names the stage and suggests a fix; it
#: never applies one (spec Exclusions).
FIXES: dict[str, str] = {
    EXTRACTION_LOST: (
        "the loader did not preserve the answer text — try a different loader, "
        "or confirm the PDF has a real text layer (a scanned/image-only page "
        "extracts nothing to find)"
    ),
    BOUNDARY_CUT: (
        "increase chunk size (and overlap) so the answer is no longer split "
        "across a chunk boundary"
    ),
    MISSING_CONTEXT: "prepend section path to chunk text before embedding",
    RANK_BELOW_TOP_K: (
        "raise top_k, or revisit the embedding model / chunking so the target "
        "chunk ranks inside top_k"
    ),
}


# --------------------------------------------------------------------------- #
# The composed verdict.
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class StageReport:
    """One stage's composed status plus the check result that produced it.

    :attr:`status` is the status to *display* for the stage, which may differ
    from ``result.status`` after re-attribution: in the unembedded-heading case
    chunking is displayed ``FAIL`` even though ``survive`` returned ``PASS``
    (with ``context_missing``), and retrieval is also displayed ``FAIL`` as the
    observed symptom. :attr:`result` is the raw check result, kept so the
    reporter can render the evidence (page, span, rank, section path) regardless
    of the displayed status. ``generation`` carries ``result=None`` — there is no
    generation check in v0 and it is always ``not_reached`` (spec Exclusions).
    """

    stage: str
    status: CheckStatus
    result: Optional[CheckResult] = None


@dataclass(frozen=True)
class Verdict:
    """The single attribution a run produced, with the per-stage picture behind it.

    Attributes:
        stage: the stage the verdict names (``"extraction"`` / ``"chunking"`` /
            ``"retrieval"``), or ``None`` when no stage failed or the verdict was
            withheld.
        reason_code: a stable reason code (see the module constants) a reporter or
            test can branch on without matching prose, or ``None``.
        reason: the human-readable failure reason, or ``None``.
        fix: the suggested fix tied to :attr:`reason_code`, or ``None``.
        stages: the four stages in pipeline order (extraction, chunking,
            retrieval, generation) as :class:`StageReport`\\ s.
        withheld: ``True`` when a stage could not be evaluated and the verdict was
            withheld (:attr:`stage` is then ``None``) rather than falling through
            to the next plausible stage.
    """

    stage: Optional[str]
    reason_code: Optional[str]
    reason: Optional[str]
    fix: Optional[str]
    stages: tuple[StageReport, ...]
    withheld: bool = False

    @property
    def has_failure(self) -> bool:
        """True when the verdict names a failing stage."""
        return self.stage is not None


def _status_of(result: Optional[CheckResult]) -> CheckStatus:
    """The status to display for a stage from its (maybe absent) check result.

    A check that was never part of the run (``result is None``) is reported
    ``not_reached``, matching how the engine records a check the chain never
    reached.
    """
    return result.status if result is not None else CheckStatus.NOT_REACHED


def _missing_context_reason(survive_result: CheckResult) -> str:
    """Phrase the unembedded-heading reason from ``survive``'s evidence."""
    section_path = survive_result.evidence.get("section_path") or ()
    if section_path:
        heading = " / ".join(section_path)
        return (
            f"the answer's chunk sits under {heading!r} but that heading is not "
            "embedded in the chunk text (kept as metadata), so it cannot help the "
            "chunk rank for the question"
        )
    return (
        "the answer's chunk lacks the section/heading context that would make it "
        "retrievable, so it ranked below top_k"
    )


def compose_verdict(result: EngineResult) -> Verdict:
    """Collapse an :class:`EngineResult` into one :class:`Verdict`.

    Applies the four composition rules (``.ai/product/product.md``; plan step 6):

    1. ``locate`` fails → **extraction**; chunking, retrieval, generation
       ``not_reached``.
    2. ``survive`` hard-fails (boundary cut) → **chunking**; retrieval and
       generation ``not_reached``.
    3. ``survive`` passed-with-``context_missing`` *and* ``rank`` fails →
       **chunking**, attributed upstream to the missing context; retrieval is
       also shown failing (the observed symptom), generation ``not_reached``.
    4. ``rank`` fails with no upstream ``context_missing`` flag → **retrieval**.

    When any check was ``unevaluated`` the verdict is withheld (``stage is None``,
    ``withheld is True``) and later stages read ``not_reached`` — the engine
    already stopped the chain; this only names it. When every check passed there
    is no failing stage and no fix.

    The returned :class:`Verdict` always carries all four stages in pipeline
    order (generation last, always ``not_reached``).
    """
    locate_r = result.result_for_check("locate")
    survive_r = result.result_for_check("survive")
    rank_r = result.result_for_check("rank")

    # Start from the raw per-stage statuses; re-attribution below may override
    # the chunking status (the unembedded-heading case displays it as FAIL).
    extraction_status = _status_of(locate_r)
    chunking_status = _status_of(survive_r)
    retrieval_status = _status_of(rank_r)

    stage: Optional[str] = None
    reason_code: Optional[str] = None
    reason: Optional[str] = None
    withheld = False

    if locate_r is not None and locate_r.is_fail:
        # Rule 1 — extraction lost the answer.
        stage, reason_code = "extraction", EXTRACTION_LOST
        reason = locate_r.reason
    elif locate_r is not None and locate_r.is_unevaluated:
        withheld = True
    elif locate_r is not None and locate_r.is_pass:
        if survive_r is not None and survive_r.is_fail:
            # Rule 2 — a chunk boundary cut the answer.
            stage, reason_code = "chunking", BOUNDARY_CUT
            reason = survive_r.reason
        elif survive_r is not None and survive_r.is_unevaluated:
            withheld = True
        elif survive_r is not None and survive_r.is_pass:
            context_missing = bool(survive_r.evidence.get("context_missing"))
            if rank_r is not None and rank_r.is_fail:
                if context_missing:
                    # Rule 3 — chunking root cause, retrieval symptom. The verdict
                    # is chunking; retrieval stays FAIL (the observed symptom) and
                    # chunking is re-attributed to FAIL for display.
                    stage, reason_code = "chunking", MISSING_CONTEXT
                    reason = _missing_context_reason(survive_r)
                    chunking_status = CheckStatus.FAIL
                else:
                    # Rule 4 — a genuine retrieval failure.
                    stage, reason_code = "retrieval", RANK_BELOW_TOP_K
                    reason = rank_r.reason
            elif rank_r is not None and rank_r.is_unevaluated:
                withheld = True
            # rank passed (or was absent) → no failing stage.
    # locate absent → defensive: no verdict, everything stays not_reached.

    fix = FIXES.get(reason_code) if reason_code is not None else None

    stages = (
        StageReport("extraction", extraction_status, locate_r),
        StageReport("chunking", chunking_status, survive_r),
        StageReport("retrieval", retrieval_status, rank_r),
        StageReport("generation", CheckStatus.NOT_REACHED, None),
    )

    return Verdict(
        stage=stage,
        reason_code=reason_code,
        reason=reason,
        fix=fix,
        stages=stages,
        withheld=withheld,
    )


# --------------------------------------------------------------------------- #
# Wiring the ``rank`` check into the engine's check sequence.
#
# The engine runs whatever ordered sequence it is given; ``DEFAULT_CHECKS`` are
# the two offline checks (T-004). The full v0 sequence adds ``rank`` for the
# retrieval stage, assembled here via that seam so the runner stays untouched.
# ``rank`` needs the target chunk id ``survive`` placed in its evidence, the full
# ranking, and ``top_k`` from the config.
# --------------------------------------------------------------------------- #

def _run_rank(state: RunState) -> CheckResult:
    """Adapt :func:`checks.rank` to the engine's ``RunState``.

    Reads the target chunk id from ``survive``'s evidence, the full ranking from
    the run state, and ``top_k`` from the config. Withholds (``unevaluated``)
    rather than guessing when any of those is missing — a retrieval verdict
    without a ranking would be an attribution with no evidence behind it
    (``.ai/constitution.md``).
    """
    survived = state.results.get("survive")
    if survived is None or not survived.is_pass:
        # The engine only runs a check after every earlier one passed, so this is
        # defensive; if it ever trips, withhold rather than guess a target.
        return CheckResult.unevaluated(
            "rank needs a target chunk from a passing survive"
        )

    # Every chunk holding a surviving copy; rank measures the best-ranking one.
    target_id = survived.evidence.get("target_chunk_ids") or survived.evidence.get(
        "target_chunk_id"
    )
    if not target_id:
        return CheckResult.unevaluated(
            "survive passed without a target chunk id in its evidence"
        )

    if state.ranking is None:
        return CheckResult.unevaluated(
            "no ranking available; the retrieval stage cannot be evaluated"
        )

    top_k = (state.config or RunConfig()).top_k
    return checks.rank(target_id, state.ranking, top_k)


#: The ``rank`` check and the stage a failure of it attributes to (retrieval).
RANK_CHECK: Check = Check(name="rank", stage="retrieval", run=_run_rank)

#: The complete v0 check sequence: the two offline checks, then ``rank``.
V0_CHECKS: tuple[Check, ...] = (*DEFAULT_CHECKS, RANK_CHECK)


def build_engine() -> AttributionEngine:
    """An :class:`AttributionEngine` wired with the full v0 sequence (+``rank``)."""
    return AttributionEngine(V0_CHECKS)
