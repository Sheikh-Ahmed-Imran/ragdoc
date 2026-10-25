"""The attribution engine runner: run the checks in order, stop correctly.

This is the component that makes ``ragdoc`` a diagnostic tool rather than a
pipeline (``.ai/architecture/components.md``). It reads the artefacts the
forward stages produced and runs the backward checks in order, enforcing the
two rules the constitution is strictest about:

* **Stop at the first failure.** The verdict is the first failing stage; every
  later check is reported as ``not_reached``, never as passing (spec AC#2).
* **Withhold, do not fall through.** If a check cannot be evaluated it returns
  ``unevaluated``; the engine stops and emits *no* failure stage, rather than
  letting the diagnosis fall through to the next plausible stage
  (``.ai/constitution.md`` — "a verdict is never produced without evidence from
  every stage it rules out"; spec AC#3).

The engine depends only on the artefact shapes (``ragdoc.artifacts`` /
``ragdoc.config``) and the checks in :mod:`ragdoc.attribution.checks`. It
imports nothing from the CLI or any view layer (``.ai/constitution.md`` — the
headless engine never depends on the UI).

Scope note (T-004): this runner wires the two offline checks, ``locate`` and
``survive``. The ``rank`` check and the final verdict *composition* (including
how a context-missing chunking cause that only manifests as a bad rank is
attributed to chunking) are T-006. The seam for that is deliberate and small:
:attr:`AttributionEngine` takes an injectable list of :class:`Check`\\ s, and
:meth:`AttributionEngine.run` already accepts the ``ranking`` and ``config``
that ``rank`` will need. This module defines no retrieval logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from ragdoc.artifacts import CheckResult, Chunk, ExtractedDoc, Ranking
from ragdoc.attribution import checks
from ragdoc.config import RunConfig

__all__ = [
    "RunState",
    "Check",
    "CheckOutcome",
    "EngineResult",
    "AttributionEngine",
    "DEFAULT_CHECKS",
]


@dataclass
class RunState:
    """The artefacts and accumulated results one engine run has to work with.

    A :class:`Check` reads whatever it needs from here, including the results of
    earlier checks (``survive`` reads the span ``locate`` placed in its
    evidence). ``ranking`` and ``config`` are present for the ``rank`` check
    T-006 adds; the offline checks ignore them.
    """

    expected_answer: str
    doc: ExtractedDoc
    chunks: Sequence[Chunk]
    ranking: Optional[Ranking]
    config: Optional[RunConfig]
    results: dict[str, CheckResult]


CheckFn = Callable[[RunState], CheckResult]


@dataclass(frozen=True)
class Check:
    """One attribution check and the pipeline stage it rules on.

    ``name`` is the check's own name (``"locate"``, ``"survive"``, ``"rank"``);
    ``stage`` is the stage a failure of it attributes to (``"extraction"``,
    ``"chunking"``, ``"retrieval"``). The engine uses ``stage`` to report the
    first failing stage; the reporter (T-007) renders the stage table.
    """

    name: str
    stage: str
    run: CheckFn


@dataclass(frozen=True)
class CheckOutcome:
    """A check's name, the stage it rules on, and the result it produced."""

    check: str
    stage: str
    result: CheckResult


@dataclass(frozen=True)
class EngineResult:
    """The outcome of running the attribution checks over one pipeline run.

    Carries every check's :class:`~ragdoc.artifacts.CheckResult` in order, plus
    the two mechanical signals the verdict composition (T-006) builds on:

    * :attr:`first_failure_stage` / :attr:`first_failure_check` — the stage and
      check of the first ``fail``, or ``None`` if nothing failed. This is the
      *raw* first failing stage; T-006's verdict rules may re-attribute it (e.g.
      a rank failure with an upstream ``context_missing`` flag attributes to
      chunking, not retrieval).
    * :attr:`withheld` — ``True`` if the run stopped because a check could not be
      evaluated (``unevaluated``). When withheld, :attr:`first_failure_stage` is
      ``None``: no verdict is emitted for or past the unevaluated check.
    """

    outcomes: tuple[CheckOutcome, ...]
    first_failure_stage: Optional[str]
    first_failure_check: Optional[str]
    withheld: bool

    def result_for_check(self, name: str) -> Optional[CheckResult]:
        """Return the result of the named check, or ``None`` if it was not run."""
        for outcome in self.outcomes:
            if outcome.check == name:
                return outcome.result
        return None

    def result_for_stage(self, stage: str) -> Optional[CheckResult]:
        """Return the result for the named stage, or ``None`` if absent."""
        for outcome in self.outcomes:
            if outcome.stage == stage:
                return outcome.result
        return None

    @property
    def has_failure(self) -> bool:
        """True if some check failed (a stage can be named as the verdict)."""
        return self.first_failure_stage is not None


# --------------------------------------------------------------------------- #
# The default (v0) check sequence: the two offline checks, in order.
# --------------------------------------------------------------------------- #

def _run_locate(state: RunState) -> CheckResult:
    return checks.locate(state.expected_answer, state.doc)


def _run_survive(state: RunState) -> CheckResult:
    located = state.results.get("locate")
    if located is None or not located.is_pass:
        # The engine only runs a check after every earlier one passed, so this
        # is defensive; if it ever trips, withhold rather than guess a span.
        return CheckResult.unevaluated(
            "survive needs a located span from a passing locate"
        )
    spans = located.evidence.get("spans")
    if not spans:
        single = located.evidence.get("span")
        spans = (single,) if single is not None else ()
    if not spans:
        return CheckResult.unevaluated(
            "locate passed without a span in its evidence"
        )
    return checks.survive(spans, state.chunks)


#: The checks T-004 wires, in run order. T-006 appends a ``rank`` check for the
#: ``retrieval`` stage; the engine runs whatever sequence it is given.
DEFAULT_CHECKS: tuple[Check, ...] = (
    Check(name="locate", stage="extraction", run=_run_locate),
    Check(name="survive", stage="chunking", run=_run_survive),
)


def _retarget_outcomes(
    outcomes: list[CheckOutcome], doc: ExtractedDoc
) -> list[CheckOutcome]:
    """Point ``locate``/``survive`` evidence at the copy ``rank`` judged.

    ``rank`` picks the best-ranking chunk among ``survive``'s candidates. The
    reported span/page/snippet, ``context_missing`` flag and target chunk must
    describe that same copy, so the two earlier results are rewritten for it.
    With no ``rank`` result (the offline sequence) the earliest copy stands.
    """
    by_check = {o.check: o for o in outcomes}
    locate_o, survive_o, rank_o = (by_check.get(n) for n in ("locate", "survive", "rank"))
    if locate_o is None or survive_o is None or rank_o is None:
        return outcomes
    chosen = rank_o.result.evidence.get("target_chunk_id")
    if chosen is None:
        return outcomes
    new_locate, new_survive = checks.retarget(
        locate_o.result, survive_o.result, chosen, doc
    )
    swap = {
        "locate": CheckOutcome(locate_o.check, locate_o.stage, new_locate),
        "survive": CheckOutcome(survive_o.check, survive_o.stage, new_survive),
    }
    return [swap.get(o.check, o) for o in outcomes]


class AttributionEngine:
    """Runs attribution checks in order, stopping correctly at the first break.

    The sequence is injectable so T-006 can extend it with the ``rank`` check
    without changing this runner. With no argument it runs :data:`DEFAULT_CHECKS`
    (``locate`` then ``survive``).
    """

    def __init__(self, check_sequence: Optional[Sequence[Check]] = None) -> None:
        self._checks: tuple[Check, ...] = tuple(
            check_sequence if check_sequence is not None else DEFAULT_CHECKS
        )

    def run(
        self,
        *,
        expected_answer: str,
        doc: ExtractedDoc,
        chunks: Sequence[Chunk],
        ranking: Optional[Ranking] = None,
        config: Optional[RunConfig] = None,
    ) -> EngineResult:
        """Run the checks over one pipeline run and return their outcomes.

        Each check runs only if every earlier check passed. The first ``fail``
        is the verdict stage; the first ``unevaluated`` stops the run and
        withholds the verdict. In both cases every later check is recorded as
        ``not_reached`` — never as passing.

        ``ranking`` and ``config`` are threaded through for the ``rank`` check
        (T-006); the default offline checks do not read them.
        """
        state = RunState(
            expected_answer=expected_answer,
            doc=doc,
            chunks=chunks,
            ranking=ranking,
            config=config,
            results={},
        )

        outcomes: list[CheckOutcome] = []
        stopped = False
        first_failure_stage: Optional[str] = None
        first_failure_check: Optional[str] = None
        withheld = False

        for check in self._checks:
            if stopped:
                result = CheckResult.not_reached()
            else:
                result = check.run(state)
            state.results[check.name] = result
            outcomes.append(CheckOutcome(check.name, check.stage, result))

            if not stopped:
                if result.is_fail:
                    stopped = True
                    first_failure_stage = check.stage
                    first_failure_check = check.name
                elif result.is_unevaluated:
                    stopped = True
                    withheld = True

        outcomes = _retarget_outcomes(outcomes, doc)

        return EngineResult(
            outcomes=tuple(outcomes),
            first_failure_stage=first_failure_stage,
            first_failure_check=first_failure_check,
            withheld=withheld,
        )
