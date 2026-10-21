"""Batch mode: read a user-supplied question set.

The input is a JSON array of objects, each with a non-empty string ``question``
and a non-empty string ``expected_answer``. Order is preserved.

Rules this module keeps:

* **Ground truth is user-supplied.** Pairs are only read, never generated or
  inferred (``.ai/constitution.md``, ADR-002).
* **JSON only.** No CSV or JSONL.
* **Headless.** It imports neither the CLI nor any view layer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Union

__all__ = ["BatchInputError", "QuestionPair", "load_question_set"]


class BatchInputError(ValueError):
    """The question-set file is malformed."""


@dataclass(frozen=True)
class QuestionPair:
    """One asserted question and the answer the user says it should return."""

    question: str
    expected_answer: str


def load_question_set(path: Union[str, Path]) -> list[QuestionPair]:
    """Read and validate a JSON question set, preserving input order."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BatchInputError(f"cannot read question set {path}: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise BatchInputError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, list):
        raise BatchInputError(
            f"{path}: question set must be a JSON array, got {type(data).__name__}"
        )

    pairs: list[QuestionPair] = []
    for index, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise BatchInputError(
                f"entry {index}: must be an object, got {type(entry).__name__}"
            )
        values = {}
        for field in ("question", "expected_answer"):
            value = entry.get(field)
            if field not in entry:
                raise BatchInputError(f"entry {index}: missing '{field}'")
            if not isinstance(value, str):
                raise BatchInputError(
                    f"entry {index}: '{field}' must be a string, got {type(value).__name__}"
                )
            if not value.strip():
                raise BatchInputError(f"entry {index}: '{field}' is empty")
            values[field] = value
        pairs.append(QuestionPair(**values))
    return pairs
