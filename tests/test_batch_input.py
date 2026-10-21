import json

import pytest

from ragdoc.batch import BatchInputError, QuestionPair, load_question_set


def _write(tmp_path, content):
    p = tmp_path / "q.json"
    p.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")
    return p


def test_parses_in_order(tmp_path):
    data = [{"question": f"q{i}", "expected_answer": f"a{i}"} for i in (3, 1, 2)]
    assert load_question_set(_write(tmp_path, data)) == [
        QuestionPair("q3", "a3"), QuestionPair("q1", "a1"), QuestionPair("q2", "a2")
    ]


@pytest.mark.parametrize("content", ['{"question": "q"}', '"x"', "not json", "["])
def test_not_array_or_invalid_json(tmp_path, content):
    with pytest.raises(ValueError):
        load_question_set(_write(tmp_path, content))


@pytest.mark.parametrize("entry", [
    {"expected_answer": "a"}, {"question": "q"},
    {"question": "", "expected_answer": "a"}, {"question": "q", "expected_answer": "  "},
    {"question": 1, "expected_answer": "a"}, "str",
])
def test_bad_entry_names_index(tmp_path, entry):
    data = [{"question": "q", "expected_answer": "a"}, entry]
    with pytest.raises(BatchInputError, match="entry 1"):
        load_question_set(_write(tmp_path, data))
