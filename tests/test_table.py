"""Unit tests for the shared terminal table helper."""

from __future__ import annotations

import pytest

from ragdoc.report._table import LEFT, RIGHT, render_table


def test_header_rule_rows_and_alignment():
    lines = render_table(["name", "n"], [["ab", "5"], ["abcdef", "12345"]], [LEFT, RIGHT])
    assert lines == [
        "name        n",
        "------  -----",
        "ab          5",
        "abcdef  12345",
    ]


def test_width_is_max_of_header_and_cells():
    lines = render_table(["longheader"], [["x"]], [LEFT])
    assert lines == ["longheader", "----------", "x"]


def test_trailing_spaces_stripped():
    lines = render_table(["a", "b"], [["x", ""]], [LEFT, LEFT])
    assert all(line == line.rstrip() for line in lines)


def test_symbol_cells_align_in_either_mode():
    for found in ("✓ found", "[OK] found"):
        lines = render_table(["r", "n"], [[found, "1"], ["x", "22"]], [LEFT, RIGHT])
        assert len({len(line) for line in lines}) == 1


def test_no_rows_still_has_header_and_rule():
    assert render_table(["a", "bb"], [], [LEFT, RIGHT]) == ["a  bb", "-  --"]


def test_deterministic_and_validates_shape():
    args = (["a"], [["x"]], [LEFT])
    assert render_table(*args) == render_table(*args)
    with pytest.raises(ValueError):
        render_table(["a", "b"], [["x"]], [LEFT, LEFT])
    with pytest.raises(ValueError):
        render_table(["a", "b"], [], [LEFT])
