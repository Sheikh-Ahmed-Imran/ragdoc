"""``RunConfig`` validation: a non-positive ``top_k`` fails fast, cleanly."""

import dataclasses
import json

import pytest

from ragdoc import cli
from ragdoc.config import RunConfig


@pytest.mark.parametrize("bad", [0, -3])
def test_non_positive_top_k_raises_naming_top_k(bad):
    with pytest.raises(ValueError, match=rf"top_k.*{bad}"):
        RunConfig(top_k=bad)


def test_valid_top_k_unchanged_and_still_frozen():
    config = RunConfig(top_k=1)
    assert config.top_k == 1
    assert RunConfig().top_k == 10
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.top_k = 5


def test_replace_revalidates():
    with pytest.raises(ValueError, match="top_k"):
        dataclasses.replace(RunConfig(), top_k=0)


def _questions(tmp_path):
    path = tmp_path / "q.json"
    path.write_text(json.dumps([{"question": "Q?", "expected_answer": "A"}]), encoding="utf-8")
    return str(path)


def _assert_clean_exit(code, captured):
    assert code == 2
    assert captured.out == ""
    lines = captured.err.strip().splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("ragdoc: error: ")
    assert "top_k" in lines[0]
    assert "Traceback" not in captured.err


def test_cli_diagnose_top_k_zero(capfd):
    code = cli.main(["diagnose", "d.pdf", "q", "a", "--top-k", "0"])
    _assert_clean_exit(code, capfd.readouterr())


def test_cli_batch_top_k_zero(tmp_path, capfd):
    code = cli.main(["batch", "d.pdf", _questions(tmp_path), "--top-k", "0"])
    _assert_clean_exit(code, capfd.readouterr())


def test_cli_sweep_top_k_zero(tmp_path, capfd):
    code = cli.main(["sweep", "d.pdf", _questions(tmp_path), "--top-k", "0,5"])
    _assert_clean_exit(code, capfd.readouterr())
