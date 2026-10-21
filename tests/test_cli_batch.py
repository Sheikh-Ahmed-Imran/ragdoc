"""CLI wiring for ``ragdoc batch`` (the batch components are stubbed)."""

import json

from ragdoc import cli
from ragdoc.batch import BatchResult, QuestionResult


def _result():
    item = QuestionResult("Q?", "A", "retrieval", "retrieval", "ranked low",
                          {"position": 7, "total": 20, "top_k": 5}, None)
    return BatchResult(results=(item,), counts={"retrieval": 1})


def _stub(monkeypatch, seen):
    def fake_run(pdf, config, pairs):
        seen.update(pdf=pdf, config=config, pairs=pairs)
        return _result()
    monkeypatch.setattr(cli, "execute_batch", fake_run)


def _questions(tmp_path):
    path = tmp_path / "q.json"
    path.write_text(json.dumps([{"question": "Q?", "expected_answer": "A"}]), encoding="utf-8")
    return str(path)


def test_batch_prints_report_and_builds_config(monkeypatch, tmp_path, capfd):
    seen = {}
    _stub(monkeypatch, seen)
    code = cli.main(["batch", "doc.pdf", _questions(tmp_path), "--top-k", "3"])
    out = capfd.readouterr().out
    assert code == 0
    assert "Failure distribution (1 question)" in out and "[retrieval] Q?" in out
    assert seen["config"].top_k == 3 and seen["pdf"] == "doc.pdf"


def test_batch_json_file(monkeypatch, tmp_path, capfd):
    _stub(monkeypatch, {})
    out_path = tmp_path / "out.json"
    code = cli.main(["batch", "doc.pdf", _questions(tmp_path), "--json", str(out_path)])
    capfd.readouterr()
    assert code == 0
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["distribution"]["retrieval"] == 1 and data["questions"][0]["question"] == "Q?"


def test_batch_bad_question_set_exits_2(tmp_path, capfd):
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    assert cli.main(["batch", "doc.pdf", str(bad)]) == 2
    assert "ragdoc: error:" in capfd.readouterr().err


def test_no_subcommand_exits_2(capfd):
    assert cli.main([]) == 2
