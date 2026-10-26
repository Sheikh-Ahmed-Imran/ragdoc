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


def _spy_batch(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "execute_batch", lambda *a: calls.append(a) or _result())
    monkeypatch.setattr(cli, "load_question_set", lambda *a: calls.append(a) or [])
    return calls


def _assert_clean_failure(code, capfd, calls, needle):
    err = capfd.readouterr().err
    assert code == 2
    assert err.startswith("ragdoc: error:") and needle in err
    assert "Traceback" not in err
    assert calls == []


def test_batch_json_directory_fails_before_forward_pass(monkeypatch, tmp_path, capfd):
    calls = _spy_batch(monkeypatch)
    code = cli.main(["batch", "doc.pdf", _questions(tmp_path), "--json", str(tmp_path)])
    _assert_clean_failure(code, capfd, calls, str(tmp_path))


def test_batch_json_missing_parent_fails_and_creates_nothing(monkeypatch, tmp_path, capfd):
    calls = _spy_batch(monkeypatch)
    target = tmp_path / "nope" / "out.json"
    code = cli.main(["batch", "doc.pdf", _questions(tmp_path), "--json", str(target)])
    _assert_clean_failure(code, capfd, calls, str(target))
    assert not (tmp_path / "nope").exists()


def test_batch_json_not_writable_fails_before_forward_pass(monkeypatch, tmp_path, capfd):
    calls = _spy_batch(monkeypatch)
    monkeypatch.setattr(cli.os, "access", lambda *a: False)
    target = tmp_path / "out.json"
    code = cli.main(["batch", "doc.pdf", _questions(tmp_path), "--json", str(target)])
    _assert_clean_failure(code, capfd, calls, str(target))


def test_batch_json_residual_oserror_is_clean(monkeypatch, tmp_path, capfd):
    _stub(monkeypatch, {})
    monkeypatch.setattr(cli, "_check_output_path", lambda path: None)
    code = cli.main(["batch", "doc.pdf", _questions(tmp_path), "--json", str(tmp_path)])
    err = capfd.readouterr().err
    assert code == 2 and err.startswith("ragdoc: error:") and "Traceback" not in err


def test_batch_bad_question_set_exits_2(tmp_path, capfd):
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    assert cli.main(["batch", "doc.pdf", str(bad)]) == 2
    assert "ragdoc: error:" in capfd.readouterr().err


def test_no_subcommand_exits_2(capfd):
    assert cli.main([]) == 2
