"""An embedding model that cannot be loaded fails cleanly with ``ModelLoadError`` (T-014).

``sentence_transformers`` is replaced by a stub that raises on construction, so no
network and no ``torch`` are needed.
"""

import json
import sys
import types
from pathlib import Path

import pytest

from ragdoc import cli
from ragdoc.indexer import ModelLoadError
from ragdoc.indexer.local_embeddings import LocalEmbeddingsIndexer

FIXTURES = Path(__file__).parent / "fixtures"
_MANIFEST = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))


@pytest.fixture
def broken_model(monkeypatch):
    """``SentenceTransformer(name)`` raises a non-ValueError, multi-line library error."""

    class _Boom:
        def __init__(self, name, *a, **k):
            raise OSError(f"We couldn't connect to the hub\nto load {name}")

    stub = types.ModuleType("sentence_transformers")
    stub.SentenceTransformer = _Boom
    monkeypatch.setitem(sys.modules, "sentence_transformers", stub)


def test_model_load_error_is_value_error():
    assert issubclass(ModelLoadError, ValueError)


def test_embed_query_raises_model_load_error(broken_model):
    with pytest.raises(ModelLoadError) as exc:
        LocalEmbeddingsIndexer(model_name="no-such-model").embed_query("q")
    assert "no-such-model" in str(exc.value)
    assert "not cached" in str(exc.value)
    assert isinstance(exc.value.__cause__, OSError)


def test_empty_index_still_loads_no_model(broken_model):
    LocalEmbeddingsIndexer(model_name="no-such-model").index([])


def _args():
    entry = next(f for f in _MANIFEST["fixtures"] if f["name"] == "control")
    return str(FIXTURES / entry["pdf"]), entry["question"], entry["expected_answer"]


@pytest.mark.parametrize("command", ["diagnose", "batch", "sweep"])
def test_cli_reports_clean_error(command, broken_model, tmp_path, capfd):
    pdf, question, answer = _args()
    q = tmp_path / "q.json"
    q.write_text(json.dumps([{"question": question, "expected_answer": answer}]),
                 encoding="utf-8")
    argv = {
        "diagnose": ["diagnose", pdf, question, answer],
        "batch": ["batch", pdf, str(q)],
        "sweep": ["sweep", pdf, str(q)],
    }[command] + ["--embedding-model", "bad-name"]
    assert cli.main(argv) == 2
    err = capfd.readouterr().err
    assert err.startswith("ragdoc: error: ")
    assert "bad-name" in err
    assert "Traceback" not in err
    assert len(err.strip().splitlines()) == 1
