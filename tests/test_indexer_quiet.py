"""Model load is quiet of HF chatter and prints one first-run notice (T-019).

``sentence_transformers`` and ``huggingface_hub`` are stubbed: no network, no model.
"""

import importlib
import os
import sys
import types

import pytest

from ragdoc.indexer import ModelLoadError
from ragdoc.indexer import local_embeddings
from ragdoc.indexer.local_embeddings import LocalEmbeddingsIndexer


# --- the actual suppression mechanism -------------------------------------
# huggingface_hub/transformers read these env vars at *their* import time, so the
# env vars (not the runtime disable_* calls) are what truly silences the progress
# bar and the unauthenticated-requests warning. A real end-to-end run confirmed
# the runtime calls alone leave the noise; these tests pin the env-var fix so it
# cannot silently regress again (they need neither torch nor the network).


def test_hf_noise_quieted_via_env_at_import(monkeypatch):
    for var in (
        "HF_HUB_DISABLE_PROGRESS_BARS",
        "HF_HUB_VERBOSITY",
        "TRANSFORMERS_VERBOSITY",
        "TRANSFORMERS_NO_ADVISORY_WARNINGS",
    ):
        monkeypatch.delenv(var, raising=False)
    importlib.reload(local_embeddings)
    assert os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"
    assert os.environ["HF_HUB_VERBOSITY"] == "error"
    assert os.environ["TRANSFORMERS_VERBOSITY"] == "error"
    assert os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] == "1"


def test_user_hf_env_is_respected(monkeypatch):
    monkeypatch.setenv("HF_HUB_VERBOSITY", "info")
    importlib.reload(local_embeddings)
    assert os.environ["HF_HUB_VERBOSITY"] == "info"


@pytest.fixture
def hf(monkeypatch):
    """Stub huggingface_hub with spies; ``state['cached']`` drives the cache probe."""
    state = {"cached": False, "calls": []}

    hub = types.ModuleType("huggingface_hub")
    hub.disable_progress_bars = lambda: state["calls"].append("bars")

    def probe(repo_id, filename):
        state["calls"].append(("probe", repo_id, filename))
        return "/cache/modules.json" if state["cached"] else None

    hub.try_to_load_from_cache = probe
    hub_logging = types.ModuleType("huggingface_hub.logging")
    hub_logging.set_verbosity_error = lambda: state["calls"].append("quiet")
    hub.logging = hub_logging
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.logging", hub_logging)
    monkeypatch.setattr(local_embeddings, "_hf_quieted", False)
    return state


@pytest.fixture
def st_ok(monkeypatch):
    class _Model:
        def __init__(self, name, *a, **k):
            self.name = name

    stub = types.ModuleType("sentence_transformers")
    stub.SentenceTransformer = _Model
    monkeypatch.setitem(sys.modules, "sentence_transformers", stub)


def test_quiets_hf_once_per_process(hf, st_ok):
    LocalEmbeddingsIndexer(model_name="m")._get_model()
    LocalEmbeddingsIndexer(model_name="m")._get_model()
    assert hf["calls"].count("bars") == 1
    assert hf["calls"].count("quiet") == 1


def test_not_cached_writes_one_stderr_line(hf, st_ok, capfd):
    LocalEmbeddingsIndexer(model_name="all-MiniLM-L6-v2")._get_model()
    out, err = capfd.readouterr()
    assert out == ""
    assert len(err.splitlines()) == 1
    assert "all-MiniLM-L6-v2" in err and "first run" in err
    assert ("probe", "sentence-transformers/all-MiniLM-L6-v2", "modules.json") in hf["calls"]


def test_org_qualified_name_probed_as_is(hf, st_ok):
    LocalEmbeddingsIndexer(model_name="org/model")._get_model()
    assert ("probe", "org/model", "modules.json") in hf["calls"]


def test_cached_writes_nothing(hf, st_ok, capfd):
    hf["cached"] = True
    LocalEmbeddingsIndexer(model_name="m")._get_model()
    assert capfd.readouterr() == ("", "")


def test_second_load_after_cache_populated_is_silent(hf, st_ok, capfd):
    LocalEmbeddingsIndexer(model_name="m")._get_model()
    assert capfd.readouterr().err != ""
    hf["cached"] = True
    LocalEmbeddingsIndexer(model_name="m")._get_model()
    assert capfd.readouterr() == ("", "")


def test_missing_hf_api_never_breaks_load(monkeypatch, st_ok):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)  # import fails
    monkeypatch.setattr(local_embeddings, "_hf_quieted", False)
    assert LocalEmbeddingsIndexer(model_name="m")._get_model() is not None


def test_load_failure_still_model_load_error(hf, monkeypatch):
    class _Boom:
        def __init__(self, name, *a, **k):
            raise OSError("no network")

    stub = types.ModuleType("sentence_transformers")
    stub.SentenceTransformer = _Boom
    monkeypatch.setitem(sys.modules, "sentence_transformers", stub)
    with pytest.raises(ModelLoadError):
        LocalEmbeddingsIndexer(model_name="m")._get_model()
