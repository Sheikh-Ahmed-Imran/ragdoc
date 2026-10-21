import pytest

from ragdoc.loader import Loader, UnknownLoaderError, get, names, register
from ragdoc.loader import registry
from ragdoc.loader.pymupdf_loader import PyMuPDFLoader


@pytest.fixture
def isolated_registry():
    """Snapshot and restore the global loader registry around a test.

    A test that registers a throwaway loader must not leak it into the global
    registry: a later ``names()`` in the same process would then include it and
    could break tests that assert the exact set of registered loaders (e.g. the
    compare tests in T-006/T-007).
    """
    snapshot = dict(registry._REGISTRY)
    try:
        yield
    finally:
        registry._REGISTRY.clear()
        registry._REGISTRY.update(snapshot)


def test_get_pymupdf():
    assert isinstance(get("pymupdf"), PyMuPDFLoader)


def test_names_includes_pymupdf():
    assert "pymupdf" in names()


def test_unknown_name_raises_naming_value():
    with pytest.raises(UnknownLoaderError, match="nope"):
        get("nope")


def test_register_new_loader_and_rejects_duplicates(isolated_registry):
    class Dummy(Loader):
        name = "dummy-test"

        def load(self, path):
            raise NotImplementedError

    class Other(Dummy):
        pass

    assert register(Dummy) is Dummy
    assert isinstance(get("dummy-test"), Dummy)
    with pytest.raises(ValueError):
        register(Other)
    assert "dummy-test" in names()


def test_dummy_loader_does_not_leak_globally():
    """After the isolated test above, the throwaway loader is gone again."""
    assert "dummy-test" not in names()
