import pytest

from ragdoc.loader import Loader, UnknownLoaderError, get, names, register
from ragdoc.loader.pymupdf_loader import PyMuPDFLoader


def test_get_pymupdf():
    assert isinstance(get("pymupdf"), PyMuPDFLoader)


def test_names_includes_pymupdf():
    assert "pymupdf" in names()


def test_unknown_name_raises_naming_value():
    with pytest.raises(UnknownLoaderError, match="nope"):
        get("nope")


def test_register_new_loader_and_rejects_duplicates():
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
