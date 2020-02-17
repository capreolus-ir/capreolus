import pytest

from capreolus.collection import DummyCollection
from capreolus.index import AnseriniIndex


@pytest.fixture(scope="function")
def tmpdir_as_cache(tmpdir, monkeypatch):
    monkeypatch.setenv("CAPREOLUS_CACHE", str(tmpdir))


@pytest.fixture(scope="function")
def dummy_index(tmpdir_as_cache):
    index = AnseriniIndex({"_name": "anserini", "indexstops": False, "stemmer": "porter"})
    index.modules["collection"] = DummyCollection({"_name": "dummy"})
    index.create_index()
    return index
