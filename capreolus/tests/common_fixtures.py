import pytest
from pathlib import Path

from capreolus.collection import DummyCollection
from capreolus.index import AnseriniIndex
from capreolus import registry


@pytest.fixture(scope="function")
def tmpdir_as_cache(tmpdir, monkeypatch):
    monkeypatch.setattr(registry, "CACHE_BASE_PATH", Path(tmpdir))


@pytest.fixture(scope="function")
def dummy_index(tmpdir_as_cache):
    index = AnseriniIndex({"_name": "anserini", "indexstops": False, "stemmer": "porter"})
    index.modules["collection"] = DummyCollection({"_name": "dummy"})
    index.create_index()
    return index
