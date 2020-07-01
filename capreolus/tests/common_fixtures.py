from pathlib import Path

import pytest
from profane import constants

from capreolus.index import AnseriniIndex


@pytest.fixture(scope="function")
def tmpdir_as_cache(tmpdir, monkeypatch):
    constants._d["CACHE_BASE_PATH"] = Path(tmpdir)


@pytest.fixture(scope="function")
def dummy_index(tmpdir_as_cache):
    index = AnseriniIndex({"name": "anserini", "indexstops": False, "stemmer": "porter", "collection": {"name": "dummy"}})
    index.create_index()
    return index
