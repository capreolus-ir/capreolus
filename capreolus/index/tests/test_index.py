import pytest

from capreolus.collection import Collection
from capreolus.index import Index
from capreolus.index.anserini import AnseriniIndex
from capreolus.tests.common_fixtures import dummy_collection_config


def test_get_index_from_index_path():
    index_path_1 = "/something/anserini/foo"
    index_path_2 = "/foo/bar"

    index_class = Index.get_index_from_index_path(index_path_1)
    assert index_class == AnseriniIndex

    index_class = Index.get_index_from_index_path(index_path_2)
    assert index_class is None


def test_anserini_large_collections(dummy_collection_config, tmpdir):
    # raise Exception("TODO: Fix the jnius issue")
    collection = Collection(dummy_collection_config)
    collection.is_large_collection = True
    index = AnseriniIndex(collection, tmpdir, tmpdir)
    config = {"indexstops": False, "stemmer": "anserini", "maxthreads": 1}

    # Deliberately not calling index.create()
    docs = index.get_docs(["LA010189-0001", "LA010189-0002"])
    assert len(docs) == 2
    assert docs == [
        "Dummy Dummy Dummy Hello world, greetings from outer space!",
        "Dummy Dummy Dummy Hello world, greetings from outer space!",
    ]

    collection.is_large_collection = False
    # Because we would be trying to read from an index that is not present
    with pytest.raises(Exception):
        docs = index.get_docs(["LA010189-0001", "LA010189-0002"])
