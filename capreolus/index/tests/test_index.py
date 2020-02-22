import pytest

from capreolus.collection import Collection, DummyCollection
from capreolus.index import Index
from capreolus.index import AnseriniIndex
from capreolus.tests.common_fixtures import tmpdir_as_cache, dummy_index


def test_anserini_create_index(tmpdir_as_cache):
    index = AnseriniIndex({"_name": "anserini", "indexstops": False, "stemmer": "porter"})
    index.modules["collection"] = DummyCollection({"_name": "dummy"})
    assert not index.exists()
    index.create_index()
    assert index.exists()


def test_anserini_get_docs(tmpdir_as_cache, dummy_index):
    docs = dummy_index.get_docs(["LA010189-0001"])
    assert docs == ["Dummy Dummy Dummy Hello world, greetings from outer space!"]
    docs = dummy_index.get_docs(["LA010189-0001", "LA010189-0002"])
    assert docs == [
        "Dummy Dummy Dummy Hello world, greetings from outer space!",
        "Dummy LessDummy Hello world, greetings from outer space!"
    ]


def test_anserini_get_df(tmpdir_as_cache, dummy_index):
    df = dummy_index.getdf("hello")
    assert df == 2


def test_anserini_get_idf(tmpdir_as_cache, dummy_index):
    idf = dummy_index.getidf("hello")
    assert idf == 0.1823215567939546
