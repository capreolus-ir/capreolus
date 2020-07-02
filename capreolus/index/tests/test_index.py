import pytest

from capreolus import module_registry
from capreolus.collection import DummyCollection
from capreolus.index import Index
from capreolus.tests.common_fixtures import dummy_index, tmpdir_as_cache

indexs = set(module_registry.get_module_names("index"))


@pytest.mark.parametrize("index_name", indexs)
def test_create_index(tmpdir_as_cache, index_name):
    provide = {"collection": DummyCollection()}
    index = Index.create(index_name, provide=provide)
    assert not index.exists()
    index.create_index()
    assert index.exists()


def test_anserini_get_docs(tmpdir_as_cache, dummy_index):
    docs = dummy_index.get_docs(["LA010189-0001"])
    assert docs == ["Dummy Dummy Dummy Hello world, greetings from outer space!"]
    docs = dummy_index.get_docs(["LA010189-0001", "LA010189-0002"])
    assert docs == [
        "Dummy Dummy Dummy Hello world, greetings from outer space!",
        "Dummy LessDummy Hello world, greetings from outer space!",
    ]


def test_anserini_get_df(tmpdir_as_cache, dummy_index):
    df = dummy_index.get_df("hello")
    assert df == 2


def test_anserini_get_idf(tmpdir_as_cache, dummy_index):
    idf = dummy_index.get_idf("hello")
    assert idf == 0.1823215567939546
