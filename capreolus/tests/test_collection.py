import pytest

from capreolus import Collection, module_registry
from capreolus.collection.antique import ANTIQUE
from capreolus.collection.codesearchnet import CodeSearchNet
from capreolus.index import AnseriniIndex
from capreolus.tests.common_fixtures import tmpdir_as_cache

collections = set(module_registry.get_module_names("collection"))


@pytest.mark.parametrize("collection_name", collections)
def test_collection_creatable(tmpdir_as_cache, collection_name):
    collection = Collection.create(collection_name)


@pytest.mark.parametrize("collection_name", collections)
@pytest.mark.download
def test_collection_downloadable(tmpdir_as_cache, collection_name):
    collection = Collection.create(collection_name)
    collection.find_document_path()


@pytest.mark.download
def test_antique_downloadifmissing():
    cfg = {"name": "antique"}
    col = ANTIQUE(cfg)

    # make sure index can be built on this collection
    cfg = {"name": "anserini", "indexstops": False, "stemmer": "porter"}
    index = AnseriniIndex(cfg, provide={"collection": col})

    index.create_index()
    assert index.exists()


@pytest.mark.download
def test_csn_downloadifmissing():
    for lang in ["ruby"]:
        cfg = {"name": "codesearchnet", "lang": lang}
        col = CodeSearchNet(cfg)

        # make sure index can be built on this collection
        cfg = {"name": "anserini", "indexstops": False, "stemmer": "porter"}
        index = AnseriniIndex(cfg, provide={"collection": col})

        index.create_index()
        assert index.exists()
