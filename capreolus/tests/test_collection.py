import os
import shutil

import pytest

from capreolus import Collection, constants, module_registry
from capreolus.tests.common_fixtures import tmpdir_as_cache

collections = set(module_registry.get_module_names("collection"))


@pytest.mark.parametrize("collection_name", collections)
def test_collection_creatable(tmpdir_as_cache, collection_name):
    collection = Collection.create(collection_name)


@pytest.mark.parametrize("collection_name", collections)
@pytest.mark.download
def test_collection_downloadable(tmpdir_as_cache, collection_name):
    collection = Collection.create(collection_name)
    path = collection.find_document_path()

    # check for /tmp to reduce the impact of an invalid constants["CACHE_BASE_PATH"]
    if path.startswith("/tmp") and path.startswith(constants["CACHE_BASE_PATH"].as_posix()):
        if os.path.exists(path):
            shutil.rmtree(path)
