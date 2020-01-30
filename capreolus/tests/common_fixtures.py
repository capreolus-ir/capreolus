import os

import pytest

from capreolus.collection import COLLECTIONS, Collection
from capreolus.index.anserini import AnseriniIndex
from capreolus.utils.common import Anserini


@pytest.fixture(scope="function")
def trec_index(request, tmpdir):
    """
    Build an index based on sample data and create an AnseriniIndex instance based on it
    """
    indir = os.path.join(COLLECTIONS["dummy"].basepath, "dummy")
    outdir = os.path.join(tmpdir, "index")
    anserini_fat_jar = Anserini.get_fat_jar()
    cmd = f"java -classpath {anserini_fat_jar} -Xms512M -Xmx31G -Dapp.name=IndexCollection io.anserini.index.IndexCollection  -collection TrecCollection -generator JsoupGenerator -threads 1 -input {indir} -index {outdir} -storeTransformedDocs"
    os.system(cmd)
    collection = Collection(dummy_collection_config())
    anserini_index = AnseriniIndex(collection, outdir, os.path.join(tmpdir, "index_cache"))
    anserini_index.open()
    return anserini_index


@pytest.fixture(scope="module")
def dummy_collection_config():
    collection_path = COLLECTIONS["dummy"].basepath
    return {
        "name": "dummy",
        "topics": {"type": "trec", "path": os.path.join(collection_path, "topics.dummy.txt")},
        "qrels": {"type": "trec", "path": os.path.join(collection_path, "qrels.dummy.txt")},
        "documents": {"type": "trec", "path": os.path.join(collection_path, "dummy")},
    }
