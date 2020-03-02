import os
import shutil

from capreolus.index import AnseriniIndex
from capreolus.collection import ANTIQUE


def test_antique_downloadifmissing():
    cfg = {"_name": "antique"}
    col = ANTIQUE(cfg)

    # make sure index can be built on this collection
    cfg = {"_name": "anserini", "indexstops": False, "stemmer": "porter"}
    index = AnseriniIndex(cfg)
    index.modules["collection"] = col

    index.create_index()
    assert index.exists()
