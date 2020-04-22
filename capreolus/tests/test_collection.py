import os
import shutil

from capreolus.index import AnseriniIndex
from capreolus.collection import ANTIQUE, CodeSearchNet


def test_antique_downloadifmissing():
    cfg = {"_name": "antique"}
    col = ANTIQUE(cfg)

    # make sure index can be built on this collection
    cfg = {"_name": "anserini", "indexstops": False, "stemmer": "porter"}
    index = AnseriniIndex(cfg)
    index.modules["collection"] = col

    index.create_index()
    assert index.exists()


def test_csn_downloadifmissing():
    cfg = {"_name": "codesearchnet", "lang": "ruby"}
    col = CodeSearchNet(cfg)

    # make sure index can be built on this collection
    cfg = {"_name": "anserini", "indexstops": False, "stemmer": "porter"}
    index = AnseriniIndex(cfg)
    index.modules["collection"] = col

    index.create_index()
    assert index.exists()

