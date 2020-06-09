import os
import shutil

import pytest

from capreolus.index import AnseriniIndex
from capreolus.collection import ANTIQUE, CodeSearchNet


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
