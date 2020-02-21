import os
import shutil
from pathlib import Path

from capreolus.index import AnseriniIndex
from capreolus.collection import ANTIQUE


def _remove_folder(folder):
    for f in os.listdir(folder):
        os.remove(os.path.join(folder, f))
    shutil.rmtree(folder)


def test_downloadifmissing():
    cfg = {"_name": "antique"}
    col = ANTIQUE(cfg)

    path_to_col = "/home/x978zhan/tmp_antique22/collection"
    path_to_idx = "/home/x978zhan/tmp_antique22/index"
    if os.path.exists(path_to_col):
        _remove_folder(path_to_col)
    assert not os.path.exists(path_to_col)
    col.path = path_to_col
    # col.download_if_missing()
    path, col_path, gen_type = col.get_path_and_types()
    assert path == path_to_col

    # make sure index can be built on this collection
    cfg = {"_name": "anserini", "indexstops": False, "stemmer": "porter"}
    index = AnseriniIndex(cfg)
    index.modules["collection"] = col
    index.get_index_path = lambda: Path(path_to_idx)

    index.create_index()

    _remove_folder(path_to_col)
    _remove_folder(path_to_idx)

    assert index.exists()
