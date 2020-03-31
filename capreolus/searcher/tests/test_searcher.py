import os
import numpy as np
import pytest
from sacred.config import ConfigScope

from capreolus.benchmark import DummyBenchmark
from capreolus.searcher import BM25, BM25Grid
from capreolus.tests.common_fixtures import tmpdir_as_cache, dummy_index


def test_searcher_bm25(tmpdir_as_cache, tmpdir, dummy_index):
    searcher_config = ConfigScope(BM25.config)()
    searcher_config["_name"] = BM25.name
    searcher = BM25(searcher_config)
    searcher.modules["index"] = dummy_index
    topics_fn = DummyBenchmark.topic_file

    output_fn = searcher.query_from_file(topics_fn, os.path.join(searcher.get_cache_path(), DummyBenchmark.name))

    assert output_fn == os.path.join(searcher.get_cache_path(), DummyBenchmark.name)

    with open(os.path.join(output_fn, "searcher"), "r") as fp:
        file_contents = fp.readlines()

    assert file_contents == ["301 Q0 LA010189-0001 1 0.139500 Anserini\n", "301 Q0 LA010189-0002 2 0.097000 Anserini\n"]


def test_searcher_bm25_grid(tmpdir_as_cache, tmpdir, dummy_index):
    searcher_config = ConfigScope(BM25Grid.config)()
    searcher_config["_name"] = BM25Grid.name
    searcher = BM25Grid(searcher_config)
    searcher.modules["index"] = dummy_index
    bs = np.around(np.arange(0.1, 1 + 0.1, 0.1), 1)
    k1s = np.around(np.arange(0.1, 1 + 0.1, 0.1), 1)
    topics_fn = DummyBenchmark.topic_file

    output_fn = searcher.query_from_file(topics_fn, os.path.join(searcher.get_cache_path(), DummyBenchmark.name))
    assert output_fn == os.path.join(searcher.get_cache_path(), DummyBenchmark.name)

    for k1 in k1s:
        for b in bs:
            assert os.path.exists(os.path.join(output_fn, "searcher_k1={0},b={1}".format(k1, b)))
    assert os.path.exists(os.path.join(output_fn, "done"))
