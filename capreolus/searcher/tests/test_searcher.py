import os
import numpy as np
import pytest
from sacred.config import ConfigScope

from capreolus.benchmark import DummyBenchmark
from capreolus.searcher import BM25, BM25Grid
from capreolus.tests.common_fixtures import tmpdir_as_cache, dummy_index


def test_searcher_bm25(tmpdir_as_cache, tmpdir, dummy_index):
    searcher = BM25(provide={"index": dummy_index})
    topics_fn = DummyBenchmark.topic_file

    output_fn = searcher.query_from_file(topics_fn, os.path.join(searcher.get_cache_path(), DummyBenchmark.module_name))

    assert output_fn == os.path.join(searcher.get_cache_path(), DummyBenchmark.module_name)

    with open(os.path.join(output_fn, "searcher"), "r") as fp:
        file_contents = fp.readlines()

    assert file_contents == ["301 Q0 LA010189-0001 1 0.139500 Anserini\n", "301 Q0 LA010189-0002 2 0.097000 Anserini\n"]


def test_searcher_bm25_grid(tmpdir_as_cache, tmpdir, dummy_index):
    searcher = BM25Grid(provide={"index": dummy_index})
    bs = np.around(np.arange(0.1, 1 + 0.1, 0.1), 1)
    k1s = np.around(np.arange(0.1, 1 + 0.1, 0.1), 1)
    topics_fn = DummyBenchmark.topic_file

    output_fn = searcher.query_from_file(topics_fn, os.path.join(searcher.get_cache_path(), DummyBenchmark.module_name))
    assert output_fn == os.path.join(searcher.get_cache_path(), DummyBenchmark.module_name)

    for k1 in k1s:
        for b in bs:
            assert os.path.exists(os.path.join(output_fn, "searcher_bm25(k1={0},b={1})_default".format(k1, b)))
    assert os.path.exists(os.path.join(output_fn, "done"))
