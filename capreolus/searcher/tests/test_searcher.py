import os

import numpy as np
import pytest

from capreolus import module_registry
from capreolus.benchmark import DummyBenchmark
from capreolus.searcher.anserini import BM25, BM25Grid, Searcher
from capreolus.tests.common_fixtures import dummy_index, tmpdir_as_cache
from capreolus.utils.trec import load_trec_topics

skip_searchers = {"bm25staticrob04yang19", "BM25Grid", "BM25Postprocess", "axiomatic"}
searchers = set(module_registry.get_module_names("searcher")) - skip_searchers


@pytest.mark.parametrize("searcher_name", searchers)
def test_searcher_runnable(tmpdir_as_cache, tmpdir, dummy_index, searcher_name):
    benchmark = DummyBenchmark()
    searcher = Searcher.create(searcher_name, provide=[benchmark, dummy_index])
    output_dir = searcher.fit()
    assert os.path.exists(os.path.join(output_dir, "done"))


@pytest.mark.parametrize("searcher_name", searchers)
def test_searcher_query(tmpdir_as_cache, tmpdir, dummy_index, searcher_name):
    benchmark = DummyBenchmark()
    query = list(benchmark.topics[benchmark.query_type].values())[0]
    nhits = 1
    searcher = Searcher.create(searcher_name, config={"hits": nhits}, provide=[benchmark, dummy_index])
    results = searcher.query(query)
    if searcher_name == "SPL":
        # if searcher_name != "BM25":
        return

    print(results.values())
    if isinstance(list(results.values())[0], dict):
        assert all(len(d) == nhits for d in results.values())
    else:
        assert len(results) == nhits


def test_searcher_bm25(tmpdir_as_cache, tmpdir, dummy_index):
    benchmark = DummyBenchmark()
    searcher = BM25(provide=[benchmark, dummy_index])

    output_dir = searcher.fit()

    with open(os.path.join(output_dir, "searcher"), "r") as fp:
        file_contents = fp.readlines()

    assert file_contents == ["301 Q0 LA010189-0001 1 0.139500 Anserini\n", "301 Q0 LA010189-0002 2 0.097000 Anserini\n"]


def test_searcher_bm25_grid(tmpdir_as_cache, tmpdir, dummy_index):
    benchmark = DummyBenchmark()
    searcher = BM25Grid(provide=[benchmark, dummy_index])
    bs = np.around(np.arange(0.1, 1 + 0.1, 0.1), 1)
    k1s = np.around(np.arange(0.1, 1 + 0.1, 0.1), 1)

    output_dir = searcher.fit()

    for k1 in k1s:
        for b in bs:
            assert os.path.exists(os.path.join(output_dir, "searcher_bm25(k1={0},b={1})_default".format(k1, b)))
    assert os.path.exists(os.path.join(output_dir, "done"))
