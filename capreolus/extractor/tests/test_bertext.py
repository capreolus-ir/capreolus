import os
import numpy as np
import pytest

from capreolus.benchmark.robust04 import Robust04Benchmark
from capreolus.collection import Collection
from capreolus.extractor.berttext import BertText
from capreolus.searcher.bm25 import BM25Grid
from capreolus.tests.common_fixtures import trec_index, dummy_collection_config


def test_transform_qid_posdocid_negdocid(monkeypatch, tmpdir, trec_index, dummy_collection_config):
    collection = Collection(dummy_collection_config)
    pipeline_config = {
        "indexstops": True,
        "maxthreads": 1,
        "stemmer": "anserini",
        "bmax": 0.2,
        "k1max": 0.2,
        "maxqlen": 5,
        "maxdoclen": 10,
        "keepstops": True,
        "rundocsonly": False,
    }
    bm25_run = BM25Grid(trec_index, collection, os.path.join(tmpdir, "searcher"), pipeline_config)
    bm25_run.create()
    folds = {"s1": {"train_qids": ["301"], "predict": {"dev": ["301"], "test": ["301"]}}}
    benchmark = Robust04Benchmark(bm25_run, collection, pipeline_config)
    benchmark.create_and_store_train_and_pred_pairs(folds)

    feature = BertText(tmpdir, tmpdir, pipeline_config, index=trec_index, collection=collection, benchmark=benchmark)
    feature.build_from_benchmark()
    transformed = feature.transform_qid_posdocid_negdocid("301", "LA010189-0001", "LA010189-0001")

    assert np.array_equal(
        transformed["postoks"],
        [101, 24369, 9986, 0, 0, 0, 102, 24369, 24369, 24369, 7592, 2088, 1010, 14806, 2015, 2013, 6058, 102],
    )
    assert np.array_equal(transformed["posmask"], [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    assert np.array_equal(transformed["possegs"], [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    assert np.array_equal(transformed["posqmask"], [1, 1, 0, 0, 0])
    assert np.array_equal(transformed["posdmask"], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    assert np.array_equal(
        transformed["negtoks"],
        [101, 24369, 9986, 0, 0, 0, 102, 24369, 24369, 24369, 7592, 2088, 1010, 14806, 2015, 2013, 6058, 102],
    )
    assert np.array_equal(transformed["negmask"], [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    assert np.array_equal(transformed["negsegs"], [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    assert np.array_equal(transformed["negqmask"], [1, 1, 0, 0, 0])
    assert np.array_equal(transformed["negdmask"], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    assert transformed["posdocid"] == "LA010189-0001"
    assert transformed["negdocid"] == "LA010189-0001"
    assert transformed["qid"] == "301"
