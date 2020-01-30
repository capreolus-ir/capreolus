import os

import pytest
import pytrec_eval

from capreolus.tests.common_fixtures import trec_index, dummy_collection_config
from capreolus.collection import Collection
from capreolus.searcher.bm25 import BM25Grid
from capreolus.searcher import Searcher


def test_bm25grid_create(trec_index, dummy_collection_config, tmpdir):
    collection = Collection(dummy_collection_config)
    pipeline_config = {"indexstops": True, "maxthreads": 1, "stemmer": "anserini", "bmax": 0.2, "k1max": 0.2}

    bm25_run = BM25Grid(trec_index, collection, os.path.join(tmpdir, "searcher"), pipeline_config)
    bm25_run.create()

    # Make sure that the searcher file is generated
    os.path.isfile(os.path.join(tmpdir, "searcher", "done"))


def test_cross_validated_ranking(trec_index, dummy_collection_config, tmpdir):
    collection = Collection(dummy_collection_config)
    pipeline_config = {"indexstops": True, "maxthreads": 1, "stemmer": "anserini", "bmax": 0.2, "k1max": 0.2}

    bm25_run = BM25Grid(trec_index, collection, os.path.join(tmpdir, "searcher"), pipeline_config)
    test_ranking = bm25_run.crossvalidated_ranking(["301"], ["301"])

    assert test_ranking["301"]["LA010189-0001"] > 0
    assert test_ranking["301"]["LA010189-0002"] > 0


def test_search_run_metrics(tmpdir):
    qrels_dict = {"q1": {"d1": 1, "d2": 0, "d3": 2}, "q2": {"d5": 0, "d6": 1}}
    run_dict = {"q1": {"d1": 1.1, "d2": 1.0}, "q2": {"d5": 9.0, "d6": 8.0}, "q3": {"d7": 1.0, "d8": 2.0}}
    valid_metrics = {"P", "map", "map_cut", "ndcg_cut", "Rprec", "recip_rank"}

    fn = tmpdir / "searcher"
    Searcher.write_trec_run(run_dict, fn)

    # calculate results with q1 and q2
    searcher = Searcher(None, None, None, None)
    qids = set(qrels_dict.keys())
    evaluator = pytrec_eval.RelevanceEvaluator(qrels_dict, valid_metrics)
    partial_metrics = searcher.search_run_metrics(fn, evaluator, qids)

    # cache file exists?
    assert os.path.exists(fn + ".metrics")

    # add q3 and re-run to update cache
    qrels_dict["q3"] = {"d7": 0, "d8": 2}
    qids = set(qrels_dict.keys())
    evaluator = pytrec_eval.RelevanceEvaluator(qrels_dict, valid_metrics)
    metrics = searcher.search_run_metrics(fn, evaluator, qids)
    assert "q3" in metrics
    assert "q2" in metrics

    # remove original file to ensure results loaded from cache,
    # then make sure metrics haven't changed (and include the new q3)
    os.remove(fn)
    cached_metrics = searcher.search_run_metrics(fn, evaluator, qids)
    assert metrics == cached_metrics
