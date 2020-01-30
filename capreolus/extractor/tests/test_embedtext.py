import os

import numpy
import pytest
import mock
from pymagnitude import Magnitude

from capreolus.benchmark.robust04 import Robust04Benchmark
from capreolus.collection import Collection
from capreolus.extractor.embedtext import EmbedText
from capreolus.index.anserini import AnseriniIndex
from capreolus.searcher.bm25 import BM25Grid
from capreolus.tests.common_fixtures import trec_index, dummy_collection_config


def test_tokenize_text(trec_index, tmpdir):
    toks_list = [["to", "be", "or", "not", "to", "be"]]
    feature = EmbedText(tmpdir, tmpdir, {}, index=trec_index)
    feature.build_stoi(toks_list, True, False)
    assert feature.stoi == {"<pad>": 0, "to": 1, "be": 2, "or": 3, "not": 4}

    assert feature.idf == {}


def test_tokenize_text_with_calculate_idf(dummy_collection_config, trec_index, tmpdir):
    toks_list = [["to", "be", "or", "not", "to", "be"]]
    feature = EmbedText(tmpdir, tmpdir, {}, index=trec_index)
    feature.build_stoi(toks_list, True, True)
    assert feature.stoi == {"<pad>": 0, "to": 1, "be": 2, "or": 3, "not": 4}

    assert feature.idf == {"be": 1.791759469228055, "not": 1.791759469228055, "or": 1.791759469228055, "to": 1.791759469228055}


def test_create_embedding_matrix(monkeypatch, tmpdir, trec_index):
    feature = EmbedText(tmpdir, tmpdir, {"reranker": "KNRM"}, index=trec_index)
    feature.stoi = {"<pad>": 0, "hello": 1, "world": 2}

    # Prevents a download when the unit tests are searcher
    def fake_magnitude_embedding(*args, **kwargs):
        return Magnitude(None)

    monkeypatch.setattr(feature, "get_magnitude_embeddings", fake_magnitude_embedding)
    matrix = feature.create_embedding_matrix("glove6b")

    # We cannot assert the entire matrix because since there are no downloaded embeddings, the embedding for a word
    # would be random each time we searcher the test
    assert matrix.shape == (3, 8)
    assert numpy.array_equal(matrix[0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def test_transform_qid_posdocid_negdocid_only_posdoc(tmpdir, trec_index, dummy_collection_config):
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
    }
    bm25_run = BM25Grid(trec_index, collection, os.path.join(tmpdir, "searcher"), pipeline_config)
    benchmark = Robust04Benchmark(bm25_run, collection, pipeline_config)
    feature = EmbedText(tmpdir, tmpdir, pipeline_config, index=trec_index, collection=collection, benchmark=benchmark)
    feature.stoi["dummy"] = 1
    feature.itos[1] = "dummy"
    feature.doc_id_to_doc_toks = {
        "LA010189-0001": ["dummy", "dummy", "dummy", "hello", "world", "greetings", "from", "outer", "space"],
        "LA010189-0001": ["dummy", "dummy", "dummy", "hello", "world", "greetings", "from", "outer", "space"],
    }
    transformed = feature.transform_qid_posdocid_negdocid("301", "LA010189-0001")

    # stoi only knows about the word 'dummy'. So the transformation of every other word is set as 0
    assert transformed["qid"] == "301"
    assert transformed["posdocid"] == "LA010189-0001"
    assert transformed["negdocid"] is None
    assert numpy.array_equal(transformed["query"], [1, 0, 0, 0, 0])
    assert numpy.array_equal(transformed["posdoc"], [1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    assert numpy.array_equal(transformed["query_idf"], [0, 0, 0, 0, 0])

    # Learn another word
    feature.stoi["hello"] = 2
    feature.itos[2] = "hello"
    transformed = feature.transform_qid_posdocid_negdocid("301", "LA010189-0001")
    # The posdoc transformation changes to reflect the new word
    assert numpy.array_equal(transformed["posdoc"], [1, 1, 1, 2, 0, 0, 0, 0, 0, 0])


def test_transform_qid_posdocid_negdocid_with_negdoc(tmpdir, trec_index, dummy_collection_config):
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
    }
    bm25_run = BM25Grid(trec_index, collection, os.path.join(tmpdir, "searcher"), pipeline_config)
    benchmark = Robust04Benchmark(bm25_run, collection, pipeline_config)
    feature = EmbedText(tmpdir, tmpdir, pipeline_config, index=trec_index, collection=collection, benchmark=benchmark)
    feature.stoi["dummy"] = 1
    feature.itos[1] = "dummy"
    feature.doc_id_to_doc_toks = {
        "LA010189-0001": ["dummy", "dummy", "dummy", "hello", "world", "greetings", "from", "outer", "space"],
        "LA010189-0001": ["dummy", "dummy", "dummy", "hello", "world", "greetings", "from", "outer", "space"],
    }

    transformed = feature.transform_qid_posdocid_negdocid("301", "LA010189-0001", "LA010189-0001")

    # stoi only knows about the word 'dummy'. So the transformation of every other word is set as 0
    assert transformed["qid"] == "301"
    assert transformed["posdocid"] == "LA010189-0001"
    assert transformed["negdocid"] == "LA010189-0001"
    assert numpy.array_equal(transformed["query"], [1, 0, 0, 0, 0])
    assert numpy.array_equal(transformed["posdoc"], [1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    assert numpy.array_equal(transformed["negdoc"], [1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    assert numpy.array_equal(transformed["query_idf"], [0, 0, 0, 0, 0])


def test_build_from_benchmark(monkeypatch, tmpdir, trec_index, dummy_collection_config):
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
        "reranker": "KNRM",
    }
    bm25_run = BM25Grid(trec_index, collection, os.path.join(tmpdir, "searcher"), pipeline_config)
    bm25_run.create()
    folds = {"s1": {"train_qids": ["301"], "predict": {"dev": ["301"], "test": ["301"]}}}
    benchmark = Robust04Benchmark(bm25_run, collection, pipeline_config)
    benchmark.create_and_store_train_and_pred_pairs(folds)

    # Prevents a download when the unit tests are searcher
    def fake_magnitude_embedding(*args, **kwargs):
        return Magnitude(None)

    feature = EmbedText(tmpdir, tmpdir, pipeline_config, index=trec_index, collection=collection, benchmark=benchmark)
    monkeypatch.setattr(feature, "get_magnitude_embeddings", fake_magnitude_embedding)

    feature.build_from_benchmark("glove6b", True)
    assert feature.stoi == {
        "<pad>": 0,
        "dummy": 1,
        "doc": 2,
        "hello": 3,
        "world": 4,
        "greetings": 5,
        "from": 6,
        "outer": 7,
        "space": 8,
    }

    assert feature.itos == {v: k for k, v in feature.stoi.items()}
    assert numpy.array_equal(feature.embeddings[0], [0, 0, 0, 0, 0, 0, 0, 0])
    assert feature.embeddings.shape == (9, 8)


# def test_build_from_benchmark_large_collection_cached(monkeypatch, tmpdir, trec_index, dummy_collection_config):
#     # raise Exception("TODO: Fix large collection crawling")
#     trec_index.collection.is_large_collection = True
#     pipeline_config = {
#         "indexstops": True,
#         "maxthreads": 1,
#         "stemmer": "anserini",
#         "bmax": 0.2,
#         "k1max": 0.2,
#         "maxqlen": 5,
#         "maxdoclen": 10,
#         "keepstops": True,
#         "rundocsonly": False,
#         "reranker": "KNRM",
#     }
#
#     bm25_run = BM25Grid(trec_index, trec_index.collection, os.path.join(tmpdir, "searcher"), pipeline_config)
#     bm25_run.create()
#     folds = {"s1": {"train_qids": ["301"], "predict": {"dev": ["301"], "test": ["301"]}}}
#     benchmark = Robust04Benchmark(bm25_run, trec_index.collection, pipeline_config)
#     benchmark.create_and_store_train_and_pred_pairs(folds)
#
#     # Prevents a download when the unit tests are searcher
#     def fake_magnitude_embedding(*args, **kwargs):
#         return Magnitude(None)
#
#     extractor = EmbedText(
#         tmpdir, tmpdir, pipeline_config, index=trec_index, collection=trec_index.collection, benchmark=benchmark
#     )
#     monkeypatch.setattr(extractor, "get_magnitude_embeddings", fake_magnitude_embedding)
#
#     with mock.patch.object(trec_index, "get_documents_from_disk", wraps=trec_index.get_documents_from_disk) as mock_get_from_disk:
#         extractor.build_from_benchmark("glove6b", True)
#         mock_get_from_disk.assert_called()
#
#     # Test if extractor works with is_large_collection
#     assert extractor.stoi == {
#         "<pad>": 0,
#         "dummy": 1,
#         "doc": 2,
#         "hello": 3,
#         "world": 4,
#         "greetings": 5,
#         "from": 6,
#         "outer": 7,
#         "space": 8,
#     }
#
#     assert extractor.itos == {v: k for k, v in extractor.stoi.items()}
#     assert numpy.array_equal(extractor.embeddings[0], [0, 0, 0, 0, 0, 0, 0, 0])
#     assert extractor.embeddings.shape == (9, 8)
#
#     # Repeat the same thing, but assert that cache was used
#     with mock.patch.object(trec_index, "get_documents_from_disk", wraps=trec_index.get_documents_from_disk) as mock_get_from_disk:
#         extractor = EmbedText(
#             tmpdir, tmpdir, pipeline_config, index=trec_index, collection=trec_index.collection, benchmark=benchmark
#         )
#         monkeypatch.setattr(extractor, "get_magnitude_embeddings", fake_magnitude_embedding)
#         extractor.build_from_benchmark("glove6b", True)
#         mock_get_from_disk.assert_not_called()
#
#     assert extractor.stoi == {
#         "<pad>": 0,
#         "dummy": 1,
#         "doc": 2,
#         "hello": 3,
#         "world": 4,
#         "greetings": 5,
#         "from": 6,
#         "outer": 7,
#         "space": 8,
#     }
#
#     assert extractor.itos == {v: k for k, v in extractor.stoi.items()}
#     assert numpy.array_equal(extractor.embeddings[0], [0, 0, 0, 0, 0, 0, 0, 0])
#     assert extractor.embeddings.shape == (9, 8)
