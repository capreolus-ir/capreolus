import os

import numpy
import pytest
import mock
from pymagnitude import Magnitude

from capreolus.benchmark.robust04 import Robust04Benchmark
from capreolus.collection import Collection
from capreolus.extractor.bagofwords import BagOfWords
from capreolus.extractor.embedtext import EmbedText
from capreolus.index.anserini import AnseriniIndex
from capreolus.searcher.bm25 import BM25Grid
from capreolus.tests.common_fixtures import trec_index, dummy_collection_config


def test_tokenize_text(trec_index, tmpdir):
    toks_list = [["to", "be", "or", "not", "to", "be"]]
    feature = BagOfWords(tmpdir, tmpdir, {"datamode": "unigram"}, index=trec_index)
    feature.build_stoi(toks_list, True, False)
    assert feature.stoi == {"<pad>": 0, "to": 1, "be": 2, "or": 3, "not": 4}

    assert feature.idf == {}


def test_tokenize_text_trigram(trec_index, tmpdir):
    toks_list = [["to", "be", "or", "not", "to", "be"]]
    feature = BagOfWords(tmpdir, tmpdir, {"datamode": "trigram"}, index=trec_index)
    feature.build_stoi(toks_list, True, False)

    # trigrams would be - ['#to', 'to#', '#be', 'be#', '#or', 'or#', "#no', 'not', "ot#']
    assert feature.stoi == {"<pad>": 0, "#to": 1, "to#": 2, "#be": 3, "be#": 4, "#or": 5, "or#": 6, "#no": 7, "not": 8, "ot#": 9}

    assert feature.idf == {}


def test_tokenize_text_with_calculate_idf(dummy_collection_config, trec_index, tmpdir):
    toks_list = [["to", "be", "or", "not", "to", "be"]]
    feature = BagOfWords(tmpdir, tmpdir, {"datamode": "unigram"}, index=trec_index)
    feature.build_stoi(toks_list, True, True)
    assert feature.stoi == {"<pad>": 0, "to": 1, "be": 2, "or": 3, "not": 4}

    assert feature.idf == {"be": 1.791759469228055, "not": 1.791759469228055, "or": 1.791759469228055, "to": 1.791759469228055}


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
        "datamode": "unigram",
    }
    bm25_run = BM25Grid(trec_index, collection, os.path.join(tmpdir, "searcher"), pipeline_config)
    benchmark = Robust04Benchmark(bm25_run, collection, pipeline_config)
    feature = BagOfWords(tmpdir, tmpdir, pipeline_config, index=trec_index, collection=collection, benchmark=benchmark)
    feature.stoi["dummy"] = 1
    feature.stoi["doc"] = 2
    feature.itos[1] = "dummy"
    feature.itos[2] = "doc"
    feature.doc_id_to_doc_toks = {
        "LA010189-0001": ["dummy", "dummy", "dummy", "hello", "world", "greetings", "from", "outer", "space"],
        "LA010189-0001": ["dummy", "dummy", "dummy", "hello", "world", "greetings", "from", "outer", "space"],
    }
    transformed = feature.transform_qid_posdocid_negdocid("301", "LA010189-0001")
    # stoi only knows about the word 'dummy'. So the transformation of every other word is set as 0
    assert transformed["qid"] == "301"
    assert transformed["posdocid"] == "LA010189-0001"
    assert transformed["negdocid"] is None

    # Right now we have only 3 words in the vocabular - "<pad>", "dummy" and "doc"
    assert numpy.array_equal(transformed["query"], [0, 1, 1])
    assert numpy.array_equal(
        transformed["posdoc"], [6, 3, 0]
    )  # There  are 6 unknown words in the doc, so all of them is encoded as 0
    assert numpy.array_equal(transformed["query_idf"], [0, 0, 0])

    # Learn another word
    feature.stoi["hello"] = 3
    feature.itos[3] = "hello"
    transformed = feature.transform_qid_posdocid_negdocid("301", "LA010189-0001")
    # The posdoc transformation changes to reflect the new word
    assert numpy.array_equal(transformed["posdoc"], [5, 3, 0, 1])


def test_transform_qid_posdocid_negdocid_only_posdoc_trigram(tmpdir, trec_index, dummy_collection_config):
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
        "datamode": "trigram",
    }
    bm25_run = BM25Grid(trec_index, collection, os.path.join(tmpdir, "searcher"), pipeline_config)
    benchmark = Robust04Benchmark(bm25_run, collection, pipeline_config)
    feature = BagOfWords(tmpdir, tmpdir, pipeline_config, index=trec_index, collection=collection, benchmark=benchmark)

    # My vocabulary is only partially constructed
    feature.stoi["#du"] = 1
    feature.stoi["dum"] = 2
    feature.stoi["umm"] = 3
    feature.itos[1] = "#du"
    feature.itos[2] = "dum"
    feature.itos[3] = "umm"

    feature.doc_id_to_doc_toks = {
        "LA010189-0001": ["dummy", "dummy", "dummy", "hello", "world", "greetings", "from", "outer", "space"],
        "LA010189-0001": ["dummy", "dummy", "dummy", "hello", "world", "greetings", "from", "outer", "space"],
    }
    transformed = feature.transform_qid_posdocid_negdocid("301", "LA010189-0001")
    # stoi only knows about the word 'dummy'. So the transformation of every other word is set as 0
    assert transformed["qid"] == "301"
    assert transformed["posdocid"] == "LA010189-0001"
    assert transformed["negdocid"] is None

    # Right now we have only 3 words in the vocabular - "<pad>", "dummy" and "doc"
    assert numpy.array_equal(transformed["query"], [5, 1, 1, 1])
    assert numpy.array_equal(
        transformed["posdoc"], [39, 3, 3, 3]
    )  # There  are 6 unknown words in the doc, so all of them is encoded as 0
    assert numpy.array_equal(transformed["query_idf"], [0, 0, 0, 0])

    # Learn another word
    feature.stoi["mmy"] = 4
    feature.stoi["my#"] = 5
    feature.stoi["#he"] = 6
    feature.itos[4] = "mmy"
    feature.itos[5] = "my#"
    feature.itos[6] = "#he"

    transformed = feature.transform_qid_posdocid_negdocid("301", "LA010189-0001")
    # The posdoc transformation changes to reflect the new word
    assert numpy.array_equal(transformed["posdoc"], [32, 3, 3, 3, 3, 3, 1])


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
        "datamode": "unigram",
    }
    bm25_run = BM25Grid(trec_index, collection, os.path.join(tmpdir, "searcher"), pipeline_config)
    benchmark = Robust04Benchmark(bm25_run, collection, pipeline_config)
    feature = BagOfWords(tmpdir, tmpdir, pipeline_config, index=trec_index, collection=collection, benchmark=benchmark)
    feature.stoi["dummy"] = 1
    feature.stoi["doc"] = 2
    feature.itos[1] = "dummy"
    feature.itos[2] = "doc"
    feature.doc_id_to_doc_toks = {
        "LA010189-0001": ["dummy", "dummy", "dummy", "hello", "world", "greetings", "from", "outer", "space"],
        "LA010189-0001": ["dummy", "dummy", "dummy", "hello", "world", "greetings", "from", "outer", "space"],
    }
    transformed = feature.transform_qid_posdocid_negdocid("301", "LA010189-0001", "LA010189-0001")
    # stoi only knows about the word 'dummy' and 'doc'. So the transformation of every other word is set as 0

    assert transformed["qid"] == "301"
    assert transformed["posdocid"] == "LA010189-0001"
    assert transformed["negdocid"] == "LA010189-0001"
    assert numpy.array_equal(transformed["query"], [0, 1, 1])
    assert numpy.array_equal(transformed["posdoc"], [6, 3, 0])
    assert numpy.array_equal(transformed["negdoc"], [6, 3, 0])
    assert numpy.array_equal(transformed["query_idf"], [0, 0, 0])


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
        "datamode": "unigram",
    }
    bm25_run = BM25Grid(trec_index, collection, os.path.join(tmpdir, "searcher"), pipeline_config)
    bm25_run.create()
    folds = {"s1": {"train_qids": ["301"], "predict": {"dev": ["301"], "test": ["301"]}}}
    benchmark = Robust04Benchmark(bm25_run, collection, pipeline_config)
    benchmark.create_and_store_train_and_pred_pairs(folds)

    feature = BagOfWords(tmpdir, tmpdir, pipeline_config, index=trec_index, collection=collection, benchmark=benchmark)

    feature.build_from_benchmark(True)
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
    assert feature.embeddings == {
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


def test_build_from_benchmark_with_trigram(monkeypatch, tmpdir, trec_index, dummy_collection_config):
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
        "datamode": "trigram",
    }
    bm25_run = BM25Grid(trec_index, collection, os.path.join(tmpdir, "searcher"), pipeline_config)
    bm25_run.create()
    folds = {"s1": {"train_qids": ["301"], "predict": {"dev": ["301"], "test": ["301"]}}}
    benchmark = Robust04Benchmark(bm25_run, collection, pipeline_config)
    benchmark.create_and_store_train_and_pred_pairs(folds)

    feature = BagOfWords(tmpdir, tmpdir, pipeline_config, index=trec_index, collection=collection, benchmark=benchmark)

    feature.build_from_benchmark(True)
    assert feature.stoi == {
        "<pad>": 0,
        "#du": 1,
        "dum": 2,
        "umm": 3,
        "mmy": 4,
        "my#": 5,
        "#do": 6,
        "doc": 7,
        "oc#": 8,
        "#he": 9,
        "hel": 10,
        "ell": 11,
        "llo": 12,
        "lo#": 13,
        "#wo": 14,
        "wor": 15,
        "orl": 16,
        "rld": 17,
        "ld#": 18,
        "#gr": 19,
        "gre": 20,
        "ree": 21,
        "eet": 22,
        "eti": 23,
        "tin": 24,
        "ing": 25,
        "ngs": 26,
        "gs#": 27,
        "#fr": 28,
        "fro": 29,
        "rom": 30,
        "om#": 31,
        "#ou": 32,
        "out": 33,
        "ute": 34,
        "ter": 35,
        "er#": 36,
        "#sp": 37,
        "spa": 38,
        "pac": 39,
        "ace": 40,
        "ce#": 41,
    }

    assert feature.itos == {v: k for k, v in feature.stoi.items()}
