import os
import numpy as np
import json
import pytest
from django.utils.datastructures import MultiValueDict

from pytest_mock import mocker

import torch

from capreolus.collection import Collection
from capreolus.demo_app.views import ConfigsView, NeuralQueryView, QueryDictParserMixin, CompareNeuralQueryView, BM25View
from capreolus.extractor.embedding import EmbeddingHolder
from capreolus.reranker.KNRM import KNRM
from capreolus.pipeline import Pipeline
from capreolus.tests.common_fixtures import trec_index
from capreolus.tokenizer import AnseriniTokenizer


@pytest.fixture()
def embedding_holder():
    return EmbeddingHolder.get_instance("glove6b.50d")


@pytest.fixture(scope="module")
def anserini_tokenizer():
    tokenizer = AnseriniTokenizer(None, use_cache=False)
    tokenizer.create()
    return tokenizer


def create_dummy_configs(tmpdir):
    config_1 = {"name": "test_config_1", "model": "some_model_name"}
    config_2 = {"name": "test_config_2", "model": "KNRM"}

    # write these configs to files
    tmpdir.mkdir("dummy_dir")
    tmpdir.mkdir("dummy_dir/nested_dummy_dir_1/")
    tmpdir.mkdir("dummy_dir/nested_dummy_dir_2/")
    tmpdir.join("dummy_dir/nested_dummy_dir_1/config.json").write(json.dumps(config_1))
    tmpdir.join("dummy_dir/nested_dummy_dir_2/config.json").write(json.dumps(config_2))


def test_get_config_from_results(tmpdir, monkeypatch):
    monkeypatch.setenv("CAPREOLUS_RESULTS", tmpdir.strpath)
    create_dummy_configs(tmpdir)
    configs = ConfigsView.get_config_from_results()
    config_set = set(tuple(sorted(d.items())) for d in configs)
    expected_configs = [{"name": "test_config_1", "model": "some_model_name"}, {"name": "test_config_2", "model": "KNRM"}]
    expected_config_set = set(tuple(sorted(d.items())) for d in expected_configs)

    assert config_set == expected_config_set


def test_get_available_indices(tmpdir, monkeypatch):
    monkeypatch.setenv("CAPREOLUS_CACHE", tmpdir.strpath)
    tmpdir.mkdir("nested_dir_1")
    tmpdir.mkdir("nested_dir_1/index")
    tmpdir.mkdir("nested_dir_2")
    tmpdir.mkdir("nested_dir_2/index")
    tmpdir.join("nested_dir_2/index/done").write(json.dumps({1: 2}))  # Just create the file
    available_indices = ConfigsView.get_available_indices()
    expected_indices = [tmpdir.join("nested_dir_2/index").strpath]

    assert available_indices == expected_indices


def test_query_view_do_bm25_query_two_docs(trec_index):
    doc_ids, docs = BM25View.do_query("world", trec_index, 4)
    assert doc_ids == ["LA010189-0001", "LA010189-0002"]
    assert len(docs) == 2


def test_query_view_get_tokens_from_docs_and_query(trec_index, anserini_tokenizer):
    query_string = "world"
    _, docs = BM25View.do_query("world", trec_index, 1)
    all_tokens = NeuralQueryView.get_tokens_from_docs_and_query(anserini_tokenizer, docs, query_string)
    expected_tokens = [
        "dummy",
        "dummy",
        "dummy",
        "hello",
        "world",
        "greetings",
        "from",
        "outer",
        "space",
        "dummy",
        "dummy",
        "dummy",
        "hello",
        "world",
        "greetings",
        "from",
        "outer",
        "space",
        "world",
    ]

    assert all_tokens == expected_tokens


def test_query_view_create_tensor_from_docs(trec_index, anserini_tokenizer, embedding_holder):
    query_string = "world"
    _, docs = BM25View.do_query("world", trec_index, 3)
    all_tokens = NeuralQueryView.get_tokens_from_docs_and_query(anserini_tokenizer, docs, query_string)

    embedding_holder.create_indexed_embedding_layer_from_tokens(all_tokens)

    # Limiting max doc len to 5
    doc_features = NeuralQueryView.create_tensor_from_docs(docs, anserini_tokenizer, embedding_holder, 5)

    # Equivalent to ["dummy", "dummy", "dummy", "hello", "world"] repeated twice
    expected_features = torch.from_numpy(np.array([[1, 1, 1, 2, 3], [1, 1, 1, 2, 3]]))
    assert torch.all(torch.eq(doc_features, expected_features))


def test_query_view_create_tensor_from_query_string(trec_index, anserini_tokenizer, embedding_holder):
    query_string = "world"

    # embedding_holder.stoi would be: {"<pad>": 0, "world": 1}
    embedding_holder.create_indexed_embedding_layer_from_tokens(["world"])
    query_features, query_idf = NeuralQueryView.create_tensor_from_query_string(
        query_string, trec_index, anserini_tokenizer, embedding_holder, 3, 3
    )
    expected_query_features = torch.stack([torch.from_numpy(np.array([1, 0, 0])) for i in range(0, 3)])

    assert torch.all(torch.eq(query_features, expected_query_features))


def test_query_view_get_most_relevant_doc(trec_index, anserini_tokenizer, embedding_holder, mocker):
    @property
    def mock_qrels(collection, *args):
        collection._qrels = {"q_s1": {"doc_1": "LA010189-0001"}}
        return collection._qrels

    @property
    def mock_topics(collection, *args):
        collection._topics = {"title": {"q_1": "Dummy Doc"}}
        return collection._topics

    mocker.patch.object(Collection, "qrels", mock_qrels)
    mocker.patch.object(Collection, "topics", mock_topics)

    query_string = "world"
    _, docs = BM25View.do_query("world", trec_index, 5)
    all_tokens = NeuralQueryView.get_tokens_from_docs_and_query(anserini_tokenizer, docs, query_string)
    embedding_holder.create_indexed_embedding_layer_from_tokens(all_tokens)

    config = {
        "maxdoclen": 10,
        "maxqlen": 5,
        "gradkernels": True,
        "singlefc": True,
        "scoretanh": False,
        "pad_token": 0,
        "batch": 3,
    }

    pipeline = Pipeline(dict())
    model_class = KNRM

    result_dicts = NeuralQueryView.do_query(
        config, query_string, pipeline, trec_index, anserini_tokenizer, embedding_holder, model_class
    )

    expected = [
        {"doc_id": "LA010189-0002", "doc": "Dummy Dummy Dummy Hello world, greetings from outer space!", "relevance": 0},
        {"doc_id": "LA010189-0001", "doc": "Dummy Dummy Dummy Hello world, greetings from outer space!", "relevance": 0},
    ]
    assert set([tuple(x.items()) for x in result_dicts]) == set([tuple(x.items()) for x in expected])

def test_get_two_configs_from_query_dict():
    config = MultiValueDict({"model": ["KNRM", "DRMM"], "target_index": ["dummy_index"]})
    config_1, config_2 = QueryDictParserMixin.get_two_configs_from_query_dict(config)

    assert config_1 == {"model": "KNRM", "target_index": "dummy_index"}

    assert config_2 == {"model": "DRMM", "target_index": "dummy_index"}


def test_combine_results():
    results_1 = [
        {"doc_id": "1", "doc": "foo bar", "relevance": 0},
        {"doc_id": "2", "doc": "foo bar 2", "relevance": 1},
        {"doc_id": "3", "doc": "foo bar 2", "relevance": 1},
    ]

    results_2 = [
        {"doc_id": "2", "doc": "foo bar 2", "relevance": 1},
        {"doc_id": "4", "doc": "foo bar 2", "relevance": 0},
        {"doc_id": "1", "doc": "foo bar", "relevance": 0},
        {"doc_id": "3", "doc": "foo bar 2", "relevance": 1},
    ]

    combined_results = CompareNeuralQueryView.combine_results(results_1, results_2)
    expected_results = {
        "1": {"config_1_rank": 1, "config_2_rank": 3, "relevance": 0},
        "2": {"config_1_rank": 2, "config_2_rank": 1, "relevance": 1},
        "3": {"config_1_rank": 3, "config_2_rank": 4, "relevance": 1},
        "4": {"config_1_rank": None, "config_2_rank": 2, "relevance": 0},
    }

    assert combined_results == expected_results
