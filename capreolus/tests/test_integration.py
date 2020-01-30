"""
Integrations tests. Trains different rerankers from scratch.
"""
import json
import os

import pytest
from pymagnitude import Magnitude

from capreolus import train
from capreolus import train_pipeline, evaluate_pipeline
from capreolus.collection import COLLECTIONS
from capreolus.demo_app.utils import search_files_or_folders_in_directory
from capreolus.extractor.deeptileextractor import DeepTileExtractor
from capreolus.extractor.embedtext import EmbedText
from capreolus.pipeline import Pipeline
from capreolus.reranker.KNRM import KNRM
from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)  # pylint: disable=invalid-name


def test_knrm(monkeypatch, tmpdir):
    monkeypatch.setenv("CAPREOLUS_RESULTS", str(os.path.join(tmpdir, "results")))
    monkeypatch.setenv("CAPREOLUS_CACHE", str(os.path.join(tmpdir, "cache")))

    def fake_magnitude_embedding(*args, **kwargs):
        return Magnitude(None)

    pipeline = Pipeline({"reranker": "KNRM", "niters": 1, "benchmark": "dummy", "itersize": 1, "batch": 1})
    pipeline.ex.main(train.train)
    monkeypatch.setattr(train, "pipeline", pipeline)
    monkeypatch.setattr(EmbedText, "get_magnitude_embeddings", fake_magnitude_embedding)
    pipeline.ex.run(config_updates={"reranker": "KNRM", "niters": 1, "benchmark": "dummy", "itersize": 1, "batch": 1})
    logger.info("Base path is {0}".format(pipeline.base_path))

    config_files = search_files_or_folders_in_directory(pipeline.base_path, "config.json")
    assert len(config_files) == 1
    config_file = json.load(open(config_files[0], "rt"))
    assert config_file["reranker"] == "KNRM"
    assert config_file["niters"] == 1

    run_path = os.path.join(pipeline.reranker_path, pipeline.cfg["fold"])
    weight_dir = os.path.join(run_path, "weights")
    weight_file = search_files_or_folders_in_directory(weight_dir, "dev")
    assert len(weight_file) == 1


def test_convknrm(monkeypatch, tmpdir):
    monkeypatch.setenv("CAPREOLUS_RESULTS", str(os.path.join(tmpdir, "results")))
    monkeypatch.setenv("CAPREOLUS_CACHE", str(os.path.join(tmpdir, "cache")))

    def fake_magnitude_embedding(*args, **kwargs):
        return Magnitude(None)

    pipeline = Pipeline({"reranker": "ConvKNRM", "niters": 1, "benchmark": "dummy", "itersize": 1, "batch": 1})
    pipeline.ex.main(train.train)
    monkeypatch.setattr(train, "pipeline", pipeline)
    monkeypatch.setattr(EmbedText, "get_magnitude_embeddings", fake_magnitude_embedding)
    pipeline.ex.run(config_updates={"reranker": "ConvKNRM", "niters": 1, "benchmark": "dummy", "itersize": 1, "batch": 1})
    logger.info("Base path is {0}".format(pipeline.base_path))

    config_files = search_files_or_folders_in_directory(pipeline.base_path, "config.json")
    assert len(config_files) == 1
    config_file = json.load(open(config_files[0], "rt"))
    assert config_file["reranker"] == "ConvKNRM"
    assert config_file["niters"] == 1

    run_path = os.path.join(pipeline.reranker_path, pipeline.cfg["fold"])
    weight_dir = os.path.join(run_path, "weights")
    weight_file = search_files_or_folders_in_directory(weight_dir, "dev")
    assert len(weight_file) == 1


def test_drmm(monkeypatch, tmpdir):
    monkeypatch.setenv("CAPREOLUS_RESULTS", str(os.path.join(tmpdir, "results")))
    monkeypatch.setenv("CAPREOLUS_CACHE", str(os.path.join(tmpdir, "cache")))

    def fake_magnitude_embedding(*args, **kwargs):
        return Magnitude(None)

    pipeline = Pipeline({"reranker": "DRMM", "niters": 1, "benchmark": "dummy", "itersize": 1, "batch": 1})
    pipeline.ex.main(train.train)
    monkeypatch.setattr(train, "pipeline", pipeline)
    monkeypatch.setattr(EmbedText, "get_magnitude_embeddings", fake_magnitude_embedding)
    pipeline.ex.run(config_updates={"reranker": "DRMM", "niters": 1, "benchmark": "dummy", "itersize": 1, "batch": 1})
    logger.info("Base path is {0}".format(pipeline.base_path))

    config_files = search_files_or_folders_in_directory(pipeline.base_path, "config.json")
    assert len(config_files) == 1
    config_file = json.load(open(config_files[0], "rt"))
    assert config_file["reranker"] == "DRMM"
    assert config_file["niters"] == 1

    run_path = os.path.join(pipeline.reranker_path, pipeline.cfg["fold"])
    weight_dir = os.path.join(run_path, "weights")
    weight_file = search_files_or_folders_in_directory(weight_dir, "dev")
    assert len(weight_file) == 1


def test_positdrmm(monkeypatch, tmpdir):
    monkeypatch.setenv("CAPREOLUS_RESULTS", str(os.path.join(tmpdir, "results")))
    monkeypatch.setenv("CAPREOLUS_CACHE", str(os.path.join(tmpdir, "cache")))

    def fake_magnitude_embedding(*args, **kwargs):
        return Magnitude(None)

    pipeline = Pipeline({"reranker": "POSITDRMM", "niters": 1, "benchmark": "dummy", "itersize": 1, "batch": 1})
    pipeline.ex.main(train.train)
    monkeypatch.setattr(train, "pipeline", pipeline)
    monkeypatch.setattr(EmbedText, "get_magnitude_embeddings", fake_magnitude_embedding)
    pipeline.ex.run(config_updates={"reranker": "POSITDRMM", "niters": 1, "benchmark": "dummy", "itersize": 1, "batch": 1})
    logger.info("Base path is {0}".format(pipeline.base_path))

    config_files = search_files_or_folders_in_directory(pipeline.base_path, "config.json")
    assert len(config_files) == 1
    config_file = json.load(open(config_files[0], "rt"))
    assert config_file["reranker"] == "POSITDRMM"
    assert config_file["niters"] == 1

    run_path = os.path.join(pipeline.reranker_path, pipeline.cfg["fold"])
    weight_dir = os.path.join(run_path, "weights")
    weight_file = search_files_or_folders_in_directory(weight_dir, "dev")
    assert len(weight_file) == 1


def test_dssm_trigram(monkeypatch, tmpdir):
    monkeypatch.setenv("CAPREOLUS_RESULTS", str(os.path.join(tmpdir, "results")))
    monkeypatch.setenv("CAPREOLUS_CACHE", str(os.path.join(tmpdir, "cache")))

    def fake_magnitude_embedding(*args, **kwargs):
        return Magnitude(None)

    pipeline = Pipeline({"reranker": "DSSM", "niters": 1, "benchmark": "dummy", "itersize": 1, "batch": 1, "datamode": "trigram"})
    pipeline.ex.main(train.train)
    monkeypatch.setattr(train, "pipeline", pipeline)
    pipeline.ex.run(config_updates={"reranker": "DSSM", "niters": 1, "benchmark": "dummy", "itersize": 1, "batch": 1})

    config_files = search_files_or_folders_in_directory(pipeline.base_path, "config.json")
    assert len(config_files) == 1
    config_file = json.load(open(config_files[0], "rt"))
    assert config_file["reranker"] == "DSSM"
    assert config_file["niters"] == 1

    run_path = os.path.join(pipeline.reranker_path, pipeline.cfg["fold"])
    weight_dir = os.path.join(run_path, "weights")
    weight_file = search_files_or_folders_in_directory(weight_dir, "dev")
    assert len(weight_file) == 1


def test_deeptilebar(monkeypatch, tmpdir):
    monkeypatch.setenv("CAPREOLUS_RESULTS", str(os.path.join(tmpdir, "results")))
    monkeypatch.setenv("CAPREOLUS_CACHE", str(os.path.join(tmpdir, "cache")))

    def fake_magnitude_embedding(*args, **kwargs):
        return Magnitude(None)

    monkeypatch.setattr(DeepTileExtractor, "get_magnitude_embeddings", fake_magnitude_embedding)
    pipeline = Pipeline(
        {"reranker": "DeepTileBar", "niters": 1, "benchmark": "dummy", "itersize": 1, "batch": 1, "passagelen": "3"}
    )
    pipeline.ex.main(train.train)
    monkeypatch.setattr(train, "pipeline", pipeline)
    pipeline.ex.run(config_updates={"reranker": "DeepTileBar", "niters": 1, "benchmark": "dummy", "itersize": 1, "batch": 1})

    config_files = search_files_or_folders_in_directory(pipeline.base_path, "config.json")
    assert len(config_files) == 1
    config_file = json.load(open(config_files[0], "rt"))
    assert config_file["reranker"] == "DeepTileBar"
    assert config_file["niters"] == 1

    run_path = os.path.join(pipeline.reranker_path, pipeline.cfg["fold"])
    weight_dir = os.path.join(run_path, "weights")
    weight_file = search_files_or_folders_in_directory(weight_dir, "dev")
    assert len(weight_file) == 1


def test_train_api(monkeypatch, tmpdir):
    monkeypatch.setenv("CAPREOLUS_RESULTS", str(os.path.join(tmpdir, "results")))
    monkeypatch.setenv("CAPREOLUS_CACHE", str(os.path.join(tmpdir, "cache")))

    def fake_magnitude_embedding(*args, **kwargs):
        return Magnitude(None)

    monkeypatch.setattr(EmbedText, "get_magnitude_embeddings", fake_magnitude_embedding)
    pipeline = train_pipeline({"reranker": "KNRM", "niters": 1, "benchmark": "dummy", "itersize": 1, "batch": 1})

    assert pipeline.reranker.__class__ == KNRM

    ndcg_vals = evaluate_pipeline(pipeline)
    assert ndcg_vals == [0.6309297535714575]


def test_train_api_early_stopping(monkeypatch, tmpdir):
    monkeypatch.setenv("CAPREOLUS_RESULTS", str(os.path.join(tmpdir, "results")))
    monkeypatch.setenv("CAPREOLUS_CACHE", str(os.path.join(tmpdir, "cache")))

    def fake_magnitude_embedding(*args, **kwargs):
        return Magnitude(None)

    monkeypatch.setattr(EmbedText, "get_magnitude_embeddings", fake_magnitude_embedding)
    pipeline = train_pipeline({"reranker": "KNRM", "niters": 5, "benchmark": "dummy", "itersize": 1, "batch": 1}, early_stopping=True)

    assert pipeline.reranker.__class__ == KNRM

    ndcg_vals = evaluate_pipeline(pipeline)
    assert ndcg_vals == [0.6309297535714575]


def test_api_data_sources(monkeypatch, tmpdir):
    monkeypatch.setenv("CAPREOLUS_RESULTS", str(os.path.join(tmpdir, "results")))
    monkeypatch.setenv("CAPREOLUS_CACHE", str(os.path.join(tmpdir, "cache")))

    fake_qrels_path = os.path.join(tmpdir, "fake_qrels.txt")
    with open(fake_qrels_path, "w") as fp:
        qrels = "301 0 LA010189-0001 0\n301 0 LA010189-0002 1"
        fp.write(qrels)

    def fake_magnitude_embedding(*args, **kwargs):
        return Magnitude(None)

    monkeypatch.setattr(EmbedText, "get_magnitude_embeddings", fake_magnitude_embedding)
    pipeline = train_pipeline(
        {"reranker": "KNRM", "niters": 1, "benchmark": "dummy", "itersize": 1, "batch": 1}, {"qrels": fake_qrels_path}
    )

    assert pipeline.reranker.__class__ == KNRM

    ndcg_vals = evaluate_pipeline(pipeline)
    # The ndcg score changed since the retrieved order is now the best order
    assert ndcg_vals == [1.0]


# This test should be placed at the very end. Setting is_large_collection will mess up other tests
# TODO: Fix this? Simple fix: set dummy_collection.is_large_collection = False after the test
# def test_knrm_for_is_large_collection(monkeypatch, tmpdir):
#     monkeypatch.setenv("CAPREOLUS_RESULTS", str(os.path.join(tmpdir, "results")))
#
#     def fake_magnitude_embedding(*args, **kwargs):
#         return Magnitude(None)
#
#     pipeline = Pipeline({"reranker": "KNRM", "niters": 1, "benchmark": "dummy", "itersize": 1, "batch": 1})
#     pipeline.ex.main(train.train)
#     COLLECTIONS["dummy"].is_large_collection = True
#     monkeypatch.setattr(train, "pipeline", pipeline)
#     monkeypatch.setattr(EmbedText, "get_magnitude_embeddings", fake_magnitude_embedding)
#     pipeline.ex.run(config_updates={"reranker": "KNRM", "niters": 1, "benchmark": "dummy", "itersize": 1, "batch": 1})
#     logger.info("Is collection large? : {0}".format(pipeline.collection.is_large_collection))
#
#     config_files = search_files_or_folders_in_directory(pipeline.base_path, "config.json")
#     assert len(config_files) == 1
#     config_file = json.load(open(config_files[0], "rt"))
#     assert config_file["reranker"] == "KNRM"
#     assert config_file["niters"] == 1
#
#     run_path = os.path.join(pipeline.reranker_path, pipeline.cfg["fold"])
#     weight_dir = os.path.join(run_path, "weights")
#     weight_file = search_files_or_folders_in_directory(weight_dir, "dev")
#     assert len(weight_file) == 1
