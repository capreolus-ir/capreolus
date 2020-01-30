import os

import pytest
from pymagnitude import Magnitude
from pytest_mock import mocker
import sacred

from capreolus.benchmark.robust04 import Robust04Benchmark
from capreolus.collection import Collection
from capreolus.extractor.embedtext import EmbedText
from capreolus.index.anserini import AnseriniIndex
from capreolus.reranker.KNRM import KNRM
from capreolus.reranker.PACRR import PACRR
from capreolus.pipeline import Pipeline
from capreolus.searcher.bm25 import BM25Grid
from capreolus.utils.common import forced_types


def test_get_parameters_to_module():
    pipeline = Pipeline({})
    ex = sacred.Experiment("capreolus")

    parameters_to_module = pipeline.get_parameters_to_module(ex)
    assert parameters_to_module == {
        "collection": "module",
        "index": "module",
        "searcher": "module",
        "benchmark": "module",
        "reranker": "module",
        "expid": "stateless",
        "earlystopping": "stateless",
        "predontrain": "stateless",
        "fold": "stateless",
        "maxdoclen": "pipeline",
        "maxqlen": "pipeline",
        "batch": "pipeline",
        "niters": "pipeline",
        "itersize": "pipeline",
        "gradacc": "pipeline",
        "lr": "pipeline",
        "seed": "pipeline",
        "sample": "pipeline",
        "softmaxloss": "pipeline",
        "dataparallel": "pipeline",
    }


def test_get_parameter_types(mocker):
    pipeline = Pipeline({})
    ex = sacred.Experiment("capreolus")

    def mock_config(method_that_generates_input_dict):
        input_dict = method_that_generates_input_dict()

        # Just messing with the types to make sure that get_parameter_types does what it should
        input_dict.update({"index": None, "niters": True})
        return lambda: input_dict

    mocker.patch.object(ex, "config", mock_config)
    parameter_types = pipeline.get_parameter_types(ex)
    assert parameter_types == {
        "pipeline": type("string"),  # "pipeline" key is added by the method
        "collection": type("robust04"),
        "earlystopping": forced_types[type(True)],
        "index": forced_types[type(None)],
        "searcher": type("bm25grid"),
        "benchmark": type("robust04.title.wsdm20demo"),
        "reranker": type("PACRR"),
        "expid": type("debug"),
        "predontrain": forced_types[type(True)],
        "fold": type("s1"),
        "maxdoclen": type(800),
        "maxqlen": type(4),
        "batch": type(32),
        "niters": forced_types[type(True)],
        "itersize": type(4096),
        "gradacc": type(1),
        "lr": type(0.001),
        "seed": type(123_456),
        "sample": type("simple"),
        "softmaxloss": forced_types[type(True)],
        "dataparallel": type("none"),
    }


def test_get_module_to_class():
    pipeline = Pipeline({})
    module_choices = {"reranker": "KNRM"}  # default is PACRR

    module2class = pipeline.get_module_to_class({})
    assert module2class["collection"].__class__ == Collection
    assert module2class["index"].__class__ == AnseriniIndex.__class__
    assert module2class["searcher"].__class__ == BM25Grid.__class__
    assert module2class["benchmark"].__class__ == Robust04Benchmark.__class__
    assert module2class["reranker"].__class__ == PACRR.__class__

    module2class = pipeline.get_module_to_class(module_choices)
    assert module2class["reranker"].__class__ == KNRM.__class__


def test_get_parameters_to_module_including_missing_and_extractors():
    """
        Calls Pipeline.__init__() which in turn calls
        1. self.get_parameters_to_module
        2. get_parameters_to_module_for_missing_parameters
        3. get_parameters_to_module_for_feature_parameters
    """
    pipeline = Pipeline({})
    ex = sacred.Experiment("capreolus")

    # parameters_to_module, parameter_types = pipeline.get_parameters_to_module_for_missing_parameters(ex)

    assert pipeline.parameters_to_module == {
        "collection": "module",
        "benchmark": "module",
        "reranker": "module",
        "expid": "stateless",
        "predontrain": "stateless",
        "earlystopping": "stateless",
        "maxdoclen": "pipeline",
        "maxqlen": "pipeline",
        "batch": "pipeline",
        "niters": "pipeline",
        "itersize": "pipeline",
        "gradacc": "pipeline",
        "lr": "pipeline",
        "seed": "pipeline",
        "sample": "pipeline",
        "softmaxloss": "pipeline",
        "dataparallel": "pipeline",
        # AnseriniIndex specific config
        "stemmer": "index",
        "indexstops": "index",
        # BM25Grid specific config
        "index": "module",
        # Robust04Benchmark specific config
        "fold": "stateless",
        "searcher": "module",
        "rundocsonly": "benchmark",
        # PACRR specific config
        "mingram": "reranker",
        "maxgram": "reranker",
        "nfilters": "reranker",
        "idf": "reranker",
        "kmax": "reranker",
        "combine": "reranker",
        "nonlinearity": "reranker",
        # EmbedText specific config
        "embeddings": "extractor",
        "keepstops": "extractor",
    }


def test_check_for_invalid_keys():
    pipeline = Pipeline({})
    ex = sacred.Experiment("capreolus")
    pipeline.check_for_invalid_keys()

    pipeline.parameters_to_module["foo_bar"] = "reranker"

    with pytest.raises(ValueError):
        pipeline.check_for_invalid_keys()
