import os
from pathlib import Path

import pytest
from pymagnitude import Magnitude

from capreolus.benchmark import DummyBenchmark
from capreolus.extractor import EmbedText
from capreolus.reranker.PACRR import PACRR
from capreolus.sampler import TrainDataset, PredDataset
from capreolus.tests.common_fixtures import tmpdir_as_cache, dummy_index
from capreolus.tokenizer import AnseriniTokenizer
from capreolus.trainer import PytorchTrainer
from reranker.TK import TK


def test_pacrr(dummy_index, tmpdir, tmpdir_as_cache, monkeypatch):
    def fake_magnitude_embedding(*args, **kwargs):
        return Magnitude(None)

    monkeypatch.setattr(EmbedText, "_get_pretrained_emb", fake_magnitude_embedding)

    reranker = PACRR({"mingram": 1, "maxgram": 3, "nfilters": 32, "idf": True, "kmax": 2, "combine": 32, "nonlinearity": "relu"})
    trainer = PytorchTrainer(
        {
            "maxdoclen": 800,
            "maxqlen": 4,
            "batch": 32,
            "niters": 1,
            "itersize": 512,
            "gradacc": 1,
            "lr": 0.001,
            "softmaxloss": True,
            "interactive": False
        }
    )
    reranker.modules["trainer"] = trainer
    reranker.modules["extractor"] = EmbedText(
        {"embeddings": "glove6b", "zerounk": False, "calcidf": True, "maxqlen": 4, "maxdoclen": 800}
    )
    extractor = reranker.modules["extractor"]
    extractor.modules["index"] = dummy_index
    tok_cfg = {"_name": "anserini", "keepstops": True, "stemmer": "none"}
    tokenizer = AnseriniTokenizer(tok_cfg)
    extractor.modules["tokenizer"] = tokenizer
    metric = "map"
    benchmark = DummyBenchmark({"fold": "s1", "rundocsonly": True})

    extractor.create(["301"], ["LA010189-0001", "LA010189-0002"], benchmark.topics[benchmark.query_type])
    reranker.build()

    train_run = {"301": ["LA010189-0001", "LA010189-0002"]}
    train_dataset = TrainDataset(qid_docid_to_rank=train_run, qrels=benchmark.qrels, extractor=extractor)
    dev_dataset = PredDataset(qid_docid_to_rank=train_run, extractor=extractor)
    reranker["trainer"].train(
        reranker, train_dataset, Path(tmpdir) / "train", dev_dataset, Path(tmpdir) / "dev", benchmark.qrels, metric
    )

    assert os.path.exists(Path(tmpdir) / "train" / "dev.best")


def test_tk(dummy_index, tmpdir, tmpdir_as_cache, monkeypatch):
    def fake_magnitude_embedding(*args, **kwargs):
        return Magnitude(None)

    monkeypatch.setattr(EmbedText, "_get_pretrained_emb", fake_magnitude_embedding)

    reranker = TK({"gradkernels": True, "scoretanh": False, "singlefc": True, "projdim": 8, "ffdim": 20, "numlayers": 2, "numattheads": 2,})
    trainer = PytorchTrainer(
        {
            "maxdoclen": 200,
            "maxqlen": 20,
            "batch": 32,
            "niters": 1,
            "itersize": 512,
            "gradacc": 1,
            "lr": 0.001,
            "softmaxloss": True,
            "interactive": False
        }
    )
    reranker.modules["trainer"] = trainer
    reranker.modules["extractor"] = EmbedText(
        {"embeddings": "glove6b", "zerounk": False, "calcidf": True, "maxqlen": 4, "maxdoclen": 800}
    )
    extractor = reranker.modules["extractor"]
    extractor.modules["index"] = dummy_index
    tok_cfg = {"_name": "anserini", "keepstops": True, "stemmer": "none"}
    tokenizer = AnseriniTokenizer(tok_cfg)
    extractor.modules["tokenizer"] = tokenizer
    metric = "map"
    benchmark = DummyBenchmark({"fold": "s1", "rundocsonly": True})

    extractor.create(["301"], ["LA010189-0001", "LA010189-0002"], benchmark.topics[benchmark.query_type])
    reranker.build()

    train_run = {"301": ["LA010189-0001", "LA010189-0002"]}
    train_dataset = TrainDataset(qid_docid_to_rank=train_run, qrels=benchmark.qrels, extractor=extractor)
    dev_dataset = PredDataset(qid_docid_to_rank=train_run, extractor=extractor)
    reranker["trainer"].train(
        reranker, train_dataset, Path(tmpdir) / "train", dev_dataset, Path(tmpdir) / "dev", benchmark.qrels, metric
    )

    assert os.path.exists(Path(tmpdir) / "train" / "dev.best")
