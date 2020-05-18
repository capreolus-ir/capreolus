import os
from pathlib import Path

import pytest
import torch
from pymagnitude import Magnitude

from capreolus.benchmark import DummyBenchmark
from capreolus.extractor import EmbedText, BertText
from capreolus.reranker.PACRR import PACRR
from capreolus.sampler import TrainDataset, PredDataset
from capreolus.tests.common_fixtures import tmpdir_as_cache, dummy_index
from capreolus.tokenizer import AnseriniTokenizer
from capreolus.trainer import PytorchTrainer
from capreolus.extractor.bagofwords import BagOfWords
from capreolus.reranker.DSSM import DSSM
from capreolus.reranker.TK import TK
from capreolus.reranker.KNRM import KNRM
from capreolus.reranker.TFKNRM import TFKNRM
from capreolus.trainer import TensorFlowTrainer
from capreolus.reranker.TFVanillaBert import TFVanillaBERT
from capreolus.tokenizer import BertTokenizer
from capreolus.extractor.deeptileextractor import DeepTileExtractor
from capreolus.reranker.DeepTileBar import DeepTileBar


def test_knrm_pytorch(dummy_index, tmpdir, tmpdir_as_cache, monkeypatch):
    def fake_magnitude_embedding(*args, **kwargs):
        return Magnitude(None)

    monkeypatch.setattr(EmbedText, "_get_pretrained_emb", fake_magnitude_embedding)

    reranker = KNRM({"gradkernels": True, "scoretanh": False, "singlefc": True, "finetune": False})
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
            "interactive": False,
            "fastforward": True,
            "validatefreq": 1,
            "usecache": False,
            "boardname": "default",
        }
    )
    reranker.modules["trainer"] = trainer
    reranker.modules["extractor"] = EmbedText(
        {
            "_name": "embedtext",
            "embeddings": "glove6b",
            "zerounk": False,
            "calcidf": True,
            "maxqlen": 4,
            "maxdoclen": 800,
            "usecache": False,
        }
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


def test_knrm_tf(dummy_index, tmpdir, tmpdir_as_cache, monkeypatch):
    def fake_magnitude_embedding(*args, **kwargs):
        return Magnitude(None)

    monkeypatch.setattr(EmbedText, "_get_pretrained_emb", fake_magnitude_embedding)

    reranker = TFKNRM({"gradkernels": True, "finetune": False})
    trainer = TensorFlowTrainer(
        {
            "_name": "tensorflow",
            "maxdoclen": 800,
            "maxqlen": 4,
            "batch": 2,
            "niters": 1,
            "itersize": 64,
            "gradacc": 1,
            "lr": 0.001,
            "softmaxloss": True,
            "interactive": False,
            "fastforward": True,
            "validatefreq": 1,
            "usecache": False,
            "tpuname": None,
            "tpuzone": None,
            "storage": None,
            "boardname": "default",
        }
    )
    reranker.modules["trainer"] = trainer
    reranker.modules["extractor"] = EmbedText(
        {
            "_name": "embedtext",
            "embeddings": "glove6b",
            "zerounk": False,
            "calcidf": True,
            "maxqlen": 4,
            "maxdoclen": 800,
            "usecache": False,
        }
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

    assert os.path.exists(Path(tmpdir) / "train" / "dev.best.index")


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
            "interactive": False,
            "fastforward": True,
            "validatefreq": 1,
            "boardname": "default",
        }
    )
    reranker.modules["trainer"] = trainer
    reranker.modules["extractor"] = EmbedText(
        {
            "_name": "embedtext",
            "embeddings": "glove6b",
            "zerounk": False,
            "calcidf": True,
            "maxqlen": 4,
            "maxdoclen": 800,
            "usecache": False,
        }
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


def test_dssm_unigram(dummy_index, tmpdir, tmpdir_as_cache, monkeypatch):
    def fake_magnitude_embedding(*args, **kwargs):
        return Magnitude(None)

    monkeypatch.setattr(EmbedText, "_get_pretrained_emb", fake_magnitude_embedding)

    reranker = DSSM({"nhiddens": "56", "lr": 0.0001})
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
            "interactive": False,
            "fastforward": True,
            "validatefreq": 1,
            "boardname": "default",
        }
    )
    reranker.modules["trainer"] = trainer
    reranker.modules["extractor"] = BagOfWords(
        {"_name": "bagofwords", "datamode": "unigram", "keepstops": True, "maxqlen": 4, "maxdoclen": 800, "usecache": False}
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

    reranker = TK(
        {
            "gradkernels": True,
            "scoretanh": False,
            "singlefc": True,
            "projdim": 32,
            "ffdim": 100,
            "numlayers": 2,
            "numattheads": 4,
            "alpha": 0.5,
            "usemask": False,
            "usemixer": True,
            "finetune": True,
        }
    )
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
            "interactive": False,
            "fastforward": False,
            "validatefreq": 1,
            "boardname": "default",
        }
    )
    reranker.modules["trainer"] = trainer
    reranker.modules["extractor"] = EmbedText(
        {
            "_name": "embedtext",
            "embeddings": "glove6b",
            "zerounk": False,
            "calcidf": True,
            "maxqlen": 4,
            "maxdoclen": 800,
            "usecache": False,
        }
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


def test_tk_get_mask(tmpdir, dummy_index, monkeypatch):
    def fake_magnitude_embedding(*args, **kwargs):
        return Magnitude(None)

    monkeypatch.setattr(EmbedText, "_get_pretrained_emb", fake_magnitude_embedding)

    reranker = TK(
        {
            "gradkernels": True,
            "scoretanh": False,
            "singlefc": True,
            "projdim": 16,
            "ffdim": 100,
            "numlayers": 1,
            "numattheads": 2,
            "alpha": 0.5,
            "usemask": True,
            "usemixer": True,
            "finetune": True,
        }
    )
    reranker.modules["extractor"] = EmbedText(
        {
            "_name": "embedtext",
            "embeddings": "glove6b",
            "zerounk": False,
            "calcidf": True,
            "maxqlen": 4,
            "maxdoclen": 800,
            "usecache": False,
            "fastforward": True,
            "validatefreq": 1,
        }
    )
    extractor = reranker.modules["extractor"]
    extractor.modules["index"] = dummy_index
    tok_cfg = {"_name": "anserini", "keepstops": True, "stemmer": "none"}
    tokenizer = AnseriniTokenizer(tok_cfg)
    extractor.modules["tokenizer"] = tokenizer
    benchmark = DummyBenchmark({"fold": "s1", "rundocsonly": True})

    extractor.create(["301"], ["LA010189-0001", "LA010189-0002"], benchmark.topics[benchmark.query_type])
    reranker.build()

    # 3 batches, each of seq len 4, and embedding dim 8
    embedding = torch.ones(3, 4, 8)
    # Set the 2nd and last token in first batch as pad
    embedding[0, 1] = torch.zeros(8)
    embedding[0, 3] = torch.zeros(8)

    # set the first and third token in second batch as pad
    embedding[1, 0] = torch.zeros(8)
    embedding[1, 2] = torch.zeros(8)

    mask = reranker.model.get_mask(embedding)

    assert torch.equal(
        mask[0],
        torch.tensor(
            [
                [0, float("-inf"), 0, float("-inf")],
                [float("-inf"), float("-inf"), float("-inf"), float("-inf")],
                [0, float("-inf"), 0, float("-inf")],
                [float("-inf"), float("-inf"), float("-inf"), float("-inf")],
            ],
            dtype=torch.float,
        ),
    )

    assert torch.equal(
        mask[1],
        torch.tensor(
            [
                [float("-inf"), float("-inf"), float("-inf"), float("-inf")],
                [float("-inf"), 0, float("-inf"), 0],
                [float("-inf"), float("-inf"), float("-inf"), float("-inf")],
                [float("-inf"), 0, float("-inf"), 0],
            ],
            dtype=torch.float,
        ),
    )

    assert torch.equal(mask[2], torch.zeros(4, 4, dtype=torch.float))


def test_deeptilebars(dummy_index, tmpdir, tmpdir_as_cache, monkeypatch):
    def fake_magnitude_embedding(*args, **kwargs):
        return Magnitude(None)

    monkeypatch.setattr(DeepTileExtractor, "_get_pretrained_emb", fake_magnitude_embedding)
    reranker_config = {
        "name": "DeepTileBar",
        "passagelen": 30,
        "numberfilter": 3,
        "lstmhiddendim": 3,
        "linearhiddendim1": 32,
        "linearhiddendim2": 16,
    }
    reranker = DeepTileBar(reranker_config)
    trainer = PytorchTrainer(
        {
            "maxdoclen": 800,
            "maxqlen": 4,
            "batch": 2,
            "niters": 1,
            "itersize": 512,
            "gradacc": 1,
            "lr": 0.001,
            "softmaxloss": True,
            "interactive": False,
            "fastforward": True,
            "validatefreq": 1,
            "boardname": "default",
        }
    )
    reranker.modules["trainer"] = trainer
    reranker.modules["extractor"] = DeepTileExtractor(
        {
            "_name": "deeptiles",
            "embeddings": "glove6b",
            "tfchannel": True,
            "slicelen": 20,
            "keepstops": False,
            "tilechannels": 3,
            "calcidf": False,
            "usecache": False,
            "maxqlen": 4,
            "maxdoclen": 800,
            "passagelen": 20,
        }
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
