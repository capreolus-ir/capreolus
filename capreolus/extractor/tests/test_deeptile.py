import os
from collections import defaultdict

import nltk
from nltk import TextTilingTokenizer
from pymagnitude import Magnitude

from capreolus.benchmark.robust04 import Robust04Benchmark
from capreolus.collection import Collection
from capreolus.extractor.deeptileextractor import DeepTileExtractor
from capreolus.searcher.bm25 import BM25Grid
from capreolus.tests.common_fixtures import trec_index, dummy_collection_config


nltk.download("stopwords")


def test_extract_segment_long_text(tmpdir):
    # nltk.TextTilingTokenizer only works with large blobs of text
    ttt = TextTilingTokenizer(k=6)
    pipeline_config = {"passagelen": 30, "slicelen": 20, "tfchannel": True}
    extractor = DeepTileExtractor(tmpdir, tmpdir, pipeline_config)

    # blob of text with Shakespeare and Shangri La. Should split into two topics
    s = (
        "O that we now had here but one ten thousand of those men in England That do no work to-day. Whats he that "
        "wishes so? My cousin, Westmorland? No, my fair cousin. If we are marked to die, we are enough To do our "
        "country loss; and if to live, The fewer men, the greater share of honour. Gods will! I pray thee, wish"
        " not one man more. Shangri-La is a fictional place described in the 1933 novel Lost Horizon "
        "by British author James Hilton. Hilton describes Shangri-La as a mystical, harmonious valley, gently guided "
        "from a lamasery, enclosed in the western end of the Kunlun Mountains. Shangri-La has become synonymous with "
        "any earthly paradise, particularly a mythical Himalayan utopia – a permanently happy land, isolated from "
        "the world"
    )
    doc_toks = s.split(" ")
    segments = extractor.extract_segment(doc_toks, ttt)
    assert len(segments) == 2

    # The split was determined by nltk.TextTilingTokenizer. Far from perfect
    assert segments == [
        "O that we now had here but one ten thousand of those men in England That do no work to-day. Whats he that wishes so? My cousin, Westmorland? No, my fair cousin. If we are marked to die, we are",
        " enough To do our country loss; and if to live, The fewer men, the greater share of honour. Gods will! I pray thee, wish not one man more. Shangri-La is a fictional place described in the 1933 novel Lost Horizon by British author James Hilton. Hilton describes Shangri-La as a mystical, harmonious valley, gently guided from a lamasery, enclosed in the western end of the Kunlun Mountains. Shangri-La has become synonymous with any earthly paradise, particularly a mythical Himalayan utopia – a permanently happy land, isolated from the world",
    ]


def test_extract_segment_short_text(tmpdir):
    # The text is too short for TextTilingTokenizer. Test if the fallback works
    ttt = TextTilingTokenizer(k=6)
    pipeline_config = {"passagelen": 30, "slicelen": 20, "tfchannel": True}
    extractor = DeepTileExtractor(tmpdir, tmpdir, pipeline_config)
    s = "But we in it shall be rememberèd We few, we happy few, we band of brothers"
    doc_toks = s.split(" ")
    segments = extractor.extract_segment(doc_toks, ttt)
    assert len(segments) == 1
    # N.B - segments are in all lowercase, special chars (comma) have been removed
    assert segments == ["But we in it shall be rememberèd We few, we happy few, we band of brothers"]

    s = (
        "But we in it shall be rememberèd We few, we happy few, we band of brothers. For he to-day that sheds his "
        "blood with me Shall be my brother"
    )
    doc_toks = s.split(" ")

    segments = extractor.extract_segment(doc_toks, ttt)
    assert len(segments) == 2
    assert segments == [
        "But we in it shall be rememberèd We few, we happy few, we band of brothers. For he to-day that",
        "sheds his blood with me Shall be my brother",
    ]


def test_clean_segments(tmpdir):
    extractor = DeepTileExtractor(tmpdir, tmpdir, {"tfchannel": True})
    assert extractor.clean_segments(["hello world", "foo bar"], p_len=4) == [
        "hello world",
        "foo bar",
        extractor.pad_tok,
        extractor.pad_tok,
    ]
    assert extractor.clean_segments(["hello world", "foo bar", "alice", "bob"], p_len=3) == ["hello world", "foo bar", "alicebob"]


def test_create_visualization_matrix(monkeypatch, tmpdir):
    pipeline_config = {"maxqlen": 5, "passagelen": 3, "slicelen": 20, "tfchannel": True}
    extractor = DeepTileExtractor(tmpdir, tmpdir, pipeline_config)
    extractor.stoi = {"<pad>": 0, "hello": 1, "world": 2, "foo": 3, "bar": 4, "alice": 5, "bob": 6}
    extractor.idf = defaultdict(lambda: 0)

    def fake_magnitude_embedding(*args, **kwargs):
        return Magnitude(None)

    monkeypatch.setattr(extractor, "get_magnitude_embeddings", fake_magnitude_embedding)
    embeddings_matrix = extractor.create_embedding_matrix("glove6b.50d")
    level = extractor.create_visualization_matrix(
        ["hello", extractor.pad_tok, extractor.pad_tok, extractor.pad_tok, extractor.pad_tok],
        ["hello world", "foo bar", "alice bob"],
        embeddings_matrix,
    )
    assert level.shape == (1, 5, 3, 3)


def test_build_from_benchmark(monkeypatch, tmpdir, trec_index, dummy_collection_config):
    # Kind of a useless test - not asserting much here. Still useful since it makes sure that the code at least runs
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
        "passagelen": 3,
        "slicelen": 20,
        "tfchannel": True,
    }
    bm25_run = BM25Grid(trec_index, collection, os.path.join(tmpdir, "searcher"), pipeline_config)
    bm25_run.create()
    folds = {"s1": {"train_qids": ["301"], "predict": {"dev": ["301"], "test": ["301"]}}}
    benchmark = Robust04Benchmark(bm25_run, collection, pipeline_config)
    benchmark.create_and_store_train_and_pred_pairs(folds)

    extractor = DeepTileExtractor(tmpdir, tmpdir, pipeline_config, index=trec_index, collection=collection, benchmark=benchmark)

    def fake_magnitude_embedding(*args, **kwargs):
        return Magnitude(None)

    monkeypatch.setattr(extractor, "get_magnitude_embeddings", fake_magnitude_embedding)
    extractor.build_from_benchmark(True)
    assert extractor.stoi == {
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

    assert extractor.itos == {v: k for k, v in extractor.stoi.items()}
