import numpy as np
from pymagnitude import Magnitude, MagnitudeUtils

from capreolus import constants, get_logger

logger = get_logger(__name__)

embedding_paths = {
    "glove6b": "glove/light/glove.6B.300d",
    "glove6b.50d": "glove/light/glove.6B.50d",
    "w2vnews": "word2vec/light/GoogleNews-vectors-negative300",
    "fasttext": "fasttext/light/wiki-news-300d-1M-subword",
}

pad_tok = "<pad>"


def load_pretrained_embeddings(embedding_name):
    if embedding_name not in embedding_paths:
        raise ValueError(f"embedding name '{embedding_name}' is not a recognized embedding: {sorted(embedding_paths.keys())}")

    embedding_cache = constants["CACHE_BASE_PATH"] / "embeddings"
    numpy_cache = embedding_cache / (embedding_name + ".npy")
    vocab_cache = embedding_cache / (embedding_name + ".vocab.txt")

    if numpy_cache.exists() and vocab_cache.exists():
        logger.debug("loading embeddings from %s", numpy_cache)
        stoi, itos = load_vocab_file(vocab_cache)
        embeddings = np.load(numpy_cache, mmap_mode="r").reshape(len(stoi), -1)

        return embeddings, itos, stoi

    logger.debug("preparing embeddings and vocab")
    magnitude = Magnitude(MagnitudeUtils.download_model(embedding_paths[embedding_name], download_dir=embedding_cache))

    terms, vectors = zip(*((term, vector) for term, vector in magnitude))
    pad_vector = np.zeros(magnitude.dim, dtype=np.float32)
    terms = [pad_tok] + list(terms)
    vectors = np.array([pad_vector] + list(vectors), dtype=np.float32)
    itos = {idx: term for idx, term in enumerate(terms)}

    logger.debug("saving embeddings to %s", numpy_cache)
    np.save(numpy_cache, vectors, allow_pickle=False)
    save_vocab_file(itos, vocab_cache)
    stoi = {s: i for i, s in itos.items()}

    return vectors, itos, stoi


def load_vocab_file(fn):
    stoi, itos = {}, {}
    with open(fn, "rt") as f:
        for idx, line in enumerate(f):
            term = line.strip()
            stoi[term] = idx
            itos[idx] = term

    assert itos[0] == pad_tok
    return stoi, itos


def save_vocab_file(itos, fn):
    with open(fn, "wt") as outf:
        for idx, term in sorted(itos.items()):
            print(term, file=outf)
