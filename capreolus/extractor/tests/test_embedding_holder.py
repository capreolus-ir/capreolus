import numpy as np
import pytest
from pymagnitude import Magnitude
from pytest_mock import mocker

from capreolus.extractor.embedding import EmbeddingHolder


def test_get_instance(monkeypatch, mocker):
    # To prevent pymagnitude from downloading features during this test
    mocker.patch.object(Magnitude, "__init__")
    Magnitude.__init__.return_value = None

    global counter
    counter = 0

    def increment_counter():
        global counter
        counter += 1

    monkeypatch.setattr(EmbeddingHolder, "__init__", lambda x, y: increment_counter())
    monkeypatch.setattr(EmbeddingHolder, "__init__", lambda x, y: increment_counter())
    assert counter == 0
    EmbeddingHolder.get_instance("foo")
    assert counter == 1
    EmbeddingHolder.get_instance("foo")
    assert counter == 1
    EmbeddingHolder.get_instance("bar")
    assert counter == 2


def test_create_indexed_embedding_layer_from_tokens_and_get_index_array_from_tokens(mocker):
    # To prevent pymagnitude from downloading features during this test
    mocker.patch.object(Magnitude, "__init__")
    Magnitude.__init__.return_value = None

    embedding_holder = EmbeddingHolder.get_instance("glove6b")
    embedding_holder.embedding = lambda: None
    embedding_holder.embedding.dim = 3

    # Mocking features. Embedding for a token = it's length. There are 3 dims
    embedding_holder.embedding.query = lambda x: [[len(y), len(y), len(y)] for y in x]

    indexed_embedding = embedding_holder.create_indexed_embedding_layer_from_tokens(["my", "ohh", "my"])

    # The "<pad>" token will be embedded as [0, 0, 0]
    expected_embedding = np.array([[0, 0, 0], [2, 2, 2], [3, 3, 3]])
    assert np.array_equal(expected_embedding, indexed_embedding)

    index_array = embedding_holder.get_index_array_from_tokens(["ohh", "my"], 5)
    expected_index_array = np.array([2, 1, 0, 0, 0])
    assert np.array_equal(index_array, expected_index_array)
