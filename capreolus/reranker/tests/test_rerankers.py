import pytest
import torch

from capreolus.reranker.POSITDRMM import POSITDRMM
from capreolus.reranker.KNRM import KNRM


def test_validate_params_for_knrm():
    with pytest.raises(ValueError):
        KNRM.validate_params({"foo": "bar"})

    with pytest.raises(ValueError):
        KNRM.validate_params({"pad_token": 0})

    config = {"pad_token": 0, "gradkernels": True, "singlefc": False, "scoretanh": True}
    KNRM.validate_params(config)


def test_positdrmm_get_exact_match_count():
    query = torch.tensor([1, 2, 3])
    doc = torch.tensor([1, 5, 3, 2, 1, 1, 9])
    query_idf = [0.5, 0.5, 0.5]

    exact_count, exact_count_idf = POSITDRMM.get_exact_match_count(query, doc, query_idf)
    assert exact_count == 5 / len(doc)
    assert exact_count_idf == (3 * 0.5 + 0.5 + 0.5) / len(doc)


def test_positdrmm_get_bigrams():
    # Each number in the doc represents an index into the vocabulary
    doc = torch.tensor([1, 2, 3, 4])

    doc_bigrams = POSITDRMM.get_bigrams(doc)
    expected_doc_bigrams = torch.tensor([[1, 2], [2, 3], [3, 4]])

    assert torch.all(torch.eq(doc_bigrams, expected_doc_bigrams))


def test_positdrmm_get_bigram_match_count():
    doc = torch.tensor([1, 2, 3, 4, 1, 2])
    query = torch.tensor([1, 5, 9, 3, 4])

    bigram_match_count = POSITDRMM.get_bigram_match_count(query, doc)
    expected_count = 1 / 5  # The only matching bigram is [3, 4], and length of doc bigrams is 5

    assert bigram_match_count == expected_count


def test_positdrmm_get_exact_match_stats():
    # 3 docs, zero padded at the end
    docs = torch.tensor([[1, 2, 3, 4, 0], [2, 3, 1, 5, 0], [3, 4, 5, 6, 0]])
    # 1 query repeated 3 times (i.e, batch size = 3), zero padded at the end
    queries = torch.tensor([[3, 1, 5, 7, 0], [3, 1, 5, 7, 0], [3, 1, 5, 7, 0]])
    query_idf = torch.tensor([[0.5, 0.5, 0.5, 0.5, 0], [0.5, 0.5, 0.5, 0.5, 0], [0.5, 0.5, 0.5, 0.5, 0]])

    exact_matches, exact_match_idf, bigram_matches = POSITDRMM.get_exact_match_stats(query_idf, queries, docs)

    assert torch.all(torch.eq(exact_matches.reshape(3), torch.tensor([2 / 4, 3 / 4, 2 / 4])))
    assert torch.all(torch.eq(exact_match_idf.reshape(3), torch.tensor([2 * 0.5 / 4, 3 * 0.5 / 4, 2 * 0.5 / 4])))

    # The query bigrams are:
    # [[3, 1], [1, 5], [5, 7], [7, 0]] - we don't clean the query
    # The doc bigrams are:
    # [[1, 2], [2, 3], [3, 4]]
    # [[2, 3], [3, 1], [1, 5]]
    # [[3, 4], [4, 5], [5, 6]]
    assert torch.all(torch.eq(bigram_matches.reshape(3), torch.tensor([0, 2 / 3, 0])))
