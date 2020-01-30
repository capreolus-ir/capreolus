import torch
import torch.nn as nn

from capreolus.extractor.embedtext import EmbedText
from capreolus.reranker.reranker import Reranker
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from capreolus.reranker.common import create_emb_layer

from nltk.tokenize import TextTilingTokenizer
import re
import numpy as np
import math
from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)  # pylint: disable=invalid-name


class POSITDRMM_basic(nn.Module):
    def __init__(self, weights_matrix, pipeline_config):
        super(POSITDRMM_basic, self).__init__()
        #        self.embedding_dim=embedding_dim
        self.embedding_dim = weights_matrix.shape[1]
        self.lstm_hidden_dim = weights_matrix.shape[1]
        self.batch_size = pipeline_config["batch"]
        self.QUERY_LENGTH = pipeline_config["maxqlen"]
        #        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = 2
        self.encoding_layer = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.embedding_dim,
            num_layers=self.lstm_num_layers,
            bidirectional=True,
            dropout=0.3,
        )
        self.pad_token = pipeline_config["pad_token"]
        self.embedding = create_emb_layer(weights_matrix, non_trainable=True)
        self.m = nn.Dropout(p=0.2)
        self.Q1 = nn.Linear(6, 1, bias=True)
        self.Wg = nn.Linear(5, 1)
        self.activation = nn.LeakyReLU()
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (
            Variable(torch.zeros(4, self.batch_size, self.lstm_hidden_dim)),
            Variable(torch.zeros(4, self.batch_size, self.lstm_hidden_dim)),
        )

    def forward(self, sentence, query_sentence, query_idf, extra):
        dtype = torch.FloatTensor
        x = self.embedding(sentence)
        query_x = self.embedding(query_sentence)
        x1 = x.norm(dim=2)[:, :, None] + 1e-7
        query_x1 = query_x.norm(dim=2)[:, :, None] + 1e-7
        x_norm = x / x1
        query_x_norm = query_x / query_x1
        M_cos = torch.matmul(query_x_norm, torch.transpose(x_norm, 1, 2))
        BAT = query_sentence.shape[0]
        A = query_sentence.shape[1]
        B = sentence.shape[1]
        nul = torch.zeros_like(M_cos)
        one = torch.ones_like(M_cos)
        XOR_matrix = torch.where(
            query_sentence.reshape(BAT, A, 1).expand(BAT, A, B) == sentence.reshape(BAT, 1, B).expand(BAT, A, B), one, nul
        )
        XOR_matrix = torch.where(query_sentence.reshape(BAT, A, 1).expand(BAT, A, B) == self.pad_token, nul, XOR_matrix)
        query_x = torch.transpose(query_x, 0, 1)
        (p1, self.hidden) = self.encoding_layer(self.m(query_x))
        p1_forward = p1[:, :, : (self.lstm_hidden_dim)]
        p1_backward = p1[:, :, (self.lstm_hidden_dim) :]
        query_context = torch.cat([p1_forward + query_x, p1_backward + query_x], dim=2)
        x = torch.transpose(x, 0, 1)
        (p1, self.hidden) = self.encoding_layer(self.m(x))
        p1_forward = p1[:, :, : (self.lstm_hidden_dim)]
        p1_backward = p1[:, :, (self.lstm_hidden_dim) :]
        doc_context = torch.cat([p1_forward + x, p1_backward + x], dim=2)
        query_context = torch.transpose(query_context, 0, 1)
        doc_context = torch.transpose(doc_context, 0, 1)
        doc_context_norm = doc_context / doc_context.norm(dim=2)[:, :, None]
        query_con_norm = query_context / query_context.norm(dim=2)[:, :, None]
        M_cos_context = torch.matmul(query_con_norm, torch.transpose(doc_context_norm, 1, 2))
        M_max = torch.randn(self.batch_size, self.QUERY_LENGTH, 6).type(dtype).to(sentence.device)
        M_max[:, :, 0], _ = torch.max(M_cos, 2)
        i, _ = torch.topk(M_cos, 5, dim=2)
        M_max[:, :, 1] = torch.sum(i, dim=2) / 5
        M_max[:, :, 2], _ = torch.max(M_cos_context, 2)
        i, _ = torch.topk(M_cos_context, 5, dim=2)
        M_max[:, :, 3] = torch.sum(i, dim=2) / 5
        M_max[:, :, 4], _ = torch.max(XOR_matrix, 2)
        i, _ = torch.topk(XOR_matrix, 5, dim=2)
        M_max[:, :, 5] = torch.sum(i, dim=2) / 5
        M_res = self.activation(self.Q1(M_max))
        M_res = M_res.view(self.batch_size, self.QUERY_LENGTH)
        mmm = nn.Softmax(dim=1)

        r = mmm(query_idf.float())
        s = r * M_res
        ans = torch.sum(s, dim=1).view(-1, 1)
        extra_ans = torch.cat([ans, extra], dim=1)
        true_ans = self.Wg(extra_ans)
        return true_ans.view(-1)


class POSITDRMM_class(nn.Module):
    def __init__(self, weights_matrix, pipeline_config):
        super(POSITDRMM_class, self).__init__()
        self.posit1 = POSITDRMM_basic(weights_matrix, pipeline_config)
        self.p = pipeline_config

    def forward(self, query_sentence, query_idf, pos_sentence, neg_sentence, posdoc_extra, negdoc_extra):
        self.posit1.hidden = self.posit1.init_hidden()
        pos_tag_scores = self.posit1(pos_sentence, query_sentence, query_idf, posdoc_extra)
        self.posit1.hidden = self.posit1.init_hidden()
        neg_tag_scores = self.posit1(neg_sentence, query_sentence, query_idf, negdoc_extra)
        return [pos_tag_scores, neg_tag_scores]

    def test_forward(self, query_sentence, query_idf, pos_sentence, extras):
        self.posit1.hidden = self.posit1.init_hidden()
        pos_tag_scores = self.posit1(pos_sentence, query_sentence, query_idf, extras)
        #        neg_tag_scores = self.posit1(neg_sentence, query_sentence, query_idf)
        #        self.posit1.hidden = self.posit1.init_hidden()
        return pos_tag_scores


dtype = torch.FloatTensor


@Reranker.register
class POSITDRMM(Reranker):
    description = """Ryan McDonald, George Brokos, and Ion Androutsopoulos. 2018. Deep Relevance Ranking Using Enhanced Document-Query Interactions. In EMNLP'18."""
    EXTRACTORS = [EmbedText]

    @staticmethod
    def config():
        # passagelen = 6
        # self.p["maxqlen"], EMBEDDING_DIM, BATCH_SIZE come from main config
        lr = 0.01
        return locals().copy()  # ignored by sacred

    @staticmethod
    def required_params():
        # Used for validation. Returns a set of params required by the class defined in get_model_class()
        return {"maxqlen", "batch", "pad_token"}

    @classmethod
    def get_model_class(cls):
        return POSITDRMM_class

    def to(self, device):
        self.model.to(device)
        self.model.posit1.to(device)
        self.device = device
        return self

    def build(self):
        config = self.config.copy()
        config["pad_token"] = EmbedText.pad
        self.model = POSITDRMM_class(self.embeddings, config)
        return self.model

    def score(self, data):
        query_idf = data["query_idf"]
        query_sentence = data["query"]
        pos_sentence, neg_sentence = data["posdoc"], data["negdoc"]
        pos_exact_matches, pos_exact_match_idf, pos_bigram_matches = self.get_exact_match_stats(
            query_idf, query_sentence, pos_sentence
        )
        neg_exact_matches, neg_exact_match_idf, neg_bigram_matches = self.get_exact_match_stats(
            query_idf, query_sentence, neg_sentence
        )
        pos_bm25 = self.get_bm25_scores(data["qid"], data["posdocid"])
        neg_bm25 = self.get_bm25_scores(data["qid"], data["negdocid"])
        posdoc_extra = torch.cat((pos_exact_matches, pos_exact_match_idf, pos_bigram_matches, pos_bm25), dim=1).to(self.device)
        negdoc_extra = torch.cat((neg_exact_matches, neg_exact_match_idf, neg_bigram_matches, neg_bm25), dim=1).to(self.device)
        return self.model(query_sentence, query_idf, pos_sentence, neg_sentence, posdoc_extra, negdoc_extra)

    def test(self, query_sentence, query_idf, pos_sentence, qids=None, posdoc_ids=None):
        assert qids is not None
        assert posdoc_ids is not None

        pos_exact_matches, pos_exact_match_idf, pos_bigram_matches = self.get_exact_match_stats(
            query_idf, query_sentence, pos_sentence
        )
        pos_bm25 = self.get_bm25_scores(qids, posdoc_ids)
        posdoc_extra = torch.cat((pos_exact_matches, pos_exact_match_idf, pos_bigram_matches, pos_bm25), dim=1).to(self.device)
        return self.model.test_forward(query_sentence, query_idf, pos_sentence, posdoc_extra)

    def zero_grad(self, *args, **kwargs):
        self.model.zero_grad(*args, **kwargs)

    def get_bm25_scores(self, qids, doc_ids):
        scores = torch.zeros((len(doc_ids)))
        for i, doc_id in enumerate(doc_ids):
            scores[i] = self.bm25_scores[qids[i]][doc_id]

        return scores.reshape(len(doc_ids), 1)

    @classmethod
    def clean(cls, text):
        """
        Remove pad tokens from the text
        """
        return text[text != EmbedText.pad]

    @classmethod
    def get_bigrams(cls, text):
        # text_batch has shape (batch_size, length)
        return torch.stack([text[i : i + 2] for i in range(len(text) - 1)])

    @classmethod
    def get_bigram_match_count(cls, query, doc):
        bigram_matches = 0
        query_bigrams = cls.get_bigrams(query)
        doc_bigrams = cls.get_bigrams(doc)

        for q_bigram in query_bigrams:
            bigram_matches += len(torch.all((doc_bigrams == q_bigram), dim=1).nonzero())
        return bigram_matches / len(doc_bigrams)

    @classmethod
    def get_exact_match_count(cls, query, single_doc, query_idf):
        exact_match_count = 0
        exact_match_idf = 0
        for i, q_word in enumerate(query):
            curr_count = len(single_doc[single_doc == q_word])
            exact_match_count += curr_count
            exact_match_idf += curr_count * query_idf[i]

        return exact_match_count / len(single_doc), exact_match_idf / len(single_doc)

    @classmethod
    def get_exact_match_stats(cls, query_idf_batch, query_batch, doc_batch):
        """
        The extra 4 features that must be combined to the relevant score.
        See https://www.aclweb.org/anthology/D18-1211.pdf section 4.1 beginning.
        """
        batch_size = doc_batch.shape[0]
        exact_matches = torch.zeros(batch_size)
        exact_match_idf = torch.zeros(batch_size)
        bigram_matches = torch.zeros(batch_size)

        for batch in range(batch_size):
            # The query is deliberately not cleaned, because the `query_idf` array we get is padded
            query = query_batch[batch]
            query_idf = query_idf_batch[batch]
            curr_doc = doc_batch[batch]

            # Remove pad tokens from the doc so that pad-token in query and pad-token in doc does not result in an exact
            # match count
            cleaned_doc = cls.clean(curr_doc)

            curr_exact_matches, curr_exact_matches_idf = cls.get_exact_match_count(query, cleaned_doc, query_idf)
            exact_matches[batch] = curr_exact_matches
            exact_match_idf[batch] = curr_exact_matches_idf
            bigram_matches[batch] = cls.get_bigram_match_count(query, cleaned_doc)

        return exact_matches.reshape(batch_size, 1), exact_match_idf.reshape(batch_size, 1), bigram_matches.reshape(batch_size, 1)
