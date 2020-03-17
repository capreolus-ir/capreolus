from capreolus.reranker.KNRM import KNRM, KNRM_class
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch import nn
import math
from capreolus.utils.loginit import get_logger


logger = get_logger(__name__)  # pylint: disable=invalid-name


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TK_class(KNRM_class):
    """
    Adapted from https://github.com/sebastian-hofstaetter/transformer-kernel-ranking/blob/master/matchmaker/models/tk.py
    TK is a neural IR model - a fusion between transformer contextualization & kernel-based scoring
    -> uses 1 transformer block to contextualize embeddings
    -> soft-histogram kernels to score interactions
    """

    def __init__(self, extractor, config):
        super(TK_class, self).__init__(extractor, config)
        self.embeddim = extractor.embeddings.shape[1]
        dropout = 0.1
        self.position_encoder = PositionalEncoding(self.embeddim)
        self.mixer = nn.Parameter(torch.full([1, 1, 1], 0.5, dtype=torch.float32, requires_grad=True))
        encoder_layers = TransformerEncoderLayer(
            self.embeddim, config["numattheads"], config["ffdim"], dropout
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layers, config["numlayers"]
        )

    def get_mask(self, embedding):
        """
        Gets a mask of shape (seq_len, seq_len). This is an additive mask, hence masked elements should be -inf
        """
        batch_size = embedding.shape[0]
        seq_len = embedding.shape[1]
        # Get a normal mask of shape (batch_size, seq_len). Entry would be 0 if a seq element should be masked
        mask = ((embedding != torch.zeros(self.embeddim).to(embedding.device)).to(dtype=embedding.dtype).sum(
            -1) != 0).to(dtype=embedding.dtype)

        # The square attention mask
        encoder_mask = torch.zeros(batch_size, seq_len, seq_len).to(embedding.device)
        # Set -inf on all rows corresponding to a pad token
        encoder_mask[mask == 0] = float('-inf')
        # Set -inf on all columns corresponding to a pad token (the tricky bit)
        col_mask = mask.reshape(batch_size, 1, seq_len).expand(batch_size, seq_len, seq_len)
        encoder_mask[col_mask == 0] = float('-inf')

        return torch.cat([encoder_mask] * self.p["numattheads"])

    def get_embedding(self, toks):
        """
        Overrides KNRM_Class's get_embedding to return contextualized word embeddings
        """
        embedding = self.embedding(toks)

        # Transformer layers expect input in shape (L, N, E), where L is sequence len, N is batch, E is embed dims
        reshaped_embedding = embedding.permute(1, 0, 2)
        position_encoded_embedding = self.position_encoder(reshaped_embedding)
        # TODO: Mask should be additive
        mask = self.get_mask(embedding) if self.p["usemask"] else None
        contextual_embedding = self.transformer_encoder(position_encoded_embedding, mask).permute(1, 0, 2)
        if self.p["usemixer"]:
            return self.mixer * embedding + (1 - self.mixer) * contextual_embedding
        else:
            return self.p["alpha"] * embedding + (1-self.p["alpha"]) * contextual_embedding


class TK(KNRM):
    name = "TK"
    citation = """Add citation"""
    # TODO: Declare the dependency on EmbedText

    @staticmethod
    def config():
        gradkernels = True  # backprop through mus and sigmas
        scoretanh = (
            False
        )  # use a tanh on the prediction as in paper (True) or do not use a nonlinearity (False)
        singlefc = (
            True
        )  # use single fully connected layer as in paper (True) or 2 fully connected layers (False)
        projdim = 32
        ffdim = 100
        numlayers = 2
        numattheads = 8
        alpha = 0.5
        usemask = False
        usemixer = False

    def build(self):
        if not hasattr(self, "model"):
            self.model = TK_class(self["extractor"], self.cfg)
        return self.model

    def score(self, d):
        query_idf = d["query_idf"]
        query_sentence = d["query"]
        pos_sentence, neg_sentence = d["posdoc"], d["negdoc"]
        return [
            self.model(pos_sentence, query_sentence, query_idf).view(-1),
            self.model(neg_sentence, query_sentence, query_idf).view(-1),
        ]

    def test(self, d):
        query_idf = d["query_idf"]
        query_sentence = d["query"]
        pos_sentence = d["posdoc"]
        return self.model(pos_sentence, query_sentence, query_idf).view(-1)
