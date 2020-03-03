import torch
from allennlp.modules.seq2seq_encoders import StackedSelfAttentionEncoder
from torch import nn
from capreolus.reranker.KNRM import KNRM_class, KNRM
from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)  # pylint: disable=invalid-name


class TK_class(KNRM_class):
    '''
    Adapted from https://github.com/sebastian-hofstaetter/transformer-kernel-ranking/blob/master/matchmaker/models/tk.py
    TK is a neural IR model - a fusion between transformer contextualization & kernel-based scoring
    -> uses 1 transformer block to contextualize embeddings
    -> soft-histogram kernels to score interactions
    '''

    def __init__(self, extractor, config):
        super(TK_class, self).__init__(extractor, config)
        self.embeddim = extractor.embeddings.shape[1]  # Dim of the glove6b embedding
        self.attention_encoder = StackedSelfAttentionEncoder(input_dim=self.embeddim,
                                                          hidden_dim=self.embeddim,
                                                          projection_dim=config["projdim"],
                                                          feedforward_hidden_dim=config["ffdim"],
                                                          num_layers=config["numlayers"],
                                                          num_attention_heads=config["numattheads"],
                                                          dropout_prob=0,
                                                          residual_dropout_prob=0,
                                                          attention_dropout_prob=0)
        self.mixer = nn.Parameter(torch.full([1, 1, 1], 0.5, dtype=torch.float32, requires_grad=True))
        self.pad = extractor.pad

    def get_embedding(self, toks):
        """
        Overrides KNRM_Class's get_embedding to return contextualized word embeddings
        """
        embedding = self.embedding(toks)
        # TODO: Hoffstaeter's implementation makes use of masking. Check if it's required here
        # See https://github.com/sebastian-hofstaetter/transformer-kernel-ranking/blob/master/matchmaker/models/tk.py#L88
        # The embedding is of the shape (batch_size, maxdoclen, embedding_size)
        # We want the mask to be of the shape (batch_size, maxdoclen). In other words, the mask says 1 if a token is not the pad token
        mask = ((embedding != torch.zeros(self.embeddim).to(embedding.device)).to(dtype=embedding.dtype).sum(-1) != 0).to(dtype=embedding.dtype)
        embedding = embedding * mask.unsqueeze(-1)
        contextual_embedding = self.attention_encoder(embedding, mask)

        return (self.mixer * embedding + (1 - self.mixer) * contextual_embedding) * mask.unsqueeze(-1)


class TK(KNRM):
    name = "TK"
    citation = """Add citation"""
    # TODO: Declare the dependency on EmbedText

    @staticmethod
    def config():
        gradkernels = True  # backprop through mus and sigmas
        scoretanh = False  # use a tanh on the prediction as in paper (True) or do not use a nonlinearity (False)
        singlefc = True  # use single fully connected layer as in paper (True) or 2 fully connected layers (False)
        projdim = 32
        ffdim = 100
        numlayers = 2
        numattheads = 8

    def build(self):
        if not hasattr(self, "model"):
            self.model = TK_class(self["extractor"], self.cfg)
        return self.model
