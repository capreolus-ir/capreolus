import torch.nn
from capreolus.encoder import Encoder
from capreolus.reranker.common import create_emb_layer


class GloveAvgEncoder_class(torch.nn.Module):
    def __init__(self, extractor, config):
        super(GloveAvgEncoder_class, self).__init__()

        self.config = config
        self.embedding = create_emb_layer(extractor.embeddings, non_trainable=True)
        
    def forward(self, numericalized_doc_toks):
        word_vectors = self.embedding(numericalized_doc_toks)

        # word_vectors has shape (batch, maxdoc_len, embedding_dim)
        return torch.mean(word_vectors, dim=2)


@Encoder.register
class GloveAvgEncoder(Encoder):
    """
    The simplest encoder that can be used for ANN.
    For each word in the doc, take its embedding if it exists, else ignore the word.
    The embedding for the doc is the average of all the word embeddings.
    """

    module_name = "gloveavg"

    def build_model(self):
        if not hasattr(self, "model"):
            self.model = GloveAvgEncoder_class(self.extractor, self.config)
        return self.model

    def encode(self, numericalized_doc_toks):
        return self.model(numericalized_doc_toks)

