import torch
from allennlp.modules.seq2seq_encoders import StackedSelfAttentionEncoder
from allennlp.modules.matrix_attention.cosine_matrix_attention import CosineMatrixAttention
from capreolus.reranker.KNRM import KNRM
from torch import nn
from torch.autograd import Variable
from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)  # pylint: disable=invalid-name


def create_emb_layer(weights, non_trainable=True):
    layer = torch.nn.Embedding(*weights.shape)
    layer.load_state_dict({"weight": torch.tensor(weights)})

    if non_trainable:
        layer.weight.requires_grad = False
    else:
        layer.weight.requires_grad = True

    return layer


class TK_class(nn.Module):
    '''
    Adapted from https://github.com/sebastian-hofstaetter/transformer-kernel-ranking/blob/master/matchmaker/models/tk.py
    TK is a neural IR model - a fusion between transformer contextualization & kernel-based scoring
    -> uses 1 transformer block to contextualize embeddings
    -> soft-histogram kernels to score interactions
    '''

    def __init__(self, extractor, config):
        super(TK_class, self).__init__()
        self.embeddim = extractor.embeddings.shape[1]  # Dim of the glove6b embedding
        self.embedding = create_emb_layer(extractor.embeddings, non_trainable=True)
        self.attention_encoder = StackedSelfAttentionEncoder(input_dim=self.embeddim,
                                                          hidden_dim=self.embeddim,
                                                          projection_dim=config["projdim"],
                                                          feedforward_hidden_dim=config["ffdim"],
                                                          num_layers=config["numlayers"],
                                                          num_attention_heads=config["numattheads"],
                                                          dropout_prob=0,
                                                          residual_dropout_prob=0,
                                                          attention_dropout_prob=0)
        kernels_mu = torch.tensor([1.0, 0.9, 0.7, 0.5, 0.3, 0.1, -0.1, -0.3, -0.5, -0.7, -0.9], requires_grad=False).float().cuda()
        kernels_sigma = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], requires_grad=False).float().cuda()
        n_kernels = len(kernels_mu)

        self.mu = Variable(kernels_mu).view(1, 1, 1, n_kernels)
        self.sigma = Variable(kernels_sigma).view(1, 1, 1, n_kernels)
        self.nn_scaler = nn.Parameter(torch.full([1], 0.01, dtype=torch.float32, requires_grad=True))
        self.mixer = nn.Parameter(torch.full([1, 1, 1], 0.5, dtype=torch.float32, requires_grad=True))
        self.pad = extractor.pad
        self.cosine_module = CosineMatrixAttention()

        # bias is set to True in original code (we found it to not help, how could it?)
        self.dense = nn.Linear(n_kernels, 1, bias=False)
        self.dense_mean = nn.Linear(n_kernels, 1, bias=False)
        self.dense_comb = nn.Linear(2, 1, bias=False)

        # init with small weights, otherwise the dense output is way to high for the tanh -> resulting in loss == 1 all the time
        torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014)  # inits taken from matchzoo
        torch.nn.init.uniform_(self.dense_mean.weight, -0.014, 0.014)  # inits taken from matchzoo

        # init with small weights, otherwise the dense output is way to high for the tanh -> resulting in loss == 1 all the time
        torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014)  # inits taken from matchzoo

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

        return (self.mixer * embedding + (1 - self.mixer) * contextual_embedding) * mask.unsqueeze(-1), mask

    def forward(self, doctoks, querytoks, query_idf):
        doc, doc_mask = self.get_embedding(doctoks)
        query, query_mask = self.get_embedding(querytoks)
        query_by_doc_mask = torch.bmm(query_mask.unsqueeze(-1),
                                      doc_mask.unsqueeze(-1).transpose(-1, -2))
        query_by_doc_mask_view = query_by_doc_mask.unsqueeze(-1)

        #
        # cosine matrix
        # -------------------------------------------------------

        # shape: (batch, query_max, doc_max)
        cosine_matrix = self.cosine_module.forward(query, doc)
        cosine_matrix_masked = cosine_matrix * query_by_doc_mask
        cosine_matrix_extradim = cosine_matrix_masked.unsqueeze(-1)

        #
        # gaussian kernels & soft-TF
        #
        # first run through kernel, then sum on doc dim then sum on query dim
        # -------------------------------------------------------

        raw_kernel_results = torch.exp(
            - torch.pow(cosine_matrix_extradim - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))
        kernel_results_masked = raw_kernel_results * query_by_doc_mask_view

        #
        # mean kernels
        #
        # kernel_results_masked2 = kernel_results_masked.clone()

        doc_lengths = torch.sum(doc_mask, 1)

        # kernel_results_masked2_mean = kernel_results_masked / doc_lengths.unsqueeze(-1)

        per_kernel_query = torch.sum(kernel_results_masked, 2)
        log_per_kernel_query = torch.log2(torch.clamp(per_kernel_query, min=1e-10)) * self.nn_scaler
        log_per_kernel_query_masked = log_per_kernel_query * query_mask.unsqueeze(
            -1)  # make sure we mask out padding values
        per_kernel = torch.sum(log_per_kernel_query_masked, 1)

        # per_kernel_query_mean = torch.sum(kernel_results_masked2_mean, 2)

        per_kernel_query_mean = per_kernel_query / (doc_lengths.view(-1, 1,
                                                                     1) + 1)  # well, that +1 needs an explanation, sometimes training data is just broken ... (and nans all the things!)

        log_per_kernel_query_mean = per_kernel_query_mean * self.nn_scaler
        # logger.info("log_per_kernel_query_mean is {}".format(log_per_kernel_query_mean.shape))
        # logger.info("doc_mask is {}".format(doc_mask.unsqueeze(-1).shape))
        log_per_kernel_query_masked_mean = log_per_kernel_query_mean * query_mask.unsqueeze(
            -1)  # make sure we mask out padding values
        per_kernel_mean = torch.sum(log_per_kernel_query_masked_mean, 1)

        ##
        ## "Learning to rank" layer - connects kernels with learned weights
        ## -------------------------------------------------------

        dense_out = self.dense(per_kernel)
        dense_mean_out = self.dense_mean(per_kernel_mean)
        dense_comb_out = self.dense_comb(torch.cat([dense_out, dense_mean_out], dim=1))
        score = torch.squeeze(dense_comb_out, 1)  # torch.tanh(dense_out), 1)

        return score


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

    def score(self, d):
        query_idf = d["query_idf"]
        query_sentence = d["query"]
        pos_sentence, neg_sentence = d["posdoc"], d["negdoc"]
        return [
            self.model(pos_sentence, query_sentence, query_idf).view(-1),
            self.model(neg_sentence, query_sentence, query_idf).view(-1),
        ]
        # return [
        #     self.model(neg_sentence, query_sentence, query_idf).view(-1),
        #     self.model(pos_sentence, query_sentence, query_idf).view(-1),
        # ]

    def test(self, d):
        query_idf = d["query_idf"]
        query_sentence = d["query"]
        pos_sentence = d["posdoc"]
        return self.model(pos_sentence, query_sentence, query_idf).view(-1)

    def query(self, query, docids):
        if not hasattr(self["extractor"], "docid2toks"):
            raise RuntimeError("reranker's extractor has not been created yet. try running the task's train() method first.")

        results = []
        for docid in docids:
            d = self["extractor"].id2vec(qid=None, query=query, posid=docid)
            results.append(self.test(d))

        return results
