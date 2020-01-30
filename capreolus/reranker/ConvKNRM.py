from capreolus.reranker.reranker import Reranker
import torch
from torch import nn
from capreolus.extractor.embedtext import EmbedText
from capreolus.reranker.common import create_emb_layer, SimilarityMatrix, RbfKernelBank


class ConvKNRM_class(nn.Module):
    def __init__(self, embeddings, config):
        super(ConvKNRM_class, self).__init__()
        pad_token = config["pad_token"]
        self.p = config
        self.simmat = SimilarityMatrix(padding=pad_token)
        self.embedding = create_emb_layer(embeddings)

        mus = [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        sigmas = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.001]
        self.kernels = RbfKernelBank(mus, sigmas, dim=1, requires_grad=config["gradkernels"])

        self.padding, self.convs = nn.ModuleList(), nn.ModuleList()
        self.padding_token = pad_token
        for conv_size in range(1, config["maxngram"] + 1):
            if conv_size > 1:
                self.padding.append(nn.ConstantPad1d((0, conv_size - 1), 0))
            else:
                self.padding.append(nn.Sequential())  # identity
            self.convs.append(nn.ModuleList())
            for _ in range(1):
                self.convs[-1].append(nn.Conv1d(embeddings.shape[1], config["filters"], conv_size))

        channels = config["maxngram"] ** 2 if config["crossmatch"] else config["maxngram"]
        channels *= 1 ** 2 if config["crossmatch"] else 1
        if config["singlefc"]:
            combine_steps = [nn.Linear(self.kernels.count() * channels, 1)]
        else:
            combine_steps = [nn.Linear(self.kernels.count() * channels, 30), nn.Tanh(), nn.Linear(30, 1)]
        if config["scoretanh"]:
            combine_steps.append(nn.Tanh())
        self.combine = nn.Sequential(*combine_steps)

    def forward(self, sentence, query_sentence, query_idf):
        a_embed = [self.embedding(query_sentence)]
        b_embed = [self.embedding(sentence)]
        a_reps, b_reps = [], []
        for layer, (a_emb, b_emb) in enumerate(zip(a_embed, b_embed)):
            for pad, conv in zip(self.padding, self.convs):
                a_reps.append(conv[layer](pad(a_emb.permute(0, 2, 1))).permute(0, 2, 1))
                b_reps.append(conv[layer](pad(b_emb.permute(0, 2, 1))).permute(0, 2, 1))

        simmats = []
        if self.p["crossmatch"]:
            for a_rep in a_reps:
                for b_rep in b_reps:
                    simmats.append(self.simmat(a_rep, b_rep, query_sentence, sentence))
        else:
            for i, a_rep in enumerate(a_reps):
                b_rep = b_reps[i]
                simmats.append(self.simmat(a_rep, b_rep, query_sentence, sentence))
        simmats = torch.cat(simmats, dim=1)

        # remainder is the same as KNRM
        kernels = self.kernels(simmats)
        BATCH, KERNELS, VIEWS, QLEN, DLEN = kernels.shape
        kernels = kernels.reshape(BATCH, KERNELS * VIEWS, QLEN, DLEN)
        simmats = (
            simmats.reshape(BATCH, 1, VIEWS, QLEN, DLEN)
            .expand(BATCH, KERNELS, VIEWS, QLEN, DLEN)
            .reshape(BATCH, KERNELS * VIEWS, QLEN, DLEN)
        )
        result = kernels.sum(dim=3)  # sum over document
        mask = simmats.sum(dim=3) != 0.0  # which query terms are not padding?
        result = torch.where(mask, (result + 1e-6).log(), mask.float())
        result = result.sum(dim=2)  # sum over query terms
        scores = self.combine(result)  # linear combination over kernels
        return scores


dtype = torch.FloatTensor


@Reranker.register
class ConvKNRM(Reranker):
    description = """Zhuyun Dai, Chenyan Xiong, Jamie Callan, and Zhiyuan Liu. 2018. Convolutional Neural Networks for Soft-Matching N-Grams in Ad-hoc Search. In WSDM'18."""
    EXTRACTORS = [EmbedText]

    @staticmethod
    def config():
        gradkernels = True  # backprop through mus and sigmas
        maxngram = 3  # maximum length of ngram considered
        crossmatch = True  # match query and document ngrams of different lengths (e.g., bigram vs. unigram)
        filters = 128  # number of filters used in convolutional layers
        scoretanh = False  # use a tanh on the prediction as in paper (True) or do not use a nonlinearity (False)
        singlefc = True  # use single fully connected layer as in paper (True) or 2 fully connected layers (False)
        return locals().copy()  # ignored by sacred

    @staticmethod
    def required_params():
        # Used for validation. Returns a set of params required by the class defined in get_model_class()
        return {"scoretanh", "singlefc", "crossmatch", "filters", "maxngram", "gradkernels", "pad_token"}

    @classmethod
    def get_model_class(cls):
        return ConvKNRM_class

    def build(self):
        config = self.config.copy()
        config["pad_token"] = EmbedText.pad
        self.model = ConvKNRM_class(self.embeddings, config)
        return self.model

    def score(self, d):
        query_idf = d["query_idf"]
        query_sentence = d["query"]
        pos_sentence, neg_sentence = d["posdoc"], d["negdoc"]
        return [
            self.model(pos_sentence, query_sentence, query_idf).view(-1),
            self.model(neg_sentence, query_sentence, query_idf).view(-1),
        ]

    def test(self, query_sentence, query_idf, pos_sentence, *args, **kwargs):
        return self.model(pos_sentence, query_sentence, query_idf).view(-1)

    def zero_grad(self, *args, **kwargs):
        self.model.zero_grad(*args, **kwargs)
