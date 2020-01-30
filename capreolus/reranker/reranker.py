import os
import pickle
import torch


class Reranker:
    ALL = {}

    @classmethod
    def register(cls, modelcls):
        name = modelcls.__name__

        # ensure we don't have two modules containing model classes with the same name
        if name in cls.ALL and cls.ALL[name] != modelcls:
            raise RuntimeError(f"encountered two models with the same name: {name}")

        cls.ALL[name] = modelcls
        return modelcls

    @staticmethod
    def required_params():
        # Used to enforce that the config options passed contains required values
        raise NotImplementedError("config method must be provided by subclass")

    @staticmethod
    def build_doc_scorer():
        raise NotImplementedError("build_doc_scorer method be provided by subclass")

    @classmethod
    def get_model_class(cls):
        raise NotImplementedError

    @classmethod
    def validate_params(cls, params):
        """
        Makes sure that all the `config` argument passed to the class specified by `get_model_class()` has all the
        required parameters.
        """
        expected_params = cls.required_params()
        supplied_params = params.keys()
        for expected_param in expected_params:
            if expected_param not in supplied_params:
                raise ValueError("Expected param not supplied to model: {0}".format(expected_param))

    def __init__(self, embeddings, bm25_scores, config):
        self.embeddings = embeddings
        self.config = config
        self.model = None
        self.oniter = 0
        self.bm25_scores = bm25_scores  # bm25 score for each qid and doc in this fold
        self.device = "cuda"

    def build(self):
        raise NotImplementedError

    def to(self, device):
        self.model.to(device)
        self.device = device
        return self

    def save(self, outfn):
        if not os.path.exists(os.path.dirname(outfn)):
            os.makedirs(os.path.dirname(outfn))

        d = self.model.state_dict()
        for k in list(d.keys()):
            if "embedding.weight" in k or "_nosave_" in k:
                del d[k]

        with open(outfn, "wb") as outf:
            pickle.dump(d, outf, protocol=-1)

    def load(self, fn):
        with open(fn, "rb") as f:
            d = pickle.load(f)

        cur_keys = set(k for k in self.model.state_dict().keys() if not ("embedding.weight" in k or "_nosave_" in k))
        missing = cur_keys - set(d.keys())
        if len(missing) > 0:
            raise RuntimeError("loading state_dict with keys that do not match current model: %s" % missing)

        self.model.load_state_dict(d, strict=False)

    def get_optimizer(self):
        opt = torch.optim.Adam(filter(lambda param: param.requires_grad, self.model.parameters()), lr=self.config["lr"])
        return opt

    def next_iteration(self):
        """ Called by train to indicate the beginning of a new iteration/epoch (1 indexed) """
        self.oniter += 1
        if hasattr(self.model, "oniter"):
            self.model.oniter = self.oniter
