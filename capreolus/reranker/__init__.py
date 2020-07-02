import os
import pickle

from capreolus import Dependency, ModuleBase


class Reranker(ModuleBase):
    """Base class for Reranker modules. The purpose of a Reranker is to predict relevance scores for input documents. Rerankers are generally supervised methods implemented in PyTorch or TensorFlow.

    Modules should provide:
        - a ``build_model`` method that initializes the model used
        - a ``score`` and a ``test`` method that take a representation created by an :class:`~capreolus.extractor.Extractor` module as input and return document scores
        - a ``load_weights`` and a ``save_weights`` method, if the base class' PyTorch methods cannot be used
    """

    module_type = "reranker"
    dependencies = [
        Dependency(key="extractor", module="extractor", name="embedtext"),
        Dependency(key="trainer", module="trainer", name="pytorch"),
    ]

    def add_summary(self, summary_writer, niter):
        """
        Write to the summay_writer custom visualizations/data specific to this reranker
        """
        for name, weight in self.model.named_parameters():
            summary_writer.add_histogram(name, weight.data.cpu(), niter)
            # summary_writer.add_histogram(f'{name}.grad', weight.grad, niter)

    def save_weights(self, weights_fn, optimizer):
        if not os.path.exists(os.path.dirname(weights_fn)):
            os.makedirs(os.path.dirname(weights_fn))

        d = {k: v for k, v in self.model.state_dict().items() if ("embedding.weight" not in k and "_nosave_" not in k)}
        with open(weights_fn, "wb") as outf:
            pickle.dump(d, outf, protocol=-1)

        optimizer_fn = weights_fn.as_posix() + ".optimizer"
        with open(optimizer_fn, "wb") as outf:
            pickle.dump(optimizer.state_dict(), outf, protocol=-1)

    def load_weights(self, weights_fn, optimizer):
        with open(weights_fn, "rb") as f:
            d = pickle.load(f)

        cur_keys = set(k for k in self.model.state_dict().keys() if not ("embedding.weight" in k or "_nosave_" in k))
        missing = cur_keys - set(d.keys())
        if len(missing) > 0:
            raise RuntimeError("loading state_dict with keys that do not match current model: %s" % missing)

        self.model.load_state_dict(d, strict=False)

        optimizer_fn = weights_fn.as_posix() + ".optimizer"
        with open(optimizer_fn, "rb") as f:
            optimizer.load_state_dict(pickle.load(f))


from profane import import_all_modules


import_all_modules(__file__, __package__)
