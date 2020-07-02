import os

from capreolus import ModuleBase, get_logger

logger = get_logger(__name__)  # pylint: disable=invalid-name


class Trainer(ModuleBase):
    """Base class for Trainer modules. The purpose of a Trainer is to train a :class:`~capreolus.reranker.Reranker` module and use it to make predictions. Capreolus provides two trainers: :class:`~capreolus.trainer.pytorch.PytorchTrainer` and :class:`~capreolus.trainer.tensorflow.TensorFlowTrainer`

    Modules should provide:
        - a ``train`` method that trains a reranker on training and dev (validation) data
        - a ``predict`` method that uses a reranker to make predictions on data
    """

    module_type = "trainer"
    requires_random_seed = True

    def get_paths_for_early_stopping(self, train_output_path, dev_output_path):
        os.makedirs(dev_output_path, exist_ok=True)
        dev_best_weight_fn = train_output_path / "dev.best"
        weights_output_path = train_output_path / "weights"
        info_output_path = train_output_path / "info"
        os.makedirs(weights_output_path, exist_ok=True)
        os.makedirs(info_output_path, exist_ok=True)

        loss_fn = info_output_path / "loss.txt"
        # metrics_fn = dev_output_path / "metrics.json"

        return dev_best_weight_fn, weights_output_path, info_output_path, loss_fn


from profane import import_all_modules

from .pytorch import PytorchTrainer
from .tensorflow import TensorFlowTrainer

import_all_modules(__file__, __package__)
