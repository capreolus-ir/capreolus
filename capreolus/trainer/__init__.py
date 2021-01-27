import os
import json

import numpy as np
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

    @staticmethod
    def load_loss_file(fn):
        """Loads loss history from fn

        Args:
           fn (Path): path to a loss.txt file

        Returns:
            a list of losses ordered by iterations

        """

        loss = []
        with fn.open(mode="rt") as f:
            for lineidx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                iteridx, iterloss = line.rstrip().split()

                if int(iteridx) != lineidx:
                    raise IOError(f"malformed loss file {fn} ... did two processes write to it?")

                loss.append(float(iterloss))

        return loss

    @staticmethod
    def load_metric(fn):
        with fn.open(mode="rt") as f:
            return json.load(f)

    @staticmethod
    def load_best_metric(fn, metric):
        return Trainer.load_metric(fn).get(metric, -np.inf)

    @staticmethod
    def write_to_loss_file(fn, losses):
        fn.write_text("\n".join(f"{idx} {loss}" for idx, loss in enumerate(losses)))

    @staticmethod
    def write_to_metric_file(fn, metrics):
        assert isinstance(metrics, dict)
        json.dump(metrics, open(fn, "wt"))

    @staticmethod
    def exhaust_used_train_data(train_data_generator, n_batch_to_exhaust):
        for i, batch in enumerate(train_data_generator):
            if (i + 1) == n_batch_to_exhaust:
                break

    @property
    def n_batch_per_iter(self):
        return (self.config["itersize"] // self.config["batch"]) or 1

    @staticmethod
    def get_paths_for_early_stopping(train_output_path, dev_output_path):
        dev_best_weight_fn = train_output_path / "dev.best"
        weights_output_path = train_output_path / "weights"
        info_output_path = train_output_path / "info"
        os.makedirs(dev_output_path, exist_ok=True)
        os.makedirs(weights_output_path, exist_ok=True)
        os.makedirs(info_output_path, exist_ok=True)

        loss_fn = info_output_path / "loss.txt"
        metrics_fn = dev_output_path / "metrics.json"

        return dev_best_weight_fn, weights_output_path, info_output_path, loss_fn, metrics_fn

    def change_lr(self, step, lr):
        """
        Apply warm up or decay depending on the current epoch
        """
        return lr * self.lr_multiplier(step)

    def lr_multiplier(self, step):
        warmup_steps = self.config["warmupiters"] * self.n_batch_per_iter
        if warmup_steps and step <= warmup_steps:
            return min((step + 1) / warmup_steps, 1)
        elif self.config["decaytype"] == "exponential":
            decay_steps = self.config["decayiters"] * self.n_batch_per_iter
            return self.config["decay"] ** ((step - warmup_steps) / decay_steps)
        elif self.config["decaytype"] == "linear":
            epoch = (step - warmup_steps) / self.n_batch_per_iter
            return 1 / (1 + self.config["decay"] * epoch)  # todo: support endlr

        return 1


from profane import import_all_modules

from .pytorch import PytorchTrainer
from .tensorflow import TensorflowTrainer

import_all_modules(__file__, __package__)
