import contextlib
import math
import os
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from capreolus import ConfigOption, Searcher, constants, evaluator, get_logger
from capreolus.reranker.common import pair_hinge_loss, pair_softmax_loss

from . import Trainer

logger = get_logger(__name__)  # pylint: disable=invalid-name
RESULTS_BASE_PATH = constants["RESULTS_BASE_PATH"]


@Trainer.register
class PytorchTrainer(Trainer):
    module_name = "pytorch"
    config_spec = [
        ConfigOption("batch", 32, "batch size"),
        ConfigOption("evalbatch", 0, "batch size at inference time (or 0 to use training batch size)"),
        ConfigOption("niters", 20, "number of iterations to train for"),
        ConfigOption("itersize", 512, "number of training instances in one iteration"),
        ConfigOption("gradacc", 1, "number of batches to accumulate over before updating weights"),
        ConfigOption("lr", 0.001, "learning rate"),
        ConfigOption("softmaxloss", False, "True to use softmax loss (over pairs) or False to use hinge loss"),
        ConfigOption("fastforward", False),
        ConfigOption("validatefreq", 1),
        ConfigOption(
            "multithread",
            False,
            "True to load data in a separate thread; faster but causes PyTorch deadlock in some environments",
        ),
        ConfigOption("boardname", "default"),
        ConfigOption("warmupiters", 0),
        ConfigOption("decay", 0.0, "learning rate decay"),
        ConfigOption("decayiters", 3),
        ConfigOption("decaytype", None),
        ConfigOption("amp", None, "Automatic mixed precision mode; one of: None, train, pred, both"),
    ]
    config_keys_not_in_path = ["boardname"]

    def build(self):
        # sanity checks
        if self.config["batch"] < 1:
            raise ValueError("batch must be >= 1")

        if self.config["evalbatch"] < 0:
            raise ValueError("evalbatch must be 0 (to use the training batch size) or  >= 1")

        if self.config["niters"] <= 0:
            raise ValueError("niters must be > 0")

        if self.config["itersize"] < self.config["batch"]:
            raise ValueError("itersize must be >= batch")

        if self.config["gradacc"] < 1 or not float(self.config["gradacc"]).is_integer():
            raise ValueError("gradacc must be an integer >= 1")

        if self.config["lr"] <= 0:
            raise ValueError("lr must be > 0")

        if self.config["amp"] not in (None, "train", "pred", "both"):
            raise ValueError("amp must be one of: None, train, pred, both")

        torch.manual_seed(self.config["seed"])
        torch.cuda.manual_seed_all(self.config["seed"])

    def single_train_iteration(self, reranker, train_dataloader):
        """Train model for one iteration using instances from train_dataloader.

        Args:
           model (Reranker): a PyTorch Reranker
           train_dataloader (DataLoader): a PyTorch DataLoader that iterates over training instances

        Returns:
            float: average loss over the iteration

        """

        iter_loss = []
        batches_since_update = 0
        batches_per_step = self.config["gradacc"]

        for bi, batch in tqdm(enumerate(train_dataloader), desc="Training iteration", total=self.n_batch_per_iter):
            batch = {k: v.to(self.device) if not isinstance(v, list) else v for k, v in batch.items()}

            with self.amp_train_autocast():
                doc_scores = reranker.score(batch)
                loss = self.loss(doc_scores)

            iter_loss.append(loss)
            loss = self.scaler.scale(loss) if self.scaler else loss
            loss.backward()

            batches_since_update += 1
            if batches_since_update == batches_per_step:
                batches_since_update = 0
                # REF-TODO: save scaler state
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

            if (bi + 1) % self.n_batch_per_iter == 0:
                # REF-TODO: save scheduler state along with optimizer
                self.lr_scheduler.step()
                break

        return torch.stack(iter_loss).mean()

    def fastforward_training(self, reranker, weights_path, loss_fn, best_metric_fn):
        """Skip to the last training iteration whose weights were saved.

        If saved model and optimizer weights are available, this method will load those weights into model
        and optimizer, and then return the next iteration to be run. For example, if weights are available for
        iterations 0-10 (11 zero-indexed iterations), the weights from iteration index 10 will be loaded, and
        this method will return 11.

        If an error or inconsistency is encountered when checking for weights, this method returns 0.

        This method checks several files to determine if weights "are available". First, loss_fn is read to
        determine the last recorded iteration. (If a path is missing or loss_fn is malformed, 0 is returned.)
        Second, the weights from the last recorded iteration in loss_fn are loaded into the model and optimizer.
        If this is successful, the method returns `1 + last recorded iteration`. If not, it returns 0.
        (We consider loss_fn because it is written at the end of every training iteration.)

        Args:
           model (Reranker): a PyTorch Reranker whose state should be loaded
           weights_path (Path): directory containing model and optimizer weights
           loss_fn (Path): file containing loss history

        Returns:
            int: the next training iteration after fastforwarding. If successful, this is > 0.
                 If no weights are available or they cannot be loaded, 0 is returned.

        """
        default_return_values = (0, {})
        if not (weights_path.exists() and loss_fn.exists()):
            return default_return_values

        try:
            loss = self.load_loss_file(loss_fn)
            metrics = self.load_metric(best_metric_fn)
        except IOError:
            return default_return_values

        last_loss_iteration = len(loss) - 1
        weights_fn = weights_path / f"{last_loss_iteration}.p"

        try:
            reranker.load_weights(weights_fn, self.optimizer)
            return last_loss_iteration + 1, metrics
        except:  # lgtm [py/catch-base-exception]
            logger.info("attempted to load weights from %s but failed, starting at iteration 0", weights_fn)
            return default_return_values

    def train(self, reranker, train_dataset, train_output_path, dev_data, dev_output_path, qrels, metric, relevance_level=1):
        """Train a model following the trainer's config (specifying batch size, number of iterations, etc).

        Args:
           train_dataset (IterableDataset): training dataset
           train_output_path (Path): directory under which train_dataset runs and training loss will be saved
           dev_data (IterableDataset): dev dataset
           dev_output_path (Path): directory where dev_data runs and metrics will be saved

        """
        # Set up logging
        # TODO why not put this under train_output_path?
        summary_writer = SummaryWriter(RESULTS_BASE_PATH / "runs" / self.config["boardname"], comment=train_output_path)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = reranker.model.to(self.device)
        self.optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad, model.parameters()), lr=self.config["lr"])

        if self.config["amp"] in ("both", "train"):
            self.amp_train_autocast = torch.cuda.amp.autocast
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.amp_train_autocast = contextlib.nullcontext
            self.scaler = None

        # REF-TODO how to handle interactions between fastforward and schedule? --> just save its state
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda epoch: self.lr_multiplier(step=epoch * self.n_batch_per_iter)
        )

        if self.config["softmaxloss"]:
            self.loss = pair_softmax_loss
        else:
            self.loss = pair_hinge_loss

        dev_best_weight_fn, weights_output_path, info_output_path, loss_fn, metric_fn = self.get_paths_for_early_stopping(
            train_output_path, dev_output_path
        )

        num_workers = 1 if self.config["multithread"] else 0
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.config["batch"], pin_memory=True, num_workers=num_workers
        )

        # if we're fastforwarding, set first iteration and load last saved weights
        initial_iter, metrics = (
            self.fastforward_training(reranker, weights_output_path, loss_fn, metric_fn)
            if self.config["fastforward"]
            else (0, {})
        )
        dev_best_metric = metrics.get(metric, -np.inf)
        logger.info("starting training from iteration %s/%s", initial_iter + 1, self.config["niters"])
        logger.info(f"Best metric loaded: {metric}={dev_best_metric}")

        train_loss = []
        # are we resuming training? fastforward loss and data if so
        if initial_iter > 0:
            train_loss = self.load_loss_file(loss_fn)

            # are we done training? if not, fastforward through prior batches
            if initial_iter < self.config["niters"]:
                logger.debug("fastforwarding train_dataloader to iteration %s", initial_iter)
                self.exhaust_used_train_data(train_dataloader, n_batch_to_exhaust=initial_iter * self.n_batch_per_iter)

        train_start_time = time.time()
        for niter in range(initial_iter, self.config["niters"]):
            niter = niter + 1  # index from 1
            model.train()

            iter_start_time = time.time()
            iter_loss_tensor = self.single_train_iteration(reranker, train_dataloader)
            logger.info("A single iteration takes {}".format(time.time() - iter_start_time))
            train_loss.append(iter_loss_tensor.item())
            logger.info("iter = %d loss = %f", niter, train_loss[-1])

            # save model weights only when fastforward enabled
            if self.config["fastforward"]:
                weights_fn = weights_output_path / f"{niter}.p"
                reranker.save_weights(weights_fn, self.optimizer)

            # predict performance on dev set
            if niter % self.config["validatefreq"] == 0:
                pred_fn = dev_output_path / f"{niter}.run"
                preds = self.predict(reranker, dev_data, pred_fn)

                # log dev metrics
                metrics = evaluator.eval_runs(preds, qrels, evaluator.DEFAULT_METRICS, relevance_level)
                logger.info("dev metrics: %s", " ".join([f"{metric}={v:0.3f}" for metric, v in sorted(metrics.items())]))
                summary_writer.add_scalar("ndcg_cut_20", metrics["ndcg_cut_20"], niter)
                summary_writer.add_scalar("map", metrics["map"], niter)
                summary_writer.add_scalar("P_20", metrics["P_20"], niter)
                # write best dev weights to file
                if metrics[metric] > dev_best_metric:
                    dev_best_metric = metrics[metric]
                    logger.info("new best dev metric: %0.4f", dev_best_metric)
                    reranker.save_weights(dev_best_weight_fn, self.optimizer)
                    self.write_to_metric_file(metric_fn, metrics)

            # write train_loss to file
            # loss_fn.write_text("\n".join(f"{idx} {loss}" for idx, loss in enumerate(train_loss)))
            self.write_to_loss_file(loss_fn, train_loss)

            summary_writer.add_scalar("training_loss", iter_loss_tensor.item(), niter)
            reranker.add_summary(summary_writer, niter)
            summary_writer.flush()
        logger.info("training loss: %s", train_loss)
        logger.info("Training took {}".format(time.time() - train_start_time))
        summary_writer.close()

        # TODO should we write a /done so that training can be skipped if possible when fastforward=False? or in Task?

    def load_best_model(self, reranker, train_output_path):
        self.optimizer = torch.optim.Adam(
            filter(lambda param: param.requires_grad, reranker.model.parameters()), lr=self.config["lr"]
        )

        dev_best_weight_fn = train_output_path / "dev.best"
        reranker.load_weights(dev_best_weight_fn, self.optimizer)

    def predict(self, reranker, pred_data, pred_fn):
        """Predict query-document scores on `pred_data` using `model` and write a corresponding run file to `pred_fn`

        Args:
           model (Reranker): a PyTorch Reranker
           pred_data (IterableDataset): data to predict on
           pred_fn (Path): path to write the prediction run file to

        Returns:
           TREC Run

        """

        if self.config["amp"] in ("both", "pred"):
            self.amp_pred_autocast = torch.cuda.amp.autocast
        else:
            self.amp_pred_autocast = contextlib.nullcontext

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # save to pred_fn
        model = reranker.model.to(self.device)
        model.eval()

        preds = {}
        evalbatch = self.config["evalbatch"] if self.config["evalbatch"] > 0 else self.config["batch"]
        num_workers = 1 if self.config["multithread"] else 0
        pred_dataloader = torch.utils.data.DataLoader(pred_data, batch_size=evalbatch, pin_memory=True, num_workers=num_workers)
        with torch.autograd.no_grad():
            for batch in tqdm(pred_dataloader, desc="Predicting", total=len(pred_data) // evalbatch):
                if len(batch["qid"]) != evalbatch:
                    batch = self.fill_incomplete_batch(batch, batch_size=evalbatch)

                batch = {k: v.to(self.device) if not isinstance(v, list) else v for k, v in batch.items()}
                with self.amp_pred_autocast():
                    scores = reranker.test(batch)
                scores = scores.view(-1).cpu().numpy()
                for qid, docid, score in zip(batch["qid"], batch["posdocid"], scores):
                    # Need to use float16 because pytrec_eval's c function call crashes with higher precision floats
                    preds.setdefault(qid, {})[docid] = score.astype(np.float16).item()

        os.makedirs(os.path.dirname(pred_fn), exist_ok=True)
        Searcher.write_trec_run(preds, pred_fn)

        return preds

    def fill_incomplete_batch(self, batch, batch_size=None):
        """
        If a batch is incomplete (i.e shorter than the desired batch size), this method fills in the batch with some data.
        How the data is chosen:
        If the values are just a simple list, use the first element of the list to pad the batch
        If the values are tensors/numpy arrays, use repeat() along the batch dimension
        """
        # logger.debug("filling in an incomplete batch")
        if not batch_size:
            batch_size = self.config["batch"]
        repeat_times = math.ceil(batch_size / len(batch["qid"]))
        diff = batch_size - len(batch["qid"])

        def pad(v):
            if isinstance(v, np.ndarray) or torch.is_tensor(v):
                _v = v.repeat((repeat_times,) + tuple([1 for x in range(len(v.shape) - 1)]))
            else:
                _v = v + [v[0]] * diff

            return _v[:batch_size]

        batch = {k: pad(v) for k, v in batch.items()}
        return batch
