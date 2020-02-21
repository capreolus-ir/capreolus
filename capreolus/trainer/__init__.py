import os

import numpy as np
import torch

from capreolus.registry import ModuleBase, RegisterableModule, Dependency, MAX_THREADS
from capreolus.reranker.common import pair_hinge_loss, pair_softmax_loss
from capreolus.searcher import Searcher
from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)  # pylint: disable=invalid-name


class Trainer(ModuleBase, metaclass=RegisterableModule):
    module_type = "trainer"


class PytorchTrainer(Trainer):
    name = "pytorch"
    dependencies = {}

    @staticmethod
    def config():
        # TODO move maxdoclen, maxqlen to extractor?
        maxdoclen = 800  # maximum document length (in number of terms after tokenization)
        maxqlen = 4  # maximum query length (in number of terms after tokenization)

        batch = 32  # batch size
        niters = 20  # number of iterations to train for
        itersize = 512  # number of training instances in one iteration (epoch)
        gradacc = 1  # number of batches to accumulate over before updating weights
        lr = 0.001  # learning rate
        softmaxloss = True  # True to use softmax loss (over pairs) or False to use hinge loss

        # sanity checks
        if batch < 1:
            raise ValueError("batch must be >= 1")

        if niters <= 0:
            raise ValueError("niters must be > 0")

        if itersize < batch:
            raise ValueError("itersize must be >= batch")

        if gradacc < 1 or not float(gradacc).is_integer():
            raise ValueError("gradacc must be an integer >= 1")

        if lr <= 0:
            raise ValueError("lr must be > 0")

    def single_train_iteration(self, reranker, optimizer, train_dataloader):
        """Train model for one iteration using instances from train_dataloader.

        Args:
           model (Reranker): a PyTorch Reranker
           optimizer (Optimizer): a PyTorch Optimizer
           train_dataloader (DataLoader): a PyTorch DataLoader that iterates over training instances

        Returns:
            float: average loss over the iteration

        """

        iter_loss = []
        batches_since_update = 0
        batches_per_epoch = self.cfg["itersize"] // self.cfg["batch"]
        batches_per_step = self.cfg["gradacc"]

        for bi, batch in enumerate(train_dataloader):
            # TODO make sure _prepare_batch_with_strings equivalent is happening inside the sampler
            doc_scores = reranker.score(batch)
            loss = self.loss(doc_scores)
            iter_loss.append(loss)
            loss.backward()

            batches_since_update += 1
            if batches_since_update == batches_per_step:
                batches_since_update = 0
                optimizer.step()
                optimizer.zero_grad()

            if (bi + 1) % batches_per_epoch == 0:
                break

        return torch.stack(iter_loss).mean()

    def load_loss_file(self, fn):
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

    def fastforward_training(self, reranker, optimizer, weights_path, loss_fn):
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
           optimizer (Optimizer): a PyTorch Optimizer whose state should be loaded
           weights_path (Path): directory containing model and optimizer weights
           loss_fn (Path): file containing loss history

        Returns:
            int: the next training iteration after fastforwarding. If successful, this is > 0.
                 If no weights are available or they cannot be loaded, 0 is returned.

        """

        if not (weights_path.exists() and loss_fn.exists()):
            return 0

        try:
            loss = self.load_loss_file(loss_fn)
        except IOError:
            return 0

        last_loss_iteration = len(loss) - 1
        weights_fn = weights_path / f"{last_loss_iteration}.p"

        try:
            reranker.load_weights(weights_fn, optimizer)
            # TODO also load optimizer state
            return last_loss_iteration + 1
        except:
            return 0

    def train(self, reranker, train_dataset, train_output_path, dev_data, dev_output_path):
        """Train a model following the trainer's config (specifying batch size, number of iterations, etc).

        Args:
           train_dataset (IterableDataset): training dataset
           train_output_path (Path): directory under which train_dataset runs and training loss will be saved
           dev_data (IterableDataset): dev dataset
           dev_output_path (Path): directory where dev_data runs and metrics will be saved

        """

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = reranker.model.to(self.device)
        optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad, model.parameters()), lr=self.cfg["lr"])

        if self.cfg["softmaxloss"]:
            self.loss = pair_softmax_loss
        else:
            self.loss = pair_hinge_loss

        os.makedirs(dev_output_path, exist_ok=True)
        weights_output_path = train_output_path / "weights"
        info_output_path = train_output_path / "info"
        os.makedirs(weights_output_path, exist_ok=True)
        os.makedirs(info_output_path, exist_ok=True)

        loss_fn = info_output_path / "loss.txt"
        initial_iter = self.fastforward_training(reranker, optimizer, weights_output_path, loss_fn)

        train_loss = []
        dev_metrics = []
        for niter in range(initial_iter, self.cfg["niters"]):
            model.train()

            # we must keep train_dataset updated with the current iteration
            # because this is used to seed the training data order!
            train_dataset.iteration = niter
            # now create a DataLoader for this iteration
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset, batch_size=self.cfg["batch"], pin_memory=True, num_workers=0
            )

            iter_loss_tensor = self.single_train_iteration(reranker, optimizer, train_dataloader)
            del train_dataloader

            train_loss.append(iter_loss_tensor.item())
            logger.info("iter = %d loss = %f", niter, train_loss[-1])

            # write model weights to file
            weights_fn = weights_output_path / f"{niter}.p"
            reranker.save_weights(weights_fn, optimizer)
            # TODO also save optimizer state

            # predict performance on dev set
            pred_fn = dev_output_path / f"{niter}.run"
            self.predict(reranker, dev_data, pred_fn)

            # write dev metrics to file
            # TODO re-enable
            # metrics = evaluator.evaluate(pred_fn)
            # dev_metrics.append(metrics)
            # logger.info("dev metrics")

            # write train_loss to file
            loss_fn.write_text("\n".join(f"{idx} {loss}" for idx, loss in enumerate(train_loss)))

    def predict(self, reranker, pred_data, pred_fn):
        """Predict query-document scores on `pred_data` using `model` and write a corresponding run file to `pred_fn`

        Args:
           model (Reranker): a PyTorch Reranker
           pred_data (IterableDataset): data to predict on
           pred_fn (Path): path to write the prediction run file to

        Returns:
           TREC Run 

        """

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # save to pred_fn
        model = reranker.model.to(self.device)
        model.eval()

        preds = {}
        pred_dataloader = torch.utils.data.DataLoader(pred_data, batch_size=self.cfg["batch"], pin_memory=True, num_workers=0)
        with torch.autograd.no_grad():
            for bi, batch in enumerate(pred_dataloader):
                scores = reranker.test(batch)
                scores = scores.view(-1).cpu().numpy()
                for qid, docid, score in zip(batch["qid"], batch["posdocid"], scores):
                    # Need to use float16 because pytrec_eval's c function call crashes with higher precision floats
                    preds.setdefault(qid, {})[docid] = score.astype(np.float16).item()

        os.makedirs(os.path.dirname(pred_fn), exist_ok=True)
        Searcher.write_trec_run(preds, pred_fn)

        return preds
