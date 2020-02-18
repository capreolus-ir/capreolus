import torch

from capreolus.registry import ModuleBase, RegisterableModule, Dependency, MAX_THREADS
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

    def single_train_iteration(self, model, train_data):
        iter_loss = []
        batches_since_update = 0
        batches_per_epoch = self.cfg["itersize"] // self.cfg["batch"]
        batches_per_step = self.cfg["gradacc"]

        for bi, batch in enumerate(train_data):
            # TODO make sure _prepare_batch_with_strings equivalent is happening inside the sampler
            doc_scores = model.score(batch)
            loss = self.loss(doc_scores)
            iter_loss.append(loss)
            loss.backward()

            batches_since_update += 1
            if batches_since_update == batches_per_step:
                batches_since_update = 0
                self.optimizer.step()
                self.optimizer.zero_grad()

            if (bi + 1) % batches_per_epoch == 0:
                break

        return iter_loss.mean()

    def train(self, model, train_data, train_output_path, dev_data, dev_output_path):
        model.to(self.device)
        self.optimizer = something

        weights_output_path = train_output_path / "weights"
        info_output_path = train_output_path / "info"

        train_loss = []
        dev_metrics = []
        initial_iter = 0
        for niter in range(initial_iter, self.cfg["niters"]):
            model.train()

            iter_avg_loss = self.single_train_iteration(model, train_data)
            iter_avg_loss = iter_avg_loss.item()
            train_loss.append(iter_avg_loss)
            logger.info("iter = %d loss = %f", niter, iter_avg_loss)

            pred_fn = dev_output_path / f"{niter}.run"
            self.predict(model, dev_data, pred_fn)
            metrics = evaluator.evaluate(pred_fn)
            dev_metrics.append(metrics)
            # logger.info("dev metrics")
            # write metrics to file

            # write model weights to file
            weights_fn = weights_output_path / f"{niter}.p"
            model.save_weights(weights_fn)

        # write train_loss to file
        loss_fn = info_output_path / "loss.txt"
        with open(loss_fn, "wt") as outf:
            for idx, loss in enumerate(train_loss):
                print(f"{idx} {loss}", file=outf)
        # write dev metrics to a combined file or leave them separate?

    def predict(self, model, pred_data, pred_fn):
        # save to pred_fn
        model.to(self.device)
        model.eval()

        with torch.autograd.no_grad():
            for bi, batch in enumerate(pred_data):
                doc_scores = model.score(batch)
                scores = scores.view(-1).cpu().numpy()
                for qid, docid, score in zip(qid_batch, docid_batch, scores):
                    # Need to use float16 because pytrec_eval's c function call crashes with higher precision floats
                    preds[qid][docid] = score.astype(np.float16).item()
