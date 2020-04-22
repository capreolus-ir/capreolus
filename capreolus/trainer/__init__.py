import os
import uuid
from collections import defaultdict

import tensorflow as  tf

import numpy as np
import torch
from tqdm import tqdm
from capreolus.registry import ModuleBase, RegisterableModule, Dependency, MAX_THREADS
from capreolus.reranker.common import pair_hinge_loss, pair_softmax_loss, tf_pair_hinge_loss
from capreolus.searcher import Searcher
from capreolus.utils.loginit import get_logger
from capreolus.utils.common import plot_metrics, plot_loss
from capreolus import evaluator

logger = get_logger(__name__)  # pylint: disable=invalid-name


class Trainer(ModuleBase, metaclass=RegisterableModule):
    module_type = "trainer"

    def get_paths_for_early_stopping(self, train_output_path, dev_output_path):
        os.makedirs(dev_output_path, exist_ok=True)
        dev_best_weight_fn = train_output_path / "dev.best"
        weights_output_path = train_output_path / "weights"
        info_output_path = train_output_path / "info"
        os.makedirs(weights_output_path, exist_ok=True)
        os.makedirs(info_output_path, exist_ok=True)

        loss_fn = info_output_path / "loss.txt"
        metrics_fn = dev_output_path / "metrics.json"

        return dev_best_weight_fn, weights_output_path, info_output_path, loss_fn


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
        softmaxloss = False  # True to use softmax loss (over pairs) or False to use hinge loss

        interactive = False  # True for training with Notebook or False for command line environment
        fastforward = False
        validatefreq = 1
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
        batches_per_epoch = self.cfg["itersize"] // self.cfg["batch"]
        batches_per_step = self.cfg["gradacc"]

        for bi, batch in enumerate(train_dataloader):
            # TODO make sure _prepare_batch_with_strings equivalent is happening inside the sampler
            batch = {k: v.to(self.device) if not isinstance(v, list) else v for k, v in batch.items()}
            doc_scores = reranker.score(batch)
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

    def fastforward_training(self, reranker, weights_path, loss_fn):
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

        if not (weights_path.exists() and loss_fn.exists()):
            return 0

        try:
            loss = self.load_loss_file(loss_fn)
        except IOError:
            return 0

        last_loss_iteration = len(loss) - 1
        weights_fn = weights_path / f"{last_loss_iteration}.p"

        try:
            reranker.load_weights(weights_fn, self.optimizer)
            return last_loss_iteration + 1
        except:
            logger.info("attempted to load weights from %s but failed, starting at iteration 0", weights_fn)
            return 0

    def train(self, reranker, train_dataset, train_output_path, dev_data, dev_output_path, qrels, metric):
        """Train a model following the trainer's config (specifying batch size, number of iterations, etc).

        Args:
           train_dataset (IterableDataset): training dataset
           train_output_path (Path): directory under which train_dataset runs and training loss will be saved
           dev_data (IterableDataset): dev dataset
           dev_output_path (Path): directory where dev_data runs and metrics will be saved

        """

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = reranker.model.to(self.device)
        self.optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad, model.parameters()), lr=self.cfg["lr"])

        if self.cfg["softmaxloss"]:
            self.loss = pair_softmax_loss
        else:
            self.loss = pair_hinge_loss

        dev_best_weight_fn, weights_output_path, info_output_path, loss_fn = self.get_paths_for_early_stopping(
            train_output_path, dev_output_path
        )

        initial_iter = self.fastforward_training(reranker, weights_output_path, loss_fn) if self.cfg["fastforward"] else 0
        logger.info("starting training from iteration %s/%s", initial_iter, self.cfg["niters"])

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.cfg["batch"], pin_memory=True, num_workers=0
        )

        train_loss = []
        # are we resuming training?
        if initial_iter > 0:
            train_loss = self.load_loss_file(loss_fn)

            # are we done training?
            if initial_iter < self.cfg["niters"]:
                logger.debug("fastforwarding train_dataloader to iteration %s", initial_iter)
                batches_per_epoch = self.cfg["itersize"] // self.cfg["batch"]
                for niter in range(initial_iter):
                    for bi, batch in enumerate(train_dataloader):
                        if (bi + 1) % batches_per_epoch == 0:
                            break

        dev_best_metric = -np.inf
        validation_frequency = self.cfg["validatefreq"]
        for niter in range(initial_iter, self.cfg["niters"]):
            model.train()

            iter_loss_tensor = self.single_train_iteration(reranker, train_dataloader)

            train_loss.append(iter_loss_tensor.item())
            logger.info("iter = %d loss = %f", niter, train_loss[-1])

            # write model weights to file
            weights_fn = weights_output_path / f"{niter}.p"
            reranker.save_weights(weights_fn, self.optimizer)
            # predict performance on dev set

            if niter % validation_frequency == 0:
                pred_fn = dev_output_path / f"{niter}.run"
                preds = self.predict(reranker, dev_data, pred_fn)

                # log dev metrics
                metrics = evaluator.eval_runs(preds, qrels, ["ndcg_cut_20", "map", "P_20"])
                logger.info("dev metrics: %s", " ".join([f"{metric}={v:0.3f}" for metric, v in sorted(metrics.items())]))

                # write best dev weights to file
                if metrics[metric] > dev_best_metric:
                    reranker.save_weights(dev_best_weight_fn, self.optimizer)

            # write train_loss to file
            loss_fn.write_text("\n".join(f"{idx} {loss}" for idx, loss in enumerate(train_loss)))

        print("training loss: ", train_loss)
        plot_loss(train_loss, str(loss_fn).replace(".txt", ".pdf"), interactive=self.cfg["interactive"])

    def load_best_model(self, reranker, train_output_path):
        self.optimizer = torch.optim.Adam(
            filter(lambda param: param.requires_grad, reranker.model.parameters()), lr=self.cfg["lr"]
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

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # save to pred_fn
        model = reranker.model.to(self.device)
        model.eval()

        preds = {}
        pred_dataloader = torch.utils.data.DataLoader(pred_data, batch_size=self.cfg["batch"], pin_memory=True, num_workers=0)
        with torch.autograd.no_grad():
            for batch in tqdm(pred_dataloader, desc="Predicting on dev"):
                batch = {k: v.to(self.device) if not isinstance(v, list) else v for k, v in batch.items()}
                scores = reranker.test(batch)
                scores = scores.view(-1).cpu().numpy()
                for qid, docid, score in zip(batch["qid"], batch["posdocid"], scores):
                    # Need to use float16 because pytrec_eval's c function call crashes with higher precision floats
                    preds.setdefault(qid, {})[docid] = score.astype(np.float16).item()

        os.makedirs(os.path.dirname(pred_fn), exist_ok=True)
        Searcher.write_trec_run(preds, pred_fn)

        return preds


class TensorFlowTrainer(Trainer):
    name = "pytorch"
    dependencies = {}
    def __init__(self, *args, **kwargs):
        super(TensorFlowTrainer, self).__init__(*args, **kwargs)

        # Use TPU if available, otherwise resort to GPU/CPU
        try:
            self.tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        except ValueError:
            self.tpu = None

        # TPUStrategy for distributed training
        if self.tpu:
            tf.config.experimental_connect_to_cluster(self.tpu)
            tf.tpu.experimental.initialize_tpu_system(self.tpu)
            self.strategy = tf.distribute.experimental.TPUStrategy(self.tpu)
        else:  # default strategy that works on CPU and single GPU
            self.strategy = tf.distribute.get_strategy()

        # Defining some props that we will alter initialize
        self.optimizer = self.get_optimizer()  # TODO: Accept a config param?
        self.loss = tf_pair_hinge_loss

    @staticmethod
    def config():
        maxdoclen = 800  # maximum document length (in number of terms after tokenization)
        maxqlen = 4  # maximum query length (in number of terms after tokenization)

        batch = 32  # batch size
        niters = 20  # number of iterations to train for
        itersize = 512  # number of training instances in one iteration (epoch)
        gradacc = 1  # number of batches to accumulate over before updating weights
        lr = 0.001  # learning rate
        softmaxloss = False  # True to use softmax loss (over pairs) or False to use hinge loss

        interactive = False  # True for training with Notebook or False for command line environment
        fastforward = False
        validatefreq = 1

    def fastforward_training(self, reranker, weights_path, loss_fn):
        return 0

    def load_best_model(self, reranker, train_output_path):
        raise NotImplementedError

    def train(self, reranker, train_dataset, train_output_path, dev_data, dev_output_path, qrels, metric):
        os.makedirs(dev_output_path, exist_ok=True)
        initial_iter = self.fastforward_training()
        logger.info("starting training from iteration %s/%s", initial_iter, self.cfg["niters"])

        train_records = reranker.convert_to_tf_record(train_dataset)
        dev_records = reranker.convert_to_tf_record(dev_data)

        validation_frequency = self.cfg["validatefreq"]
        dev_best_metric = np.inf
        for niter in range(initial_iter, self.cfg["niters"]):
            for step, (queries, posdocs, negdocs) in enumerate(train_records):
                with tf.GradientTape() as tape:
                    posdoc_scores = reranker.score(queries, posdocs)
                    negdoc_scores = reranker.score(queries, negdocs)
                    loss_value = self.loss(posdoc_scores, negdoc_scores)

                grads = tape.gradient(loss_value, reranker.model.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, reranker.model.trainable_weights))

            if niter % validation_frequency == 0:
                self.save_best_model(reranker, dev_records, dev_output_path, dev_best_metric, qrels, metric, niter)


        # Skipping dumping metrics and plotting loss since that should be done through tensorboard

    def save_best_model(self, reranker, dev_records, train_output_path, dev_output_path, dev_best_metric, qrels, metric, niter):
        """
        Attempt early stopping
        """
        dev_best_weight_fn, weights_output_path, info_output_path, loss_fn = self.get_paths_for_early_stopping(
            train_output_path, dev_output_path
        )

        pred_fn = dev_output_path / f"{niter}.run"
        preds = self.predict(reranker, dev_records, pred_fn)

        # log dev metrics
        metrics = evaluator.eval_runs(preds, qrels, ["ndcg_cut_20", "map", "P_20"])
        logger.info("dev metrics: %s", " ".join([f"{metric}={v:0.3f}" for metric, v in sorted(metrics.items())]))

        # write best dev weights to file
        if metrics[metric] > dev_best_metric:
            reranker.save_weights(dev_best_weight_fn, self.optimizer)

    def predict(self, reranker, dev_records, pred_fn):
        predictions = reranker.model.predict(dev_records)
        pred_dict = defaultdict(lambda: dict)
        for qid, docid, query, doc in dev_records:
            pred_dict[qid][docid] = reranker.score(query, doc)

        os.makedirs(os.path.dirname(pred_fn), exist_ok=True)
        Searcher.write_trec_run(pred_dict, pred_fn)

        return pred_dict

    def write_tf_record_to_file(self, qids, posdoc_ids, negdoc_ids, queries, posdocs, negdocs, query_idfs):
        filename = self.get_cache_path() / "{}.tfrecord".format(str(uuid.uuid4()))
        feature = {
            "qids": tf.train.Feature(int64_list=tf.train.Int64List(value=qids)),
            "queries": tf.train.Feature(float_list=tf.train.FloatList(value=queries)),
            "query_idfs": tf.train.Feature(float_list=tf.train.FloatList(value=query_idfs)),
            "posdoc_ids": tf.train.Feature(bytes_list=tf.train.BytesList(value=posdoc_ids)),
            "posdocs": tf.train.Feature(float_list=tf.train.FloatList(value=posdocs)),
        }

        if len(negdoc_ids):
            feature["negdoc_ids"] = tf.train.Feature(bytes_list=tf.train.BytesList(value=negdoc_ids)),
            feature["negdocs"] = tf.train.Feature(float_list=tf.train.FloatList(value=negdocs))

        tf_record = tf.train.Example(features=tf.train.Features(feature=feature))
        with tf.compat.v1.python_io.TFRecordWriter(filename) as outfile:
            outfile.write(tf_record.SerializeToString())

            return outfile

    def convert_to_tf_record(self, dataset):
        """
        Tensorflow works better if the input data is fed in as tfrecords
        Takes in a pytorch IterableDataset, iterates through it, and creates a tf record from it
        The randomization/shuffling e.t.c is handled by IterableDataset. See sampler/__init__.py
        """
        qids, posdoc_ids, negdoc_ids, queries, posdocs, negdocs, query_idfs = [], [], [], [], [], [], []
        tf_record_filenames = []

        for niter in tqdm(range(0, self.cfg["niters"]), desc="Converting data to tf records"):
            for sample_idx, data in enumerate(dataset):
                # TODO: Split into multiple files. This will destroy the RAM
                qids.append(data["qid"])
                posdoc_ids.append(data["posdocid"])
                queries.append(data["query"])
                query_idfs.append(data["query_idf"])
                posdocs.append(data["posdoc"])
                if data.get("negdocid"):
                    negdoc_ids.append(data["negdocid"])
                    negdocs.append(data["negdoc"])

            if (niter + 1) % 10 == 0:
                tf_record_filenames.append(self.write_tf_record_to_file(qids, posdoc_ids, negdoc_ids, queries, posdocs, negdocs, query_idfs))

        tf_record_filenames.append(
            self.write_tf_record_to_file(qids, posdoc_ids, negdoc_ids, queries, posdocs, negdocs, query_idfs)
        )
        return tf_record_filenames

    def cache_exists(self, dataset):
        cache_dir_name = dataset.get_hash()
        cache_dir_path = self.get_cache_path() / cache_dir_name
        # TODO: Add checks to make sure that the number of files in the director is correct
        return os.path.isdir(cache_dir_path) and len(os.listdir(cache_dir_path)) != 0

    def load_cached_dataset(self, dataset):
        cache_dir_path = self.get_cache_path() / dataset.get_hash()
        filenames = os.listdir(cache_dir_path)
        raw_dataset = tf.data.TFRecordDataset(filenames)
        feature_description = {
            'qids': tf.io.FixedLenFeature([], tf.int16),
            'queries': tf.io.FixedLenFeature([self.cfg["maxqlen"]], tf.float32),
            'query_idfs': tf.io.FixedLenFeature([self.cfg["maxqlen"]], tf.float32),
            'posdoc_ids': tf.io.FixedLenFeature([], tf.string),
            'posdoc': tf.io.FixedLenFeature([self.cfg["maxdoclen"]], tf.float32),
            'negdoc_ids': tf.io.FixedLenFeature([], tf.string),
            'negdocs': tf.io.FixedLenFeature([self.cfg['maxdoclen']], tf.float32)
        }
        parse_single_example and map it

    def get_tf_record_dataset(self, dataset):
        """
        If cached data exists, use that
        Else convert the dataset into tf record
        """

        if self.cfg["usecache"] and self.cache_exists():
            return self.load_cached_dataset()
        else:
            tf_record_filenames = self.convert_to_tf_record(dataset)

