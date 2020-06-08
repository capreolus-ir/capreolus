from profane import import_all_modules

# import_all_modules(__file__, __package__)

import hashlib
import math
import os
import sys
import time
import uuid
from collections import defaultdict
from copy import copy

from profane import ModuleBase, Dependency, ConfigOption, constants
import tensorflow as tf
import tensorflow_ranking as tfr
import numpy as np
import torch
from keras import Sequential, layers
from keras.layers import Dense
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from capreolus.reranker.common import pair_hinge_loss, pair_softmax_loss
from capreolus.searcher import Searcher
from capreolus.utils.loginit import get_logger
from capreolus.utils.common import plot_metrics, plot_loss
from capreolus import evaluator

logger = get_logger(__name__)  # pylint: disable=invalid-name
RESULTS_BASE_PATH = constants["RESULTS_BASE_PATH"]


class Trainer(ModuleBase):
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
        metrics_fn = dev_output_path / "metrics.json"

        return dev_best_weight_fn, weights_output_path, info_output_path, loss_fn


@Trainer.register
class PytorchTrainer(Trainer):
    module_name = "pytorch"
    config_spec = [
        ConfigOption("batch", 32, "batch size"),
        ConfigOption("niters", 20, "number of iterations to train for"),
        ConfigOption("itersize", 512, "number of training instances in one iteration"),
        ConfigOption("gradacc", 1, "number of batches to accumulate over before updating weights"),
        ConfigOption("lr", 0.001, "learning rate"),
        ConfigOption("softmaxloss", False, "True to use softmax loss (over pairs) or False to use hinge loss"),
        ConfigOption("fastforward", False),
        ConfigOption("validatefreq", 1),
        ConfigOption("boardname", "default"),
    ]
    config_keys_not_in_path = ["fastforward", "boardname"]

    def build(self):
        # sanity checks
        if self.config["batch"] < 1:
            raise ValueError("batch must be >= 1")

        if self.config["niters"] <= 0:
            raise ValueError("niters must be > 0")

        if self.config["itersize"] < self.config["batch"]:
            raise ValueError("itersize must be >= batch")

        if self.config["gradacc"] < 1 or not float(self.config["gradacc"]).is_integer():
            raise ValueError("gradacc must be an integer >= 1")

        if self.config["lr"] <= 0:
            raise ValueError("lr must be > 0")

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
        batches_per_epoch = (self.config["itersize"] // self.config["batch"]) or 1
        batches_per_step = self.config["gradacc"]

        for bi, batch in tqdm(enumerate(train_dataloader), desc="Iter progression"):
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
        # Set up logging
        # TODO why not put this under train_output_path?
        summary_writer = SummaryWriter(RESULTS_BASE_PATH / "runs" / self.config["boardname"], comment=train_output_path)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = reranker.model.to(self.device)
        self.optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad, model.parameters()), lr=self.config["lr"])

        if self.config["softmaxloss"]:
            self.loss = pair_softmax_loss
        else:
            self.loss = pair_hinge_loss

        dev_best_weight_fn, weights_output_path, info_output_path, loss_fn = self.get_paths_for_early_stopping(
            train_output_path, dev_output_path
        )

        initial_iter = self.fastforward_training(reranker, weights_output_path, loss_fn) if self.config["fastforward"] else 0
        logger.info("starting training from iteration %s/%s", initial_iter, self.config["niters"])

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.config["batch"], pin_memory=True, num_workers=0
        )
        # dataiter = iter(train_dataloader)
        # sample_input = dataiter.next()
        # summary_writer.add_graph(
        #     reranker.model,
        #     [
        #         sample_input["query"].to(self.device),
        #         sample_input["posdoc"].to(self.device),
        #         sample_input["negdoc"].to(self.device),
        #     ],
        # )

        train_loss = []
        # are we resuming training?
        if initial_iter > 0:
            train_loss = self.load_loss_file(loss_fn)

            # are we done training?
            if initial_iter < self.config["niters"]:
                logger.debug("fastforwarding train_dataloader to iteration %s", initial_iter)
                batches_per_epoch = self.config["itersize"] // self.config["batch"]
                for niter in range(initial_iter):
                    for bi, batch in enumerate(train_dataloader):
                        if (bi + 1) % batches_per_epoch == 0:
                            break

        dev_best_metric = -np.inf
        validation_frequency = self.config["validatefreq"]
        train_start_time = time.time()
        for niter in range(initial_iter, self.config["niters"]):
            model.train()

            iter_start_time = time.time()
            iter_loss_tensor = self.single_train_iteration(reranker, train_dataloader)
            logger.info("A single iteration takes {}".format(time.time() - iter_start_time))
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
                summary_writer.add_scalar("ndcg_cut_20", metrics["ndcg_cut_20"], niter)
                summary_writer.add_scalar("map", metrics["map"], niter)
                summary_writer.add_scalar("P_20", metrics["P_20"], niter)
                # write best dev weights to file
                if metrics[metric] > dev_best_metric:
                    reranker.save_weights(dev_best_weight_fn, self.optimizer)

            # write train_loss to file
            loss_fn.write_text("\n".join(f"{idx} {loss}" for idx, loss in enumerate(train_loss)))

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

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # save to pred_fn
        model = reranker.model.to(self.device)
        model.eval()

        preds = {}
        pred_dataloader = torch.utils.data.DataLoader(pred_data, batch_size=self.config["batch"], pin_memory=True, num_workers=0)
        with torch.autograd.no_grad():
            for batch in tqdm(pred_dataloader, desc="Predicting on dev"):
                if len(batch["qid"]) != self.config["batch"]:
                    batch = self.fill_incomplete_batch(batch)

                batch = {k: v.to(self.device) if not isinstance(v, list) else v for k, v in batch.items()}
                scores = reranker.test(batch)
                scores = scores.view(-1).cpu().numpy()
                for qid, docid, score in zip(batch["qid"], batch["posdocid"], scores):
                    # Need to use float16 because pytrec_eval's c function call crashes with higher precision floats
                    preds.setdefault(qid, {})[docid] = score.astype(np.float16).item()

        os.makedirs(os.path.dirname(pred_fn), exist_ok=True)
        Searcher.write_trec_run(preds, pred_fn)

        return preds

    def fill_incomplete_batch(self, batch):
        """
        If a batch is incomplete (i.e shorter than the desired batch size), this method fills in the batch with some data.
        How the data is chosen:
        If the values are just a simple list, use the first element of the list to pad the batch
        If the values are tensors/numpy arrays, use repeat() along the batch dimension
        """
        # logger.debug("filling in an incomplete batch")
        repeat_times = math.ceil(self.config["batch"] / len(batch["qid"]))
        diff = self.config["batch"] - len(batch["qid"])

        def pad(v):
            if isinstance(v, np.ndarray) or torch.is_tensor(v):
                _v = v.repeat((repeat_times,) + tuple([1 for x in range(len(v.shape) - 1)]))
            else:
                _v = v + [v[0]] * diff

            return _v[: self.config["batch"]]

        batch = {k: pad(v) for k, v in batch.items()}
        return batch


class TrecCheckpointCallback(tf.keras.callbacks.Callback):
    """
    A callback that runs after every epoch and calculates pytrec_eval style metrics for the dev dataset.
    See TensorflowTrainer.train() for the invocation
    Also saves the best model to disk
    """

    def __init__(self, qrels, dev_data, dev_records, output_path, tb_logdir, validate_freq=1, *args, **kwargs):
        super(TrecCheckpointCallback, self).__init__(*args, **kwargs)
        """
        qrels - a qrels dict
        dev_data - a torch.utils.IterableDataset
        dev_records - a BatchedDataset instance 
        """
        self.best_metric = -np.inf
        self.qrels = qrels
        self.dev_data = dev_data
        self.dev_records = dev_records
        self.output_path = output_path
        self.iter_start_time = time.time()
        self.validate_freq = validate_freq
        self.tb_logdir = tb_logdir

    def save_model(self):
        self.model.save_weights("{0}/dev.best".format(self.output_path))

    def on_epoch_begin(self, epoch, logs=None):
        self.iter_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logger.debug("Epoch {} took {}".format(epoch, time.time() - self.iter_start_time))
        step_time = time.time() - self.iter_start_time
        file_writer = tf.summary.create_file_writer(self.tb_logdir)
        file_writer.set_as_default()
        tf.summary.scalar('step time', data=step_time, step=step_time)

        if (epoch + 1) % self.validate_freq == 0:
            predictions = self.model.predict(self.dev_records, verbose=1, workers=8, use_multiprocessing=True)
            trec_preds = self.get_preds_in_trec_format(predictions, self.dev_data)
            metrics = evaluator.eval_runs(trec_preds, dict(self.qrels), ["ndcg_cut_20", "map", "P_20"])
            logger.info("dev metrics: %s", " ".join([f"{metric}={v:0.3f}" for metric, v in sorted(metrics.items())]))

            # TODO: Make the metric configurable
            if metrics["ndcg_cut_20"] > self.best_metric:
                self.best_metric = metrics["ndcg_cut_20"]
                # TODO: Prevent the embedding layer weights from being saved
                self.save_model()

    @staticmethod
    def get_preds_in_trec_format(predictions, dev_data):
        """
        Takes in a list of predictions and returns a dict that can be fed into pytrec_eval
        As a side effect, also writes the predictions into a file in the trec format
        """
        pred_dict = defaultdict(lambda: dict())

        for i, (qid, docid) in enumerate(dev_data.get_qid_docid_pairs()):
            # Pytrec_eval has problems with high precision floats
            pred_dict[qid][docid] = predictions[i][0].astype(np.float16).item()

        return dict(pred_dict)


@Trainer.register
class TensorFlowTrainer(Trainer):
    module_name = "tensorflow"

    config_spec = [
        ConfigOption("batch", 32, "batch size"),
        ConfigOption("niters", 20, "number of iterations to train for"),
        ConfigOption("itersize", 512, "number of training instances in one iteration"),
        # ConfigOption("gradacc", 1, "number of batches to accumulate over before updating weights"),
        ConfigOption("lr", 0.001, "learning rate"),
        ConfigOption("loss", "pairwise_hinge_loss", "must be one of tfr.losses.RankingLossKey"),
        # ConfigOption("fastforward", False),
        ConfigOption("validatefreq", 1),
        ConfigOption("boardname", "default"),
        ConfigOption("usecache", False),
        ConfigOption("tpuname", None),
        ConfigOption("tpuzone", None),
        ConfigOption("storage", None),
    ]
    config_keys_not_in_path = ["fastforward", "boardname", "usecache", "tpuname", "tpuzone", "storage"]

    def build(self):
        tf.random.set_seed(self.config["seed"])

        # Use TPU if available, otherwise resort to GPU/CPU
        try:
            self.tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=self.config["tpuname"], zone=self.config["tpuzone"])
        except ValueError:
            self.tpu = None
            logger.info("Could not find the tpu")

        # TPUStrategy for distributed training
        if self.tpu:
            logger.info("Utilizing TPUs")
            tf.config.experimental_connect_to_cluster(self.tpu)
            tf.tpu.experimental.initialize_tpu_system(self.tpu)
            self.strategy = tf.distribute.experimental.TPUStrategy(self.tpu)
        else:  # default strategy that works on CPU and single GPU
            self.strategy = tf.distribute.get_strategy()

        # Defining some props that we will later initialize
        self.optimizer = None
        self.loss = None
        self.validate()

    def validate(self):
        if self.tpu and any([self.config["storage"] is None, self.config["tpuname"] is None, self.config["tpuzone"] is None]):
            raise ValueError("storage, tpuname and tpuzone configs must be provided when training on TPU")
        if self.tpu and self.config["storage"] and not self.config["storage"].startswith("gs://"):
            raise ValueError("For TPU utilization, the storage config should start with 'gs://'")

    def get_optimizer(self):
        return tf.keras.optimizers.Adam(learning_rate=self.config["lr"])

    def fastforward_training(self, reranker, weights_path, loss_fn):
        # TODO: Fix fast forwarding
        return 0

    def load_best_model(self, reranker, train_output_path):
        # TODO: Do the train_output_path modification at one place?
        if self.tpu:
            train_output_path = "{0}/{1}/{2}".format(
                self.config["storage"], "train_output", hashlib.md5(str(train_output_path).encode("utf-8")).hexdigest()
            )

        reranker.model.load_weights("{0}/dev.best".format(train_output_path))

    def apply_gradients(self, weights, grads):
        self.optimizer.apply_gradients(zip(grads, weights))

    def train(self, reranker, train_dataset, train_output_path, dev_data, dev_output_path, qrels, metric):
        # summary_writer = tf.summary.create_file_writer("{0}/capreolus_tensorboard/{1}".format(self.config["storage"], self.config["boardname"]))

        # Because TPUs can't work with local files
        if self.tpu:
            train_output_path = "{0}/{1}/{2}".format(
                self.config["storage"], "train_output", hashlib.md5(str(train_output_path).encode("utf-8")).hexdigest()
            )

        os.makedirs(dev_output_path, exist_ok=True)
        initial_iter = self.fastforward_training(reranker, dev_output_path, None)
        logger.info("starting training from iteration %s/%s", initial_iter, self.config["niters"])

        strategy_scope = self.strategy.scope()
        with strategy_scope:
            train_records = self.get_tf_train_records(reranker, train_dataset)
            dev_records = self.get_tf_dev_records(reranker, dev_data)
            tb_logdir = "{0}/capreolus_tensorboard/{1}".format(self.config["storage"], self.config["boardname"])
            trec_callback = TrecCheckpointCallback(qrels, dev_data, dev_records, train_output_path, tb_logdir, validate_freq=self.config["validatefreq"])
            tensorboard_callback = tf.keras.callbacks.TensorBoard(
               tb_logdir
            )
            reranker.build_model()  # TODO needed here?

            self.optimizer = self.get_optimizer()
            loss = tfr.keras.losses.get(self.config["loss"])
            reranker.model.compile(optimizer=self.optimizer, loss=loss)

            train_start_time = time.time()
            reranker.model.fit(
                train_records.prefetch(tf.data.experimental.AUTOTUNE),
                epochs=self.config["niters"],
                steps_per_epoch=self.config["itersize"],
                callbacks=[tensorboard_callback, trec_callback],
                workers=8,
                use_multiprocessing=True,
            )
            logger.info("Training took {}".format(time.time() - train_start_time))

            # Skipping dumping metrics and plotting loss since that should be done through tensorboard

    def create_tf_feature(self, qid, query, query_idf, posdoc_id, posdoc, negdoc_id, negdoc):
        """
        Creates a single tf.train.Feature instance (i.e, a single sample)
        """
        feature = {
            "qid": tf.train.Feature(bytes_list=tf.train.BytesList(value=[qid.encode("utf-8")])),
            "query": tf.train.Feature(float_list=tf.train.FloatList(value=query)),
            "query_idf": tf.train.Feature(float_list=tf.train.FloatList(value=query_idf)),
            "posdoc_id": tf.train.Feature(bytes_list=tf.train.BytesList(value=[posdoc_id.encode("utf-8")])),
            "posdoc": tf.train.Feature(float_list=tf.train.FloatList(value=posdoc)),
        }

        if negdoc_id:
            feature["negdoc_id"] = (tf.train.Feature(bytes_list=tf.train.BytesList(value=[negdoc_id.encode("utf-8")])),)
            feature["negdoc"] = tf.train.Feature(float_list=tf.train.FloatList(value=negdoc))

        return feature

    def write_tf_record_to_file(self, dir_name, tf_features):
        """
        Actually write the tf record to file. The destination can also be a gcs bucket.
        TODO: Use generators to optimize memory usage
        """
        filename = "{0}/{1}.tfrecord".format(dir_name, str(uuid.uuid4()))
        examples = [tf.train.Example(features=tf.train.Features(feature=feature)) for feature in tf_features]

        if not os.path.isdir(dir_name):
            os.makedirs(dir_name, exist_ok=True)

        examples = [example.SerializeToString() for example in examples]
        with tf.io.TFRecordWriter(filename) as writer:
            for example in examples:
                writer.write(example)

        logger.info("Wrote tf record file: {}".format(filename))

        return str(filename)

    def convert_to_tf_dev_record(self, reranker, dataset):
        """
        Similar to self.convert_to_tf_train_record(), but won't result in multiple files
        """
        dir_name = self.get_tf_record_cache_path(dataset)

        tf_features = [reranker.extractor.create_tf_feature(sample) for sample in dataset]

        return [self.write_tf_record_to_file(dir_name, tf_features)]

    def convert_to_tf_train_record(self, reranker, dataset):
        """
        Tensorflow works better if the input data is fed in as tfrecords
        Takes in a dataset,  iterates through it, and creates multiple tf records from it.
        The exact structure of the tfrecords is defined by reranker.extractor. For example, see EmbedText.get_tf_feature()
        """
        dir_name = self.get_tf_record_cache_path(dataset)

        total_samples = dataset.get_total_samples()
        tf_features = []
        tf_record_filenames = []

        for niter in tqdm(range(0, self.config["niters"]), desc="Converting data to tf records"):
            for sample_idx, sample in enumerate(dataset):
                tf_features.append(reranker.extractor.create_tf_feature(sample))

                if len(tf_features) > 20000:
                    tf_record_filenames.append(self.write_tf_record_to_file(dir_name, tf_features))
                    tf_features = []

                if sample_idx + 1 >= self.config["itersize"] * self.config["batch"]:
                    break

        if len(tf_features):
            tf_record_filenames.append(self.write_tf_record_to_file(dir_name, tf_features))

        return tf_record_filenames

    def get_tf_record_cache_path(self, dataset):
        """
        Get the path to the directory where tf records are written to.
        If using TPUs, this will be a gcs path.
        """
        if self.tpu:
            return "{0}/capreolus_tfrecords/{1}".format(self.config["storage"], dataset.get_hash())
        else:
            base_path = self.get_cache_path()
            return "{0}/{1}".format(base_path, dataset.get_hash())

    def cache_exists(self, dataset):
        # TODO: Add checks to make sure that the number of files in the directory is correct
        cache_dir = self.get_tf_record_cache_path(dataset)
        logger.info("The cache path is {0} and does it exist? : {1}".format(cache_dir, tf.io.gfile.exists(cache_dir)))

        return tf.io.gfile.isdir(cache_dir)

    def load_tf_records_from_file(self, reranker, filenames, batch_size):
        raw_dataset = tf.data.TFRecordDataset(filenames)
        tf_records_dataset = raw_dataset.batch(batch_size, drop_remainder=True).map(
            reranker.extractor.parse_tf_example, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        return tf_records_dataset

    def load_cached_tf_records(self, reranker, dataset, batch_size):
        logger.info("Loading TF records from cache")
        cache_dir = self.get_tf_record_cache_path(dataset)
        filenames = tf.io.gfile.listdir(cache_dir)
        filenames = ["{0}/{1}".format(cache_dir, name) for name in filenames]

        return self.load_tf_records_from_file(reranker, filenames, batch_size)

    def get_tf_dev_records(self, reranker, dataset):
        """
        1. Returns tf records from cache (disk) if applicable
        2. Else, converts the dataset into tf records, writes them to disk, and returns them
        """
        if self.config["usecache"] and self.cache_exists(dataset):
            return self.load_cached_tf_records(reranker, dataset, 1)
        else:
            tf_record_filenames = self.convert_to_tf_dev_record(reranker, dataset)
            # TODO use actual batch size here. see issue #52
            return self.load_tf_records_from_file(reranker, tf_record_filenames, 1)  # self.config["batch"])

    def get_tf_train_records(self, reranker, dataset):
        """
        1. Returns tf records from cache (disk) if applicable
        2. Else, converts the dataset into tf records, writes them to disk, and returns them
        """

        if self.config["usecache"] and self.cache_exists(dataset):
            return self.load_cached_tf_records(reranker, dataset, self.config["batch"])
        else:
            tf_record_filenames = self.convert_to_tf_train_record(reranker, dataset)
            return self.load_tf_records_from_file(reranker, tf_record_filenames, self.config["batch"])

    def predict(self, reranker, pred_data, pred_fn):
        """Predict query-document scores on `pred_data` using `model` and write a corresponding run file to `pred_fn`

        Args:
           model (Reranker): a PyTorch Reranker
           pred_data (IterableDataset): data to predict on
           pred_fn (Path): path to write the prediction run file to

        Returns:
           TREC Run

        """

        strategy_scope = self.strategy.scope()
        with strategy_scope:
            pred_records = self.get_tf_dev_records(reranker, pred_data)
            predictions = reranker.model.predict(pred_records)
            trec_preds = TrecCheckpointCallback.get_preds_in_trec_format(predictions, pred_data)

        os.makedirs(os.path.dirname(pred_fn), exist_ok=True)
        Searcher.write_trec_run(trec_preds, pred_fn)

        return trec_preds
