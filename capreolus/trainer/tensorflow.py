import hashlib
import os
import time
import uuid
from collections import defaultdict

import numpy as np
import tensorflow as tf
import tensorflow_ranking as tfr
from tqdm import tqdm

from capreolus import ConfigOption, Searcher, constants, evaluator, get_logger

from . import Trainer

logger = get_logger(__name__)  # pylint: disable=invalid-name
RESULTS_BASE_PATH = constants["RESULTS_BASE_PATH"]


class TrecCheckpointCallback(tf.keras.callbacks.Callback):
    """
    A callback that runs after every epoch and calculates pytrec_eval style metrics for the dev dataset.
    See TensorflowTrainer.train() for the invocation
    Also saves the best model to disk
    """

    def __init__(self, qrels, dev_data, dev_records, output_path, metric, validate_freq, relevance_level, *args, **kwargs):
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
        self.metric = metric
        self.validate_freq = validate_freq
        self.relevance_level = relevance_level

    def save_model(self):
        self.model.save_weights("{0}/dev.best".format(self.output_path))

    def on_epoch_begin(self, epoch, logs=None):
        self.iter_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logger.debug("Epoch {} took {}".format(epoch, time.time() - self.iter_start_time))
        if (epoch + 1) % self.validate_freq == 0:
            predictions = self.model.predict(self.dev_records, verbose=1, workers=8, use_multiprocessing=True)
            trec_preds = self.get_preds_in_trec_format(predictions, self.dev_data)
            metrics = evaluator.eval_runs(trec_preds, dict(self.qrels), evaluator.DEFAULT_METRICS, self.relevance_level)
            logger.info("dev metrics: %s", " ".join([f"{metric}={v:0.3f}" for metric, v in sorted(metrics.items())]))

            if metrics[self.metric] > self.best_metric:
                self.best_metric = metrics[self.metric]
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

    def train(self, reranker, train_dataset, train_output_path, dev_data, dev_output_path, qrels, metric, relevance_level=1):
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
            trec_callback = TrecCheckpointCallback(
                qrels,
                dev_data,
                dev_records,
                train_output_path,
                metric,
                self.config["validatefreq"],
                relevance_level=relevance_level,
            )
            tensorboard_callback = tf.keras.callbacks.TensorBoard(
                log_dir="{0}/capreolus_tensorboard/{1}".format(self.config["storage"], self.config["boardname"])
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

        # total_samples = dataset.get_total_samples()
        tf_features = []
        tf_record_filenames = []

        for _ in tqdm(range(0, self.config["niters"]), desc="Converting data to tf records"):
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
