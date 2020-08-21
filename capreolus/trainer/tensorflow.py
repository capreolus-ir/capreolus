import hashlib
import os
import uuid
from collections import defaultdict
from copy import copy
import tensorflow as tf
from tensorflow.python.keras import backend as K
import tensorflow_ranking as tfr
import numpy as np
from tqdm import tqdm

from capreolus.searcher import Searcher
from capreolus import ConfigOption, evaluator
from capreolus.trainer import Trainer
from capreolus.utils.loginit import get_logger
from capreolus.reranker.common import TFPairwiseHingeLoss, TFCategoricalCrossEntropyLoss, KerasPairModel, KerasTripletModel

logger = get_logger(__name__)


@Trainer.register
class TensorflowTrainer(Trainer):
    """
    Trains (optionally) on the TPU.
    Uses two optimizers with different learning rates - one for the BERT layers and another for the classifier layers.
    Configurable warmup and decay for bertlr.
    WARNING: The optimizers depend on specific layer names (see train()) - if your reranker does not have layers with
    'bert' in the name, the normal learning rate will be applied to it instead of the value supplied through the
    bertlr ConfigOption
    """

    module_name = "tensorflow"
    config_spec = [
        ConfigOption("batch", 32, "batch size"),
        ConfigOption("niters", 20, "number of iterations to train for"),
        ConfigOption("itersize", 512, "number of training instances in one iteration"),
        # ConfigOption("gradacc", 1, "number of batches to accumulate over before updating weights"),
        ConfigOption("bertlr", 2e-5, "learning rate for bert parameters"),
        ConfigOption("lr", 0.001, "learning rate"),
        ConfigOption("decay", 0.0, "learning rate decay"),
        ConfigOption("warmupsteps", 0),
        ConfigOption("loss", "pairwise_hinge_loss", "must be one of tfr.losses.RankingLossKey"),
        ConfigOption("validatefreq", 1),
        ConfigOption("boardname", "default"),
        ConfigOption("usecache", False),
        ConfigOption("tpuname", None),
        ConfigOption("tpuzone", None),
        ConfigOption("storage", None),
        ConfigOption("eager", False),
        ConfigOption("decaystep", 3),
        ConfigOption("decay", 0.96),
        ConfigOption("decaytype", None),
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
        self.validate()

    def validate(self):
        if self.tpu and any([self.config["storage"] is None, self.config["tpuname"] is None, self.config["tpuzone"] is None]):
            raise ValueError("storage, tpuname and tpuzone configs must be provided when training on TPU")
        if self.tpu and self.config["storage"] and not self.config["storage"].startswith("gs://"):
            raise ValueError("For TPU utilization, the storage config should start with 'gs://'")

    def train(self, reranker, train_dataset, train_output_path, dev_data, dev_output_path, qrels, metric, relevance_level=1):
        if self.tpu:
            train_output_path = "{0}/{1}/{2}".format(
                self.config["storage"], "train_output", hashlib.md5(str(train_output_path).encode("utf-8")).hexdigest()
            )

        os.makedirs(dev_output_path, exist_ok=True)

        train_records = self.get_tf_train_records(reranker, train_dataset)
        dev_records = self.get_tf_dev_records(reranker, dev_data)
        dev_dist_dataset = self.strategy.experimental_distribute_dataset(dev_records)

        # Does not very much from https://www.tensorflow.org/tutorials/distribute/custom_training
        strategy_scope = self.strategy.scope()
        with strategy_scope:
            reranker.build_model()
            wrapped_model = self.get_wrapped_model(reranker.model)
            loss_object = self.get_loss(self.config["loss"])
            optimizer_1 = tf.keras.optimizers.Adam(learning_rate=self.config["lr"])
            optimizer_2 = tf.keras.optimizers.Adam(learning_rate=self.config["bertlr"])

            def compute_loss(labels, predictions):
                per_example_loss = loss_object(labels, predictions)
                return tf.nn.compute_average_loss(per_example_loss, global_batch_size=self.config["batch"])

        def train_step(inputs):
            data, labels = inputs

            with tf.GradientTape() as tape:
                train_predictions = wrapped_model(data, training=True)
                loss = compute_loss(labels, train_predictions)

            gradients = tape.gradient(loss, wrapped_model.trainable_variables)

            # TODO: Expose the layer names to lookout for as a ConfigOption?
            # TODO: Crystina mentioned that hugging face models have 'bert' in all the layers (including classifiers). Handle this case
            bert_variables = [
                (gradients[i], variable)
                for i, variable in enumerate(wrapped_model.trainable_variables)
                if "bert" in variable.name and "classifier" not in variable.name
            ]
            classifier_vars = [
                (gradients[i], variable)
                for i, variable in enumerate(wrapped_model.trainable_variables)
                if "classifier" in variable.name
            ]
            other_vars = [
                (gradients[i], variable)
                for i, variable in enumerate(wrapped_model.trainable_variables)
                if "bert" not in variable.name and "classifier" not in variable.name
            ]

            assert len(bert_variables) + len(classifier_vars) + len(other_vars) == len(wrapped_model.trainable_variables)
            # TODO: Clean this up for general use
            # Making sure that we did not miss any variables
            optimizer_1.apply_gradients(classifier_vars)
            optimizer_2.apply_gradients(bert_variables)
            if other_vars:
                optimizer_1.apply_gradients(other_vars)

            return loss

        def test_step(inputs):
            data, labels = inputs
            predictions = wrapped_model.predict_step(data)

            return predictions

        @tf.function
        def distributed_train_step(dataset_inputs):
            per_replica_losses = self.strategy.run(train_step, args=(dataset_inputs,))

            return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

        @tf.function
        def distributed_test_step(dataset_inputs):
            return self.strategy.run(test_step, args=(dataset_inputs,))

        best_metric = -np.inf
        epoch = 0
        num_batches = 0
        total_loss = 0
        iter_bar = tqdm(total=self.config["itersize"])

        initial_lr = self.change_lr(epoch, self.config["bertlr"])
        K.set_value(optimizer_2.lr, K.get_value(initial_lr))
        train_records = train_records.shuffle(100000)
        train_dist_dataset = self.strategy.experimental_distribute_dataset(train_records)

        # Goes through the dataset ONCE (i.e niters * itersize * batch samples). However, the dataset may already contain multiple instances of the same sample,
        # depending upon what Sampler was used. If you want multiple epochs, achieve it by tweaking the niters and
        # itersize values.
        for x in train_dist_dataset:
            total_loss += distributed_train_step(x)
            train_loss = total_loss / num_batches
            num_batches += 1
            iter_bar.update(1)

            if num_batches % self.config["itersize"] == 0:
                epoch += 1

                # Do warmup and decay
                new_lr = self.change_lr(epoch, self.config["bertlr"])
                K.set_value(optimizer_2.lr, K.get_value(new_lr))

                iter_bar.close()
                iter_bar = tqdm(total=self.config["itersize"])
                logger.info("train_loss for epoch {} is {}".format(epoch, train_loss))
                train_loss = 0
                total_loss = 0

                if epoch % self.config["validatefreq"] == 0:
                    dev_predictions = []
                    for x in tqdm(dev_dist_dataset, desc="validation"):
                        pred_batch = (
                            distributed_test_step(x).values
                            if self.strategy.num_replicas_in_sync > 1
                            else [distributed_test_step(x)]
                        )
                        for p in pred_batch:
                            dev_predictions.extend(p)

                    trec_preds = self.get_preds_in_trec_format(dev_predictions, dev_data)
                    metrics = evaluator.eval_runs(trec_preds, dict(qrels), evaluator.DEFAULT_METRICS, relevance_level)
                    logger.info("dev metrics: %s", " ".join([f"{metric}={v:0.3f}" for metric, v in sorted(metrics.items())]))
                    if metrics[metric] > best_metric:
                        logger.info("Writing checkpoint")
                        best_metric = metrics[metric]
                        wrapped_model.save_weights("{0}/dev.best".format(train_output_path))

            if num_batches >= self.config["niters"] * self.config["itersize"]:
                break

    def predict(self, reranker, pred_data, pred_fn):
        pred_records = self.get_tf_dev_records(reranker, pred_data)
        pred_dist_dataset = self.strategy.experimental_distribute_dataset(pred_records)

        strategy_scope = self.strategy.scope()

        with strategy_scope:
            wrapped_model = self.get_wrapped_model(reranker.model)

        def test_step(inputs):
            data, labels = inputs
            predictions = wrapped_model.predict_step(data)

            return predictions

        @tf.function
        def distributed_test_step(dataset_inputs):
            return self.strategy.run(test_step, args=(dataset_inputs,))

        predictions = []
        for x in pred_dist_dataset:
            pred_batch = distributed_test_step(x).values if self.strategy.num_replicas_in_sync > 1 else [distributed_test_step(x)]
            for p in pred_batch:
                predictions.extend(p)

        trec_preds = self.get_preds_in_trec_format(predictions, pred_data)
        os.makedirs(os.path.dirname(pred_fn), exist_ok=True)
        Searcher.write_trec_run(trec_preds, pred_fn)

        return trec_preds

    def form_tf_record_cache_path(self, dataset):
        """
        Get the path to the directory where tf records are written to.
        If using TPUs, this will be a gcs path.
        """
        total_samples = self.config["niters"] * self.config["itersize"] * self.config["batch"]
        if self.tpu:
            return "{0}/capreolus_tfrecords/{1}_{2}".format(self.config["storage"], dataset.get_hash(), total_samples)
        else:
            base_path = self.get_cache_path()
            return "{0}/{1}_{2}".format(base_path, dataset.get_hash(), total_samples)

    def find_cached_tf_records(self, dataset, required_sample_count):
        """
        Looks for a tf record for the passed dataset that has at least the specified number of samples
        """
        parent_dir = (
            "{0}/capreolus_tfrecords/".format(self.config["storage"]) if self.tpu else "{0}".format(self.get_cache_path())
        )
        if not tf.io.gfile.exists(parent_dir):
            return None
        else:
            child_dirs = tf.io.gfile.listdir(parent_dir)
            required_prefix = dataset.get_hash()

            for child_dir in child_dirs:
                child_dir_ending = child_dir.split("_")[-1][-1]
                # The child dir will end with '/' if it's on gcloud, but not on local disk.
                if child_dir_ending == "/":
                    sample_count = int(child_dir.split("_")[-1][:-1])
                else:
                    sample_count = int(child_dir.split("_")[-1])

                prefix = "_".join(child_dir.split("_")[:-1])

                # TODO: Add checks to make sure that the child dir is not empty
                if prefix == required_prefix and sample_count >= required_sample_count:
                    return "{0}{1}".format(parent_dir, child_dir)

            return None

    def get_tf_train_records(self, reranker, dataset):
        """
        1. Returns tf records from cache (disk) if applicable
        2. Else, converts the dataset into tf records, writes them to disk, and returns them
        """
        required_samples = self.config["niters"] * self.config["itersize"] * self.config["batch"]
        cached_tf_record_dir = self.find_cached_tf_records(dataset, required_samples)

        if self.config["usecache"] and cached_tf_record_dir is not None:
            filenames = tf.io.gfile.listdir(cached_tf_record_dir)
            filenames = ["{0}{1}".format(cached_tf_record_dir, name) for name in filenames]

            return self.load_tf_train_records_from_file(reranker, filenames, self.config["batch"])
        else:
            tf_record_filenames = self.convert_to_tf_train_record(reranker, dataset)
            return self.load_tf_train_records_from_file(reranker, tf_record_filenames, self.config["batch"])

    def load_tf_train_records_from_file(self, reranker, filenames, batch_size):
        raw_dataset = tf.data.TFRecordDataset(filenames)
        tf_records_dataset = raw_dataset.batch(batch_size, drop_remainder=True).map(
            reranker.extractor.parse_tf_train_example, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        return tf_records_dataset

    def convert_to_tf_train_record(self, reranker, dataset):
        """
        Tensorflow works better if the input data is fed in as tfrecords
        Takes in a dataset,  iterates through it, and creates multiple tf records from it.
        Creates exactly niters * itersize * batch_size samples.
        The exact structure of the tfrecords is defined by reranker.extractor. For example, see BertPassage.get_tf_train_feature()
        params:
        reranker - A capreolus.reranker.Reranker instance
        dataset - A capreolus.sampler.Sampler instance
        """
        dir_name = self.form_tf_record_cache_path(dataset)

        tf_features = []
        tf_record_filenames = []
        required_sample_count = self.config["niters"] * self.config["itersize"] * self.config["batch"]
        sample_count = 0

        iter_bar = tqdm(total=required_sample_count)
        for sample in dataset:
            tf_features.extend(reranker.extractor.create_tf_train_feature(sample))
            if len(tf_features) > 20000:
                tf_record_filenames.append(self.write_tf_record_to_file(dir_name, tf_features))
                tf_features = []

            iter_bar.update(1)
            sample_count += 1
            if sample_count >= required_sample_count:
                break

        iter_bar.close()
        assert sample_count == required_sample_count, "dataset generator ran out before generating enough samples"
        if len(tf_features):
            tf_record_filenames.append(self.write_tf_record_to_file(dir_name, tf_features))

        return tf_record_filenames

    def get_tf_dev_records(self, reranker, dataset):
        """
        1. Returns tf records from cache (disk) if applicable
        2. Else, converts the dataset into tf records, writes them to disk, and returns them
        """
        cached_tf_record_dir = self.form_tf_record_cache_path(dataset)
        if self.config["usecache"] and tf.io.gfile.exists(cached_tf_record_dir):
            filenames = tf.io.gfile.listdir(cached_tf_record_dir)
            filenames = ["{0}/{1}".format(cached_tf_record_dir, name) for name in filenames]

            return self.load_tf_dev_records_from_file(reranker, filenames, self.config["batch"])
        else:
            tf_record_filenames = self.convert_to_tf_dev_record(reranker, dataset)
            # TODO use actual batch size here. see issue #52
            return self.load_tf_dev_records_from_file(reranker, tf_record_filenames, self.config["batch"])

    def load_tf_dev_records_from_file(self, reranker, filenames, batch_size):
        raw_dataset = tf.data.TFRecordDataset(filenames)
        tf_records_dataset = raw_dataset.batch(batch_size, drop_remainder=True).map(
            reranker.extractor.parse_tf_dev_example, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        return tf_records_dataset

    def convert_to_tf_dev_record(self, reranker, dataset):
        dir_name = self.form_tf_record_cache_path(dataset)
        tf_features = []
        tf_record_filenames = []

        for sample in dataset:
            tf_features.extend(reranker.extractor.create_tf_dev_feature(sample))
            if len(tf_features) > 20000:
                tf_record_filenames.append(self.write_tf_record_to_file(dir_name, tf_features))
                tf_features = []

        # TPU's require drop_remainder = True. But we cannot drop things from validation dataset
        # As a workaroud, we pad the dataset with the last sample until it reaches the batch size.
        if len(tf_features) % self.config["batch"]:
            num_elements_to_add = self.config["batch"] - (len(tf_features) % self.config["batch"])
            logger.debug("Number of elements to add in the last batch: {}".format(num_elements_to_add))
            element_to_copy = tf_features[-1]
            for i in range(num_elements_to_add):
                tf_features.append(copy(element_to_copy))

        if len(tf_features):
            tf_record_filenames.append(self.write_tf_record_to_file(dir_name, tf_features))

        return tf_record_filenames

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

    @staticmethod
    def get_preds_in_trec_format(predictions, dev_data):
        """
        Takes in a list of predictions and returns a dict that can be fed into pytrec_eval
        As a side effect, also writes the predictions into a file in the trec format
        """
        logger.debug("There are {} predictions".format(len(predictions)))
        pred_dict = defaultdict(lambda: dict())

        for i, (qid, docid) in enumerate(dev_data.get_qid_docid_pairs()):
            # Pytrec_eval has problems with high precision floats
            pred_dict[qid][docid] = predictions[i].numpy().astype(np.float16).item()

        return dict(pred_dict)

    def change_lr(self, epoch, lr):
        """
        Apply warm up or decay depending on the current epoch
        """
        warmup_steps = self.config["warmupsteps"]
        if warmup_steps and epoch <= warmup_steps:
            return min(lr * ((epoch + 1) / warmup_steps), lr)
        elif self.config["decaytype"] == "exponential":
            return lr * self.config["decay"] ** ((epoch - warmup_steps) / self.config["decaystep"])
        elif self.config["decaytype"] == "linear":
            return lr * (1 / (1 + self.config["decay"] * epoch))

        return lr

    def get_loss(self, loss_name):
        try:
            if loss_name == "pairwise_hinge_loss":
                loss = TFPairwiseHingeLoss(reduction=tf.keras.losses.Reduction.NONE)
            elif loss_name == "crossentropy":
                loss = TFCategoricalCrossEntropyLoss(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
            else:
                loss = tfr.keras.losses.get(loss_name)
        except ValueError:
            loss = tf.keras.losses.get(loss_name)

        return loss

    def get_wrapped_model(self, model):
        """
        We need a wrapped model because the logic changes slightly depending on whether the input is pointwise or pairwise:
        1. In case of pointwise input, there's no "negative document" - so in this case we just have to execute the model's call() method
        2. In case of pairwise input, we need to execute the model's call() method twice (one for positive doc and then again for negative doc), and then
        stack the results together before passing to a pairwise loss function.

        The alternative was to let the user manually configure everything, for example:
        `loss=crossentropy reranker.trainer.input=pairwise ...` - we already have too many ConfigOptions :shrug:
        """
        if self.config["loss"] == "crossentropy":
            return KerasPairModel(model)

        return KerasTripletModel(model)

    def fastforward_training(self, reranker, weights_path, loss_fn):
        # TODO: Fix fast forwarding
        return 0

    def load_best_model(self, reranker, train_output_path):
        # TODO: Do the train_output_path modification at one place?
        if self.tpu:
            train_output_path = "{0}/{1}/{2}".format(
                self.config["storage"], "train_output", hashlib.md5(str(train_output_path).encode("utf-8")).hexdigest()
            )

        reranker.build_model()
        # Because the saved weights are that of a wrapped model.
        wrapped_model = self.get_wrapped_model(reranker.model)
        wrapped_model.load_weights("{0}/dev.best".format(train_output_path))

        return wrapped_model.model
