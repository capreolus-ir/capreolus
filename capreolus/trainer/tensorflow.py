import hashlib
import os
import uuid
from collections import defaultdict
from copy import copy
from pathlib import Path

import tensorflow as tf
import tensorflow_ranking as tfr
import numpy as np
from tensorflow.python.keras import backend as K
from tqdm import tqdm

from capreolus.searcher import Searcher
from capreolus import ConfigOption, evaluator
from capreolus.trainer import Trainer
from capreolus.utils.loginit import get_logger
from capreolus.reranker.common import TFPairwiseHingeLoss, TFCategoricalCrossEntropyLoss, KerasPairModel, KerasTripletModel
from tensorflow.keras.mixed_precision import experimental as mixed_precision


logger = get_logger(__name__)

from tensorflow.python.client import device_lib


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == "GPU"]


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
        ConfigOption("evalbatch", 0, "batch size at inference time (or 0 to use training batch size)"),
        ConfigOption("niters", 20, "number of iterations to train for"),
        ConfigOption("itersize", 512, "number of training instances in one iteration"),
        ConfigOption("bertlr", 2e-5, "learning rate for bert parameters"),
        ConfigOption("lr", 0.001, "learning rate"),
        ConfigOption("warmupiters", 0),
        ConfigOption("loss", "pairwise_hinge_loss", "must be one of tfr.losses.RankingLossKey"),
        ConfigOption("fastforward", False),
        ConfigOption("validatefreq", 1),
        ConfigOption("boardname", "default"),
        ConfigOption("usecache", False),
        ConfigOption("tpuname", None),
        ConfigOption("tpuzone", None),
        ConfigOption("storage", None),
        ConfigOption("eager", False),
        ConfigOption("decay", 0.0, "learning rate decay"),
        ConfigOption("decayiters", 3),
        ConfigOption("decaytype", None),
        ConfigOption("amp", False, "use automatic mixed precision"),
    ]
    config_keys_not_in_path = ["fastforward", "boardname", "usecache", "tpuname", "tpuzone", "storage"]

    def build(self):
        tf.random.set_seed(self.config["seed"])

        self.evalbatch = self.config["evalbatch"] if self.config["evalbatch"] > 0 else self.config["batch"]

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
        elif len(get_available_gpus()) > 1:
            self.strategy = tf.distribute.MirroredStrategy()
        else:  # default strategy that works on CPU and single GPU
            self.strategy = tf.distribute.get_strategy()

        self.amp = self.config["amp"]
        if self.amp:
            policy = mixed_precision.Policy("mixed_bfloat16" if self.tpu else "mixed_float16")
            mixed_precision.set_policy(policy)

        # Defining some props that we will later initialize
        self.validate()

    def validate(self):
        if self.tpu and any([self.config["storage"] is None, self.config["tpuname"] is None, self.config["tpuzone"] is None]):
            raise ValueError("storage, tpuname and tpuzone configs must be provided when training on TPU")
        if self.tpu and self.config["storage"] and not self.config["storage"].startswith("gs://"):
            raise ValueError("For TPU utilization, the storage config should start with 'gs://'")

    def train(self, reranker, train_dataset, train_output_path, dev_data, dev_output_path, qrels, metric, relevance_level=1):
        if self.tpu:
            # WARNING: not sure if pathlib is compatible with gs://
            train_output_path = Path(
                "{0}/{1}/{2}".format(
                    self.config["storage"], "train_output", hashlib.md5(str(train_output_path).encode("utf-8")).hexdigest()
                )
            )

        dev_best_weight_fn, weights_output_path, info_output_path, loss_fn, metric_fn = self.get_paths_for_early_stopping(
            train_output_path, dev_output_path
        )

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

            # "You should remove the use of the LossScaleOptimizer when TPUs are used."
            if self.amp and not self.tpu:
                optimizer_2 = mixed_precision.LossScaleOptimizer(optimizer_2, loss_scale="dynamic")

            def compute_loss(labels, predictions):
                per_example_loss = loss_object(labels, predictions)
                return tf.nn.compute_average_loss(per_example_loss, global_batch_size=self.config["batch"])

        def is_bert_variable(name):
            if "bert" in name:
                return True
            if "electra" in name:
                return True
            return False

        def train_step(inputs):
            data, labels = inputs

            with tf.GradientTape() as tape:
                train_predictions = wrapped_model(data, training=True)
                loss = compute_loss(labels, train_predictions)
                if self.amp and not self.tpu:
                    loss = optimizer_2.get_scaled_loss(loss)

            gradients = tape.gradient(loss, wrapped_model.trainable_variables)
            if self.amp and not self.tpu:
                optimizer_2.get_unscaled_gradients(gradients)

            bert_variables = [
                (gradients[i], variable)
                for i, variable in enumerate(wrapped_model.trainable_variables)
                if is_bert_variable(variable.name) and "classifier" not in variable.name
            ]
            classifier_vars = [
                (gradients[i], variable)
                for i, variable in enumerate(wrapped_model.trainable_variables)
                if "classifier" in variable.name
            ]
            other_vars = [
                (gradients[i], variable)
                for i, variable in enumerate(wrapped_model.trainable_variables)
                if not is_bert_variable(variable.name) and "classifier" not in variable.name
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

        train_records = train_records.shuffle(100000)
        train_dist_dataset = self.strategy.experimental_distribute_dataset(train_records)

        initial_iter, metrics = (
            self.fastforward_training(wrapped_model, weights_output_path, loss_fn, metric_fn)
            if self.config["fastforward"]
            else (0, {})
        )
        dev_best_metric = metrics.get(metric, -np.inf)
        logger.info("starting training from iteration %s/%s", initial_iter + 1, self.config["niters"])
        logger.info(f"Best metric loaded: {metric}={dev_best_metric}")

        cur_step = initial_iter * self.n_batch_per_iter
        initial_lr = self.change_lr(step=cur_step, lr=self.config["bertlr"])
        K.set_value(optimizer_2.lr, K.get_value(initial_lr))
        train_loss = self.load_loss_file(loss_fn) if initial_iter > 0 else []
        if 0 < initial_iter < self.config["niters"]:
            self.exhaust_used_train_data(train_dist_dataset, n_batch_to_exhaust=initial_iter * self.n_batch_per_iter)

        niter = initial_iter
        total_loss = 0
        trec_preds = {}
        iter_bar = tqdm(desc="Training iteration", total=self.n_batch_per_iter)
        # Goes through the dataset ONCE (i.e niters * itersize).
        # However, the dataset may already contain multiple instances of the same sample,
        # depending upon what Sampler was used.
        # If you want multiple epochs, achieve it by tweaking the niters and itersize values.
        for x in train_dist_dataset:
            total_loss += distributed_train_step(x)
            cur_step += 1
            iter_bar.update(1)

            # Do warmup and decay
            new_lr = self.change_lr(step=cur_step, lr=self.config["bertlr"])
            K.set_value(optimizer_2.lr, K.get_value(new_lr))

            if cur_step % self.n_batch_per_iter == 0:
                niter += 1

                iter_bar.close()
                iter_bar = tqdm(total=self.n_batch_per_iter)
                train_loss.append(total_loss / self.n_batch_per_iter)
                logger.info("iter={} loss = {}".format(niter, train_loss[-1]))
                self.write_to_loss_file(loss_fn, train_loss)
                total_loss = 0

                if self.config["fastforward"]:
                    wrapped_model.save_weights(f"{weights_output_path}/{niter}")

                if niter % self.config["validatefreq"] == 0:
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
                    if metrics[metric] > dev_best_metric:
                        dev_best_metric = metrics[metric]
                        logger.info("new best dev metric: %0.4f", dev_best_metric)

                        self.write_to_metric_file(metric_fn, metrics)
                        wrapped_model.save_weights(dev_best_weight_fn)
                        Searcher.write_trec_run(trec_preds, outfn=(dev_output_path / "best").as_posix())

            if cur_step >= self.config["niters"] * self.n_batch_per_iter:
                break

        return trec_preds

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
        for x in tqdm(pred_dist_dataset, desc="validation"):
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
        total_samples = self.config["niters"] * self.config["itersize"]
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
                    return "{0}/{1}".format(parent_dir.rstrip("/"), child_dir)

            return None

    def get_tf_train_records(self, reranker, dataset):
        """
        1. Returns tf records from cache (disk) if applicable
        2. Else, converts the dataset into tf records, writes them to disk, and returns them
        """
        required_samples = self.config["niters"] * self.config["itersize"]
        cached_tf_record_dir = self.find_cached_tf_records(dataset, required_samples)

        if self.config["usecache"] and cached_tf_record_dir is not None:
            filenames = tf.io.gfile.listdir(cached_tf_record_dir)
            filenames = ["{0}/{1}".format(cached_tf_record_dir.rstrip("/"), name) for name in filenames]
        else:
            filenames = self.convert_to_tf_train_record(reranker, dataset)
        return self.load_tf_train_records_from_file(reranker, filenames, self.config["batch"])

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
        Creates exactly niters * itersize samples.
        The exact structure of the tfrecords is defined by reranker.extractor. For example, see BertPassage.get_tf_train_feature()
        params:
        reranker - A capreolus.reranker.Reranker instance
        dataset - A capreolus.sampler.Sampler instance
        """
        dir_name = self.form_tf_record_cache_path(dataset)

        tf_features = []
        tf_record_filenames = []
        required_sample_count = self.config["niters"] * self.config["itersize"]
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
            filenames = sorted(tf.io.gfile.listdir(cached_tf_record_dir), key=lambda x: int(x.replace(".tfrecord", "")))
            filenames = ["{0}/{1}".format(cached_tf_record_dir, name) for name in filenames]
        else:
            filenames = self.convert_to_tf_dev_record(reranker, dataset)
        return self.load_tf_dev_records_from_file(reranker, filenames, self.evalbatch)

    def load_tf_dev_records_from_file(self, reranker, filenames, batch_size):
        raw_dataset = tf.data.TFRecordDataset(filenames)
        tf_records_dataset = raw_dataset.batch(batch_size, drop_remainder=True).map(
            reranker.extractor.parse_tf_dev_example, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        return tf_records_dataset

    def convert_to_tf_dev_record(self, reranker, dataset):
        evalbatch = self.evalbatch
        dir_name = self.form_tf_record_cache_path(dataset)
        tf_features = []
        tf_record_filenames = []

        tf_file_id = 0
        element_to_copy = None
        for sample in dataset:
            tf_features.extend(reranker.extractor.create_tf_dev_feature(sample))
            if element_to_copy is None:
                element_to_copy = tf_features[0]
            if len(tf_features) > 20000:
                tf_record_filenames.append(self.write_tf_record_to_file(dir_name, tf_features, file_name=str(tf_file_id)))
                tf_features = []
                tf_file_id += 1

        # TPU's require drop_remainder = True. But we cannot drop things from validation dataset
        # As a workaroud, we pad the dataset with the last sample until it reaches the batch size.
        for i in range(evalbatch):
            tf_features.append(copy(element_to_copy))
        tf_record_filenames.append(self.write_tf_record_to_file(dir_name, tf_features, file_name=str(tf_file_id)))
        return tf_record_filenames

    def write_tf_record_to_file(self, dir_name, tf_features, file_name=None):
        """
        Actually write the tf record to file. The destination can also be a gcs bucket.
        TODO: Use generators to optimize memory usage
        """
        file_name = str(uuid.uuid4()) if not file_name else file_name
        filename = "{0}/{1}.tfrecord".format(dir_name, file_name)
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

    def fastforward_training(self, model, weights_path, loss_fn, best_metric_fn):
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
        weights_fn = weights_path / f"{last_loss_iteration}"

        try:
            model.load_weights(weights_fn)
            return last_loss_iteration + 1, metrics
        except:  # lgtm [py/catch-base-exception]
            logger.info("attempted to load weights from %s but failed, starting at iteration 0", weights_fn)
            return default_return_values

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
