import torch
import faiss
import math
import torch.utils.data
import pickle
import numpy as np
import os

from torch.optim import AdamW
from tqdm import tqdm
import time

from transformers import get_linear_schedule_with_warmup

from capreolus import get_logger, ConfigOption, evaluator
from capreolus.reranker.common import pair_hinge_loss, pair_softmax_loss, multi_label_margin_loss

from . import Trainer
from capreolus.utils.common import pack_tensor_2D
from capreolus.utils.trec import max_pool_trec_passage_run

logger = get_logger(__name__)  # pylint: disable=invalid-name
faiss_logger = get_logger("faiss")


@Trainer.register
class PytorchANNTrainer(Trainer):
    module_name = "pytorchann"
    config_spec = [
        ConfigOption("batch", 8, "batch size"),
        ConfigOption("niters", 2, "number of iterations to train for"),
        ConfigOption("itersize", 2048, "number of training instances in one iteration"),
        ConfigOption("gradacc", 1, "number of batches to accumulate over before updating weights"),
        ConfigOption("lr", 0.001, "learning rate"),
        ConfigOption("bertlr", 0.00001, "learning rate"),
        ConfigOption("softmaxloss", False, "True to use softmax loss (over pairs) or False to use hinge loss"),
        ConfigOption("fastforward", False),
        ConfigOption("validatefreq", 2),
        ConfigOption(
            "multithread",
            False,
            "True to load data in a separate thread; faster but causes PyTorch deadlock in some environments",
        ),
        ConfigOption("boardname", "default"),
        ConfigOption("warmupsteps", 0),
        ConfigOption("decay", 0.0, "learning rate decay"),
        ConfigOption("decaystep", 3),
        ConfigOption("decaytype", None),
        ConfigOption("amp", None, "Automatic mixed precision mode; one of: None, train, pred, both"),
        ConfigOption("loss", "mlmargin")
    ]
    config_keys_not_in_path = ["boardname"]

    def single_train_iteration(self, encoder, train_dataloader):
        iter_loss = []
        batches_per_epoch = (self.config["itersize"] // self.config["batch"]) or 1
        batches_per_step = self.config["gradacc"]
        batches_since_update = 0

        for bi, batch in tqdm(enumerate(train_dataloader), desc="Training iteration", total=batches_per_epoch):
            batch = {k: v.to(self.device) if not isinstance(v, list) else v for k, v in batch.items()}

            output = encoder.score(batch)
            loss = self.loss(output)
            iter_loss.append(loss)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(encoder.model.parameters(), 1.0)
            batches_since_update += 1
            if batches_since_update == batches_per_step:
                batches_since_update = 0
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            if (bi + 1) % batches_per_epoch == 0:
                # REF-TODO: save scheduler state along with optimizer
                # self.lr_scheduler.step()
                break

        return torch.stack(iter_loss).mean()

    def load_trained_weights(self, encoder, output_path):
        encoder.instantiate_model()
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in encoder.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in encoder.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.config["bertlr"], eps=1e-8)
        weights_fn = output_path / "weights"

        if encoder.exists(weights_fn):
            encoder.load_weights(weights_fn, optimizer)
        else:
            raise ValueError("Weights not found: {}".format(weights_fn))

    def train(self, encoder, train_dataset, dev_dataset, output_path, qrels, metric="map", relevance_level=1):
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        # TODO: Do not hard-code pool_layer here
        optimizer_grouped_parameters = [
            {'params': [p for n, p in encoder.model.named_parameters() if 'pool_layer' not in n and not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in encoder.model.named_parameters() if 'pool_layer' not in n and any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]

        # TODO: Clean this. Hack for RepBERTTripletPooled
        if hasattr(encoder.model.module, "pool_layer"):
            logger.info("Setting a different learning rate for the pool layer")
            optimizer_grouped_parameters += [
                {'params': encoder.model.module.pool_layer.parameters(), 'lr': self.config["lr"]}
            ]

        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.config["bertlr"], eps=1e-8)

        steps_per_epoch = (self.config["itersize"] // (self.config["batch"] * self.config["gradacc"])) or 1
        total_steps = steps_per_epoch * self.config["niters"]
        num_warmup_steps = math.floor(0.1 * total_steps)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps)
        weights_fn = output_path / "weights"

        if encoder.exists(weights_fn):
            encoder.load_weights(weights_fn, self.optimizer)
            faiss_logger.warn("Skipping training since weights were found")
        else:
            self._train(encoder, train_dataset, dev_dataset, output_path, qrels, metric, relevance_level)

    def _train(self, encoder, train_dataset, dev_dataset, output_path, qrels, metric, relevance_level):
        if self.config["loss"] == "mlmargin":
            self.loss = multi_label_margin_loss
        elif self.config["loss"] == "pairwise_hinge_loss":
            self.loss = pair_hinge_loss

        validation_frequency = self.config["validatefreq"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder.model.to(self.device)
 
        num_workers = 1 if self.config["multithread"] else 0

        # RepBERT and RepBERTPretrained has implemented the collate method based on the original author's code
        collate_fn = encoder.collate if hasattr(encoder, "collate") else None
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.config["batch"], pin_memory=True, num_workers=num_workers, collate_fn=collate_fn
        )
        
        train_loss = []
        best_metric = -np.inf

        if self.config["niters"] == 0:
            # Useful when working with pre-trained weights
            weights_fn = output_path / "weights"
            encoder.save_weights(weights_fn, self.optimizer)
        else:
            for niter in range(self.config["niters"]):
                encoder.model.train()
                iter_start_time = time.time()
                iter_loss_tensor = self.single_train_iteration(encoder, train_dataloader)
                logger.info("A single iteration takes {}".format(time.time() - iter_start_time))
                train_loss.append(iter_loss_tensor.item())
                logger.info("iter = %d loss = %f", niter, train_loss[-1])
                faiss_logger.info("iter = %d loss = %f", niter, train_loss[-1])

                if (niter + 1) % validation_frequency == 0:
                    val_preds = self.validate(encoder, dev_dataset)
                    # TODO: This is a wasteful step for all non-passage datasets. Put it behind an if-condition maybe?
                    val_preds = max_pool_trec_passage_run(val_preds)
                    metrics = evaluator.eval_runs(val_preds, qrels, evaluator.DEFAULT_METRICS, relevance_level)
                    logger.info("dev metrics: %s", " ".join([f"{metric}={v:0.3f}" for metric, v in sorted(metrics.items())]))
                    faiss_logger.info("pytorch train dev metrics: %s", " ".join([f"{metric}={v:0.3f}" for metric, v in sorted(metrics.items())]))
                    if metrics["ndcg_cut_20"] > best_metric:
                        logger.debug("Best val set so far! Saving checkpoint")
                        best_metric = metrics["ndcg_cut_20"]
                        weights_fn = output_path / "weights"
                        encoder.save_weights(weights_fn, self.optimizer)

                        # TODO: This would fail for all non-huggingface models
                        # Will fail if dataparallel is not used
                        encoder.model.module.save_pretrained(output_path)

                # weights_fn = output_path / "weights_{}".format(train_dataset.get_hash())
                # encoder.save_weights(weights_fn, self.optimizer)
        logger.info("Encoder weights saved to {}".format(weights_fn))

    def validate(self, encoder, dev_dataset):
        encoder.model.eval()
        num_workers = 1 if self.config["multithread"] else 0
        BATCH_SIZE = 64
        dev_dataloader = torch.utils.data.DataLoader(
            dev_dataset, batch_size=BATCH_SIZE, pin_memory=True, num_workers=num_workers
        )

        qid_to_emb = {}
        sub_index = faiss.IndexFlatIP(encoder.hidden_size)
        faiss_index = faiss.IndexIDMap2(sub_index)
        faiss_id_to_doc_id = {}

        with torch.no_grad():
            for bi, batch in tqdm(enumerate(dev_dataloader), desc="Validation set"):
                batch = {k: v.to(self.device) if not isinstance(v, list) else v for k, v in batch.items()}
                doc_ids = batch["posdocid"]
                faiss_ids_for_batch = []

                for i, doc_id in enumerate(doc_ids):
                    generated_faiss_id = bi * BATCH_SIZE + i
                    faiss_id_to_doc_id[generated_faiss_id] = doc_id
                    faiss_ids_for_batch.append(generated_faiss_id)

                with torch.cuda.amp.autocast():
                    doc_emb = encoder.encode_doc(batch["posdoc"], batch["posdoc_mask"])

                doc_emb = doc_emb.cpu().numpy().astype(np.float32)
                faiss_ids_for_batch = np.array(faiss_ids_for_batch, dtype=np.long).reshape(-1, )
                faiss_index.add_with_ids(doc_emb, faiss_ids_for_batch)

                query_emb = encoder.encode_query(batch["query"], batch["query_mask"]).cpu().numpy()
                for qid, query_emb in zip(batch["qid"], query_emb):
                    qid_to_emb[qid] = query_emb

        query_vectors = np.array([emb for qid, emb in qid_to_emb.items()])
        distances, results = faiss_index.search(query_vectors, 1000)

        # Dicts in python 3.6 and above preserve insertion order
        qids = [qid for qid, emb in qid_to_emb.items()]

        return self.create_run_from_faiss_results(distances, results, qids, faiss_id_to_doc_id)

    def create_run_from_faiss_results(self, distances, results, qids, faiss_id_to_doc_id):
        num_queries, num_neighbours = results.shape
        run = {}

        for i in range(num_queries):
            qid = qids[i]

            faiss_ids = results[i][results[i] > -1]
            for j, faiss_id in enumerate(faiss_ids):
                doc_id = faiss_id_to_doc_id[faiss_id]
                run.setdefault(qid, {})[doc_id] = distances[i][j].item()

        return run




