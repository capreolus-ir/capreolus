import torch
import pickle
import numpy as np
import os

from torch.optim import AdamW
from tqdm import tqdm
import time
from capreolus import get_logger, ConfigOption, evaluator
from capreolus.reranker.common import pair_hinge_loss, pair_softmax_loss

from . import Trainer
from ..utils.common import pack_tensor_2D

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
    ]
    config_keys_not_in_path = ["boardname"]

    def single_train_iteration(self, encoder, train_dataloader):
        iter_loss = []
        batches_per_epoch = (self.config["itersize"] // self.config["batch"]) or 1
        batches_per_step = self.config["gradacc"]
        batches_since_update = 0

        for bi, batch in tqdm(enumerate(train_dataloader), desc="Training iteration", total=batches_per_epoch):
            batch = {k: v.to(self.device) if not isinstance(v, list) else v for k, v in batch.items()}

            loss = encoder.score(batch)

            iter_loss.append(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.model.parameters(), 1.0)

            batches_since_update += 1
            if batches_since_update == batches_per_step:
                batches_since_update = 0
                self.optimizer.step()
                self.optimizer.zero_grad()

            if (bi + 1) % batches_per_epoch == 0:
                # REF-TODO: save scheduler state along with optimizer
                # self.lr_scheduler.step()
                break

        return torch.stack(iter_loss).mean()

    def train(self, encoder, train_dataset, dev_dataset, output_path, qrels, metric="map", relevance_level=1):
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in encoder.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in encoder.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.config["bertlr"], eps=1e-8)
        weights_fn = encoder.get_results_path() / "weights_{}".format(train_dataset.get_hash())

        if encoder.exists(weights_fn):
            encoder.load_weights(weights_fn, self.optimizer)
            faiss_logger.warn("Skipping training since weights were found")
        else:
            self._train(encoder, train_dataset, dev_dataset, output_path, qrels, metric, relevance_level)

    def _train(self, encoder, train_dataset, dev_dataset, output_path, qrels, metric, relevance_level):
        validation_frequency = self.config["validatefreq"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder.model.to(self.device)
 
        num_workers = 1 if self.config["multithread"] else 0
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.config["batch"], pin_memory=True, num_workers=num_workers, collate_fn=self.repbert_collate
        )
        
        train_loss = []
        best_metric = -np.inf

        if self.config["niters"] == 0:
            # Useful when working with pre-trained weights
            weights_fn = output_path / "weights_{}".format(train_dataset.get_hash())
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
                    metrics = evaluator.eval_runs(val_preds, qrels, evaluator.DEFAULT_METRICS, relevance_level)
                    logger.info("dev metrics: %s", " ".join([f"{metric}={v:0.3f}" for metric, v in sorted(metrics.items())]))
                    faiss_logger.info("pytorch train dev metrics: %s", " ".join([f"{metric}={v:0.3f}" for metric, v in sorted(metrics.items())]))
                    if metrics["ndcg_cut_20"] > best_metric:
                        logger.debug("Best val set so far! Saving checkpoint")
                        best_metric = metrics["ndcg_cut_20"]
                        weights_fn = output_path / "weights_{}".format(train_dataset.get_hash())
                        encoder.save_weights(weights_fn, self.optimizer)

                # weights_fn = output_path / "weights_{}".format(train_dataset.get_hash())
                # encoder.save_weights(weights_fn, self.optimizer)

    def validate(self, encoder, dev_dataset):
        encoder.model.eval()
        num_workers = 1 if self.config["multithread"] else 0
        dev_dataloader = torch.utils.data.DataLoader(
            dev_dataset, batch_size=self.config["batch"], pin_memory=True, num_workers=num_workers
        )

        preds = {}
        with torch.autograd.no_grad():
            for bi, batch in tqdm(enumerate(dev_dataloader), desc="Validation set"):
                batch = {k: v.to(self.device) if not isinstance(v, list) else v for k, v in batch.items()}
                scores = encoder.test(batch).cpu().numpy()
                for qid, docid, score in zip(batch["qid"], batch["posdocid"], scores):
                    preds.setdefault(qid, {})[docid] = score.astype(np.float16).item()

        return preds

    @staticmethod
    def repbert_collate(batch):
        input_ids_lst = [x["query"] + x["posdoc"] for x in batch]
        token_type_ids_lst = [[0] * len(x["query"]) + [1] * len(x["posdoc"])
                              for x in batch]
        valid_mask_lst = [[1] * len(input_ids) for input_ids in input_ids_lst]
        position_ids_lst = [list(range(len(x["query"]))) +
                            list(range(len(x["posdoc"]))) for x in batch]
        data = {
            "input_ids": pack_tensor_2D(input_ids_lst, default=0, dtype=torch.int64),
            "token_type_ids": pack_tensor_2D(token_type_ids_lst, default=0, dtype=torch.int64),
            "valid_mask": pack_tensor_2D(valid_mask_lst, default=0, dtype=torch.int64),
            "position_ids": pack_tensor_2D(position_ids_lst, default=0, dtype=torch.int64),
        }
        qid_lst = [x['qid'] for x in batch]
        docid_lst = [x['posdocid'] for x in batch]
        labels = [[j for j in range(len(docid_lst)) if docid_lst[j] in x['rel_docs']] for x in batch]
        data['labels'] = pack_tensor_2D(labels, default=-1, dtype=torch.int64, length=len(batch))

        return data, qid_lst, docid_lst
                



