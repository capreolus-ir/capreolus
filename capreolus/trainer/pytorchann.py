import torch
from tqdm import tqdm
import time
from capreolus import get_logger, ConfigOption
from capreolus.reranker.common import pair_hinge_loss, pair_softmax_loss

from . import Trainer

logger = get_logger(__name__)  # pylint: disable=invalid-name


@Trainer.register
class PytorchANNTrainer(Trainer):
    module_name = "pytorchann"
    config_spec = [
        ConfigOption("batch", 1, "batch size"),
        ConfigOption("niters", 1, "number of iterations to train for"),
        ConfigOption("itersize", 32, "number of training instances in one iteration"),
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

            cosine_scores = encoder.score(batch)
            loss = self.loss_function(cosine_scores)

            iter_loss.append(loss)
            loss.backward()

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


    def train(self, encoder, train_dataset, dev_dataset):
        validation_frequency = self.config["validatefreq"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder.model.to(self.device)
        self.optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad, encoder.model.parameters()), lr=self.config["lr"])
        self.loss_function = pair_hinge_loss
        num_workers = 1 if self.config["multithread"] else 0
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.config["batch"], pin_memory=True, num_workers=num_workers
        )
        
        train_loss = []
        for niter in range(self.config["niters"]):
            encoder.model.train()
            iter_start_time = time.time()
            iter_loss_tensor = self.single_train_iteration(encoder, train_dataloader)
            logger.info("A single iteration takes {}".format(time.time() - iter_start_time))
            train_loss.append(iter_loss_tensor.item())
            logger.info("iter = %d loss = %f", niter, train_loss[-1])

            if (niter + 1) % validation_frequency == 0:
                val_loss = self.validate(encoder, dev_dataset)
                logger.info("Validation loss is {}".format(val_loss.item()))

    def validate(self, encoder, dev_dataset):
        encoder.model.eval()
        num_workers = 1 if self.config["multithread"] else 0
        dev_dataloader = torch.utils.data.DataLoader(
            dev_dataset, batch_size=self.config["batch"], pin_memory=True, num_workers=num_workers
        )

        val_loss = []
        total = sum(len(docids) for docids in dev_dataset.qid_to_docids.values())

        with torch.autograd.no_grad():
            for bi, batch in tqdm(enumerate(dev_dataloader), desc="Validation set"):
                if bi * self.config["batch"] > total:
                    break
                batch = {k: v.to(self.device) if not isinstance(v, list) else v for k, v in batch.items()}
                cosine_scores = encoder.score(batch)
                loss = self.loss_function(cosine_scores)
                val_loss.append(loss)

        return torch.stack(val_loss).mean()
                



