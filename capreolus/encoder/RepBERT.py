import torch
from transformers import BertConfig
from capreolus.encoder import Encoder
from capreolus import ConfigOption
from capreolus.encoder.RepBERTPretrained import RepBERT_Class
from capreolus.utils.common import pack_tensor_2D


@Encoder.register
class RepBERT(Encoder):
    module_name = "repbert"
    config_spec = [
        ConfigOption(
            "pretrained",
            "bert-base-uncased",
            "Pretrained model: bert-base-uncased, bert-base-msmarco, electra-base, or electra-base-msmarco",
        )
    ]
    def instantiate_model(self):
        if not hasattr(self, "model"):
            config = BertConfig.from_pretrained(self.config["pretrained"])
            self.model = torch.nn.DataParallel(RepBERT_Class.from_pretrained(self.config["pretrained"], config=config))
            self.hidden_size = self.model.module.hidden_size

    def encode_doc(self, numericalized_text, mask):
        return self.model.module.predict(numericalized_text, mask)

    def encode_query(self, numericalized_text, mask):
        return self.model.module.predict(numericalized_text, mask, is_query=True)

    def score(self, batch):
        return self.model(batch["input_ids"], batch["token_type_ids"], batch["valid_mask"], batch["position_ids"], batch["labels"])

    def test(self, d):
        query = d["query"]
        query_mask = d["query_mask"]
        doc = d["posdoc"]
        doc_mask = d["posdoc_mask"]

        query_emb = self.model.module.predict(query, query_mask)
        doc_emb = self.model.module.predict(doc, doc_mask)

        return torch.diagonal(torch.matmul(query_emb, doc_emb.T))

    @staticmethod
    def collate(batch):
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
            "is_relevant": [x["is_relevant"] for x in batch]
        }
        qid_lst = [x['qid'] for x in batch]
        docid_lst = [x['posdocid'] for x in batch]
        # `labels` contain pointers to the samples in the batch (i.e indices)
        # It's saying "hey for this qid, the docs in these rows are the relevant ones"
        labels = [[j for j in range(len(docid_lst)) if docid_lst[j] in x['rel_docs']] for x in batch]
        data['labels'] = pack_tensor_2D(labels, default=-1, dtype=torch.int64, length=len(batch))

        return data
