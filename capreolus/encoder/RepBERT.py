import torch
from transformers import BertConfig
from capreolus.encoder import Encoder
from capreolus import ConfigOption
from capreolus.encoder.RepBERTPretrained import RepBERT_Class


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
