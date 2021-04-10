import torch
import torch.nn.functional as F
from transformers import BertConfig, BertPreTrainedModel, BertModel
from capreolus.encoder import Encoder
from capreolus import ConfigOption


class RepBERTTriplet_Class(BertPreTrainedModel):
    """
    Adapted from https://github.com/jingtaozhan/RepBERT-Index/
    """
    def __init__(self, config):
        super(RepBERTTriplet_Class, self).__init__(config)
        self.bert = BertModel(config)
        self.hidden_size = self.bert.config.hidden_size
        self.init_weights()

    def forward(
            self, query, posdoc, negdoc, query_mask, posdoc_mask, negdoc_mask
    ):

        query_lengths = (query_mask != 0).sum(dim=1)
        posdoc_lengths = (posdoc_mask != 0).sum(dim=1)
        negdoc_lengths = (negdoc_mask != 0).sum(dim=1)
        query_output = self.bert(query, attention_mask=query_mask)[0]
        posdoc_output = self.bert(posdoc, attention_mask=posdoc_mask)[0]
        negdoc_output = self.bert(negdoc, attention_mask=negdoc_mask)[0]

        query_embedding = torch.sum(query_output, dim=1) / query_lengths
        posdoc_embedding = torch.sum(posdoc_output, dim=1) / posdoc_lengths
        negdoc_embedding = torch.sum(negdoc_output, dim=1) / negdoc_lengths

        posdoc_score = F.cosine_similarity(query_embedding, posdoc_embedding, dim=1)
        negdoc_score = F.cosine_similarity(query_embedding, negdoc_embedding, dim=1)

        return posdoc_score, negdoc_score

    def predict(self, input_ids, valid_mask, is_query=False):
        input_lengths = (valid_mask != 0).sum(dim=1)

        sequence_output = self.bert(input_ids,
                                    attention_mask=valid_mask,
                                    )[0]

        text_embeddings = torch.sum(sequence_output, dim=1) / input_lengths

        return text_embeddings


@Encoder.register
class RepBERTTriplet(Encoder):
    module_name = "repberttriplet"
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
            self.model = torch.nn.DataParallel(RepBERTTriplet_Class.from_pretrained(self.config["pretrained"], config=config))
            self.hidden_size = self.model.module.hidden_size

    def encode_doc(self, numericalized_text, mask):
        return self.model.module.predict(numericalized_text, mask)

    def encode_query(self, numericalized_text, mask):
        return self.model.module.predict(numericalized_text, mask)

    def score(self, batch):
        return self.model(batch["query"], batch["posdoc"], batch["negdoc"], batch["query_mask"], batch["posdoc_mask"], batch["negdoc_mask"])

    def test(self, d):
        query = d["query"]
        query_mask = d["query_mask"]
        doc = d["posdoc"]
        doc_mask = d["posdoc_mask"]

        query_emb = self.model.module.predict(query, query_mask)
        doc_emb = self.model.module.predict(doc, doc_mask)

        return torch.diagonal(torch.matmul(query_emb, doc_emb.T))
