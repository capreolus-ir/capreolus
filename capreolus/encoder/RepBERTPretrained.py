import torch
import torch.nn.functional as F
from torch import nn
import os
from transformers import BertPreTrainedModel, BertModel, BertConfig
from capreolus import get_logger
from capreolus.utils.common import download_file
from capreolus.encoder import Encoder


class RepBERT_Class(BertPreTrainedModel):
    """
    Adapted from https://github.com/jingtaozhan/RepBERT-Index/
    """
    def __init__(self, config):
        super(RepBERT_Class, self).__init__(config)
        self.bert = BertModel(config)
        self.hidden_size = self.bert.config.hidden_size
        self.init_weights()

    def _average_query_doc_embeddings(self, sequence_output, token_type_ids, valid_mask):
        query_flags = (token_type_ids == 0) * (valid_mask == 1)
        doc_flags = (token_type_ids == 1) * (valid_mask == 1)

        query_lengths = torch.sum(query_flags, dim=-1)
        query_lengths = torch.clamp(query_lengths, 1, None)
        doc_lengths = torch.sum(doc_flags, dim=-1)
        doc_lengths = torch.clamp(doc_lengths, 1, None)

        query_embeddings = torch.sum(sequence_output * query_flags[:, :, None], dim=1)
        query_embeddings = query_embeddings / query_lengths[:, None]
        doc_embeddings = torch.sum(sequence_output * doc_flags[:, :, None], dim=1)
        doc_embeddings = doc_embeddings / doc_lengths[:, None]
        return query_embeddings, doc_embeddings

    def _mask_both_directions(self, valid_mask, token_type_ids):
        assert valid_mask.dim() == 2
        attention_mask = valid_mask[:, None, :]

        type_attention_mask = torch.abs(token_type_ids[:, :, None] - token_type_ids[:, None, :])
        attention_mask = attention_mask - type_attention_mask
        attention_mask = torch.clamp(attention_mask, 0, None)
        return attention_mask

    def _average_sequence_embeddings(self, sequence_output, valid_mask):
        flags = valid_mask == 1
        lengths = torch.sum(flags, dim=-1)
        lengths = torch.clamp(lengths, 1, None)
        sequence_embeddings = torch.sum(sequence_output * flags[:, :, None], dim=1)
        sequence_embeddings = sequence_embeddings / lengths[:, None]

        return sequence_embeddings

    def forward(self, input_ids, token_type_ids, valid_mask,
                position_ids, labels=None):
        attention_mask = self._mask_both_directions(valid_mask, token_type_ids)

        sequence_output = self.bert(input_ids,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids,
                                    position_ids=position_ids)[0]

        query_embeddings, doc_embeddings = self._average_query_doc_embeddings(
            sequence_output, token_type_ids, valid_mask
        )

        similarities = torch.matmul(query_embeddings, doc_embeddings.T)

        output = (similarities, query_embeddings, doc_embeddings)
        if labels is not None:
            loss_fct = nn.MultiLabelMarginLoss()
            loss = loss_fct(similarities, labels)

        return loss

    def predict(self, input_ids, valid_mask, is_query=False):
        if is_query:
            token_type_ids = torch.zeros_like(input_ids)
        else:
            token_type_ids = torch.ones_like(input_ids)

        sequence_output = self.bert(input_ids,
                                    attention_mask=valid_mask,
                                    token_type_ids=token_type_ids)[0]

        text_embeddings = self._average_sequence_embeddings(
            sequence_output, valid_mask
        )

        return text_embeddings


@Encoder.register
class RepBERTPretrained(Encoder):
    module_name = "repbertpretrained"
    pretrained_weights_fn = "/GW/NeuralIR/nobackup/kevin_cache/msmarco_saved/repbert.ckpt-350000"

    def instantiate_model(self):
        if not hasattr(self, "model"):
            config = BertConfig.from_pretrained(self.pretrained_weights_fn)
            self.model = torch.nn.DataParallel(RepBERT_Class.from_pretrained(self.pretrained_weights_fn, config=config))
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
