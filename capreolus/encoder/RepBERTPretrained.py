import torch
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
        self.init_weights()

    def _average_sequence_embeddings(self, sequence_output, valid_mask):
        flags = valid_mask == 1
        lengths = torch.sum(flags, dim=-1)
        lengths = torch.clamp(lengths, 1, None)
        sequence_embeddings = torch.sum(sequence_output * flags[:, :, None], dim=1)
        sequence_embeddings = sequence_embeddings / lengths[:, None]

        return sequence_embeddings

    def forward(self, input_ids, valid_mask, is_query=False):
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
    pretrained_weights_fn = "/GW/NeuralIR/nobackup/msmarco_saved/repbert.ckpt-350000"

    def instantiate_model(self):
        if not hasattr(self, "model"):
            config = BertConfig.from_pretrained(self.pretrained_weights_fn)
            self.model = RepBERT_Class.from_pretrained(self.pretrained_weights_fn, config=config)

    def encode_doc(self, numericalized_text, mask):
        return self.model(numericalized_text, mask)

    def encode_query(self, numericalized_text, mask):
        return self.model(numericalized_text, mask, is_query=True)