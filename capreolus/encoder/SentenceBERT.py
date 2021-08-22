import torch
import torch.nn
import torch.nn.functional as F
from transformers import BertModel
from capreolus import get_logger
from capreolus.encoder import Encoder


logger = get_logger(__name__)


class SentenceBERTEncoder_Class(torch.nn.Module):
    def __init__(self):
        super(SentenceBERTEncoder_Class, self).__init__()

        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.hidden_size = self.bert.config.hidden_size

    def _average_sequence_embeddings(self, sequence_output, valid_mask):
        flags = valid_mask == 1
        lengths = torch.sum(flags, dim=-1)
        lengths = torch.clamp(lengths, 1, None)
        sequence_embeddings = torch.sum(sequence_output * flags[:, :, None], dim=1)
        sequence_embeddings = sequence_embeddings / lengths[:, None]

        return sequence_embeddings

    def forward(self, numericalized_text, mask=None):
        """
        `numericalized_text` has the shape (batch_size, text_len)
        """
        last_hidden_state, pooler_output = self.bert(input_ids=numericalized_text, attention_mask=mask)
        # last_hidden_state has the shape (batch_size, seq_len, hidden_size)
        # Average all the words in a text
        hidden_avg = self._average_sequence_embeddings(last_hidden_state, mask)
        # assert hidden_avg.shape == (1, self.hidden_size), "hidden avg shape is {}".format(hidden_avg.shape)

        return hidden_avg


@Encoder.register
class SentenceBERTEncoder(Encoder):
    module_name = "sentencebert"

    def instantiate_model(self):
        if not hasattr(self, "model"):
            self.model = torch.nn.DataParallel(SentenceBERTEncoder_Class())
            self.hidden_size = self.model.module.hidden_size

        return self.model
