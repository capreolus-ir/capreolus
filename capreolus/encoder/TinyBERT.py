import torch
import torch.nn
from transformers import BertModel
from capreolus import get_logger
from capreolus.encoder import Encoder


logger = get_logger(__name__)


class TinyBERTEncoder_class(torch.nn.Module):
    def __init__(self):
        super(TinyBERTEncoder_class, self).__init__()
        
        self.bert = BertModel.from_pretrained("prajjwal1/bert-tiny")

    def forward(self, numericalized_text):
        """
        `numericalized_text` has the shape (batch_size, text_len)
        """
        last_hidden_state, pooler_output = self.bert(input_ids=numericalized_text)
        # last_hidden_state has the shape (batch_size, seq_len, hidden_size)
        # Average all the words in a text
        hidden_avg = last_hidden_state[:, 0, :]
        assert hidden_avg.shape == (1, 128), "hidden avg shape is {}".format(hidden_avg.shape)
        
        
        return hidden_avg.reshape(-1, 128)


@Encoder.register
class TinyBERTEncoder(Encoder):
    module_name="tinybert"

    def build_model(self):
        if not hasattr(self, "model"):
            self.model = TinyBERTEncoder_class()
        
        return self.model

