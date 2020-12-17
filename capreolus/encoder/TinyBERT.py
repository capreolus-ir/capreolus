import torch
import torch.nn
from transformers import BertTokenizerFast, BertModel
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
        hidden_avg = torch.mean(last_hidden_state, dim=1)
        
        return hidden_avg


@Encoder.register
class TinyBERTEncoder(Encoder):
    module_name="tinybert"

    def build_model(self):
        if not hasattr(self, "model"):
            self.model = TinyBERTEncoder_class()
            self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        
        return self.model

    def encode(self, text):
        if not hasattr(self, "tokenizer") or not hasattr(self, "model"):
            raise Exception("You should call encoder.build_model() first")

        numericalized_text = torch.tensor(self.tokenizer.encode(text))
        numericalized_text = numericalized_text.reshape(1, -1)
        logger.info("Original text is {}".format(text))
        logger.info("Numericalized text is {}".format(numericalized_text))

        return self.model(numericalized_text).numpy()
   
