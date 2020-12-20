import torch
import torch.nn
from transformers import BertTokenizerFast, BertModel
from capreolus import get_logger
from capreolus.encoder import Encoder


logger = get_logger(__name__)


class TinyBERTEncoder_class(torch.nn.Module):
    def __init__(self):
        super(TinyBERTEncoder_class, self).__init__()
        
        self.bert = BertModel.from_pretrained("bert-base-uncased")

    def forward(self, numericalized_text):
        """
        `numericalized_text` has the shape (batch_size, text_len)
        """
        last_hidden_state, pooler_output = self.bert(input_ids=numericalized_text)
        # last_hidden_state has the shape (batch_size, seq_len, hidden_size)
        # Average all the words in a text
        hidden_avg = last_hidden_state[:, 0, :]
        assert hidden_avg.shape == (1, 768)
        
        
        return hidden_avg.reshape(-1, 768)


@Encoder.register
class TinyBERTEncoder(Encoder):
    module_name="tinybert"

    def build_model(self):
        if not hasattr(self, "model"):
            self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model = TinyBERTEncoder_class().to(self.device)
        
        return self.model

    def encode(self, text):
        if not hasattr(self, "tokenizer") or not hasattr(self, "model"):
            raise Exception("You should call encoder.build_model() first")

        tokenizer = self.tokenizer

        tokenized_text = tokenizer.tokenize(text)
        tokenized_text = tokenized_text[:509]  # Make it fit the BERT input
        tokenized_text = [tokenizer.cls_token] + tokenized_text + [tokenizer.sep_token]
        numericalized_text = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        numericalized_text = torch.tensor(numericalized_text).to(self.device)
        numericalized_text = numericalized_text.reshape(1, -1)

        return self.model(numericalized_text).cpu().numpy()
   
