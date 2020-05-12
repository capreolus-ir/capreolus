from transformers import BertTokenizer as HFBertTokenizer

from capreolus.tokenizer import Tokenizer


class BertTokenizer(Tokenizer):
    name = "bert"

    @staticmethod
    def config():
        pass

    def __init__(self, cfg):
        super().__init__(cfg)
        self.bert_tokenizer = HFBertTokenizer.from_pretrained(cfg["pretrained"])

    def convert_tokens_to_ids(self, tokens):
        return self.bert_tokenizer.convert_tokens_to_ids(tokens)

    def tokenize(self, sentences):
        if not sentences or len(sentences) == 0:  # either "" or []
            return []

        if isinstance(sentences, str):
            return self.bert_tokenizer.tokenize(sentences)

        return [self.bert_tokenizer.tokenize(s) for s in sentences]
