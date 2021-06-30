import pymagnitude # temporary ugly magic hack: import pymagnitude before transformers to avoid the segmentation fault on CC
from transformers import AutoTokenizer

from capreolus import ConfigOption, get_logger

from . import Tokenizer

logger = get_logger(__name__)


@Tokenizer.register
class BertTokenizer(Tokenizer):
    module_name = "berttokenizer"
    config_spec = [ConfigOption("pretrained", "bert-base-uncased", "pretrained model to load vocab from")]

    def build(self):
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.config["pretrained"], use_fast=True)
        # see supported tokenizers here: https://huggingface.co/transformers/model_doc/auto.html#transformers.AutoTokenizer

        # make sure we have cls_token and sep_token
        kwargs = {}
        if not self.bert_tokenizer.cls_token:
            kwargs["cls_token"] = "[CLS]"
        if not self.bert_tokenizer.sep_token:
            kwargs["sep_token"] = "[SEP]"

        if len(kwargs) > 0:
            logger.debug("adding missing tokens to vocab: %s", kwargs)
            self.bert_tokenizer = AutoTokenizer.from_pretrained(self.config["pretrained"], use_fast=True, **kwargs)

    def convert_tokens_to_ids(self, tokens):
        return self.bert_tokenizer.convert_tokens_to_ids(tokens)

    def tokenize(self, sentences):
        if not sentences or len(sentences) == 0:  # either "" or []
            return []

        if isinstance(sentences, str):
            return self.bert_tokenizer.tokenize(sentences)

        return [self.bert_tokenizer.tokenize(s) for s in sentences]
