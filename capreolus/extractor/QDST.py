import os
import torch
import pickle
import nltk
from collections import defaultdict
from transformers import RobertaTokenizer

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from capreolus import ConfigOption, Dependency, get_logger
from capreolus.utils.common import padlist
from capreolus.utils.exceptions import MissingDocError

from . import Extractor

logger = get_logger(__name__)


@Extractor.register
class QDSExtractor(Extractor):
    module_name = "qds"
    dependencies = [
        Dependency(
            key="index", module="index", name="anserini", default_config_overrides={"indexstops": True, "stemmer": "none"}
        )
    ]
    config_spec = [ConfigOption("maxqlen", 4), ConfigOption("maxdoclen", 800), ConfigOption("usecache", False)]

    pad = 1
    pad_tok = "<pad>"
    max_seq_len = 2048
    max_sent_num = 256

    def load_state(self, qids, docids):
        with open(self.get_state_cache_file_path(qids, docids), "rb") as f:
            state_dict = pickle.load(f)
            self.qid2text = state_dict["qid2text"]

    def cache_state(self, qids, docids):
        os.makedirs(self.get_cache_path(), exist_ok=True)
        with open(self.get_state_cache_file_path(qids, docids), "wb") as f:
            state_dict = {"qid2text": self.qid2text}
            pickle.dump(state_dict, f, protocol=-1)

    def _build_vocab(self, qids, docids, topics):
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        if self.is_state_cached(qids, docids) and self.config["usecache"]:
            self.load_state(qids, docids)
            logger.info("Vocabulary loaded from cache")
        else:
            logger.info("Building bertext vocabulary")
            self.qid2text = {qid: topics[qid] for qid in qids}
            self.cache_state(qids, docids)

    def exist(self):
        return hasattr(self, "qid2text") and len(self.qid2text)

    def preprocess(self, qids, docids, topics):
        self.index.create_index()
        self._build_vocab(qids, docids, topics)

    def get_sentences(self, doc_id):
        doc = self.index.get_doc(doc_id)
        sentences = nltk.sent_tokenize(doc)

        return sentences

    def id2vec(self, qid, posid, negid=None, label=None):
        assert posid is not None
        assert negid is None, "Don't try to train QDS Transformer. Only eval is supported right now"

        tokenizer = self.tokenizer
        query_text = self.qid2text[qid]
        query = tokenizer.encode(query_text)

        # Adding global attention token and the query tokens
        input_ids = [0] + query
        tokenized_query_length = len(input_ids)

        sentences = self.get_sentences(posid)
        for i, sent in enumerate(sentences):
            if i >= self.max_sent_num:
                break
            input_ids.append(0)
            input_ids.extend(self.tokenizer.encode(sent))

        input_ids.append(2)
        input_length = len(input_ids)
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[0: self.max_seq_len-1] + [2]
            input_length = len(input_ids)
        else:
            # The pad token id is 1 in roberta
            input_ids = padlist(input_ids, self.max_seq_len, 1)

        sos_idx = [i for i, x in enumerate(input_ids) if x == 0]
        tok_mask = torch.zeros(len(input_ids), dtype=torch.long)
        tok_mask[range(input_length)] = 1
        tok_mask[range(tokenized_query_length)] = 2
        tok_mask[sos_idx] = 2

        data = {
            "qid": qid,
            "posdocid": posid,
            "bert_input": torch.tensor(input_ids, dtype=torch.long),
            "mask": tok_mask
        }

        return data

