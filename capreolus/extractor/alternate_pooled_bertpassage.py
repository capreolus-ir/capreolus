import numpy as np

from capreolus import Dependency, ConfigOption, get_logger, Extractor
from capreolus.extractor.pooled_bertpassage import PooledBertPassage
from capreolus.utils.common import padlist

logger = get_logger(__name__)


@Extractor.register
class AlternatePooledBertPassage(PooledBertPassage):
    """
    Sames as PooledBertPassage, but query and docs are encoded separately
    """

    module_name = "altpooledbertpassage"
    dependencies = [
        Dependency(
            key="index", module="index", name="anserini", default_config_overrides={"indexstops": True, "stemmer": "none"}
        ),
        Dependency(key="tokenizer", module="tokenizer", name="berttokenizer"),
    ]

    config_spec = [
        ConfigOption("maxseqlen", 256, "Maximum input length (query+document)"),
        ConfigOption("maxqlen", 20, "Maximum query length"),
        ConfigOption("usecache", False, "Should the extracted features be cached?"),
        ConfigOption("passagelen", 150, "Length of the extracted passage"),
        ConfigOption("stride", 100, "Stride"),
        ConfigOption("sentences", False, "Use a sentence tokenizer to form passages"),
        ConfigOption("numpassages", 16, "Number of passages per document"),
    ]

    def convert_to_bert_input(self, text_toks):
        maxseqlen, maxqlen = self.config["maxseqlen"], self.config["maxqlen"]
        text_toks = text_toks[: maxseqlen - 2]

        text_toks = " ".join(text_toks).split()  # in case that psg_toks is np.array
        input_line = [self.cls_tok] + text_toks + [self.sep_tok]
        padded_input_line = padlist(input_line, padlen=maxseqlen, pad_token=self.pad_tok)
        inp = self.tokenizer.convert_tokens_to_ids(padded_input_line)
        mask = [1] * len(input_line) + [0] * (len(padded_input_line) - len(input_line))

        return inp, mask

    def id2vec(self, qid, posid, negid=None, label=None):
        """
        See parent class for docstring
        """
        assert posid is not None

        pos_bert_inputs = []
        pos_bert_masks = []

        pos_passages = self._get_passages(posid)
        for tokenized_passage in pos_passages:
            inp, mask = self.convert_to_bert_input(tokenized_passage)
            pos_bert_inputs.append(inp)
            pos_bert_masks.append(mask)

        data = {
            "posdocid": posid,
            "posdoc": np.array(pos_bert_inputs, dtype=np.long),
            "posdoc_mask": np.array(pos_bert_masks, dtype=np.long),
        }
        if qid:
            query_toks = self.qid2toks[qid]
            query, query_mask = self.convert_to_bert_input(query_toks)
            data["qid"] = qid
            data["query"] = query
            data["query_mask"] = query_mask

        if negid:
            neg_bert_inputs, neg_bert_masks = [], []
            neg_passages = self._get_passages(negid)
            for tokenized_passage in neg_passages:
                inp, mask = self.convert_to_bert_input(tokenized_passage)
                neg_bert_inputs.append(inp)
                neg_bert_masks.append(mask)

            data["negdocid"] = negid
            data["negdoc"] = np.array(neg_bert_inputs, dtype=np.long)
            data["negdoc_mask"] = np.array(neg_bert_masks, dtype=np.long)

        return data

