import tensorflow as tf
import numpy as np

from capreolus import get_logger
from capreolus.utils.exceptions import MissingDocError
from . import Extractor
from .bertpassage import BertPassage
from .common import MultipleTrainingPassagesMixin

logger = get_logger(__name__)


@Extractor.register
class BirchBertPassage(MultipleTrainingPassagesMixin, BertPassage):
    module_name = "birchbertpassage"

    config_spec = BertPassage.config_spec

    def id2vec(self, qid, posid, negid=None, label=None, **kwargs):
        """
        See parent class for docstring
        """
        assert label is not None
        maxseqlen = self.config["maxseqlen"]
        numpassages = self.config["numpassages"]

        query_toks = self.qid2toks[qid]
        pos_bert_inputs, pos_bert_masks, pos_bert_segs = [], [], []

        # N.B: The passages in self.docid2passages are not bert tokenized
        pos_passages = self._get_passages(posid)
        for tokenized_passage in pos_passages:
            inp, mask, seg = self._prepare_bert_input(query_toks, tokenized_passage)
            pos_bert_inputs.append(inp)
            pos_bert_masks.append(mask)
            pos_bert_segs.append(seg)

        # TODO: Rename the posdoc key in the below dict to 'pos_bert_input'
        data = {
            "qid": qid,
            "posdocid": posid,
            "pos_bert_input": np.array(pos_bert_inputs, dtype=np.long),
            "pos_mask": np.array(pos_bert_masks, dtype=np.long),
            "pos_seg": np.array(pos_bert_segs, dtype=np.long),
            "negdocid": "",
            "neg_bert_input": np.zeros((numpassages, maxseqlen), dtype=np.long),
            "neg_mask": np.zeros((numpassages, maxseqlen), dtype=np.long),
            "neg_seg": np.zeros((numpassages, maxseqlen), dtype=np.long),
            "label": np.repeat(np.array([label], dtype=np.float32), numpassages, 0),
        }

        if not negid:
            return data

        neg_bert_inputs, neg_bert_masks, neg_bert_segs = [], [], []
        neg_passages = self._get_passages(negid)

        for tokenized_passage in neg_passages:
            inp, mask, seg = self._prepare_bert_input(query_toks, tokenized_passage)
            neg_bert_inputs.append(inp)
            neg_bert_masks.append(mask)
            neg_bert_segs.append(seg)

        if not neg_bert_inputs:
            raise MissingDocError(qid, negid)

        data["negdocid"] = negid
        data["neg_bert_input"] = np.array(neg_bert_inputs, dtype=np.long)
        data["neg_mask"] = np.array(neg_bert_masks, dtype=np.long)
        data["neg_seg"] = np.array(neg_bert_segs, dtype=np.long)
        return data
