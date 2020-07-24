import numpy as np
import torch
from torch import nn

from capreolus import ConfigOption, Dependency
from capreolus.reranker import Reranker
from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)


class Birch_Class(nn.Module):
    def __init__(self, extractor, config):
        super().__init__()

        self.config = config
        self.extractor = extractor
        self.topk = config["topk"]

        # /GW/NeuralIR/nobackup/birch-emnlp_bert4ir_v2/models/converted
        # state = torch.load("/GW/NeuralIR/nobackup/birch-emnlp_bert4ir_v2/models/saved.msmarco_mb_1", map_location="cpu")
        # self.bert = state["model"]
        self._load_bert()
        self.combine = nn.Linear(self.topk, 1)
        self.score_cache = {}

    def _load_bert(self):
        from transformers import BertTokenizer, BertForNextSentencePrediction

        bert = BertForNextSentencePrediction.from_pretrained("bert-base-uncased")
        saved_bert = torch.load("/GW/NeuralIR/nobackup/birch/models/saved.tmp_1")["model"]
        bert.load_state_dict(saved_bert.state_dict())
        self.bert = bert

    def forward(self, k, doc, seg, mask):
        B, P, D = doc.shape
        k = [x.item() for x in k.cpu()]
        # k = seg.view(B * P, -1)
        # doc = doc.view(B * P, D)
        # seg = seg.view(B * P, D)
        # mask = mask.view(B * P, D)

        with torch.no_grad():
            bi_scores = [self.score_passages(k[bi], doc[bi], seg[bi], mask[bi], B) for bi in range(B)]
            scores = torch.stack(bi_scores)
            assert scores.shape == (B, self.extractor.config["numpassages"], 2)
            scores = scores[:, :, 1]  # take second output

            # reset weights
            self.combine.weight = nn.Parameter(torch.ones_like(self.combine.weight))
            self.combine.bias = nn.Parameter(torch.ones_like(self.combine.bias))

        topk, _ = torch.topk(scores, dim=1, k=self.topk)

        # out = torch.sum(topk, dim=1, keepdims=True)

        out = self.combine(topk)
        return out

    def score_passages(self, k, doc, seg, mask, batch):
        # assert len(k) == 1
        # k = k.item()

        if k not in self.score_cache:
            self.score_cache[k] = self._score_passages(doc, seg, mask, batch)

        return self.score_cache[k].to(doc.device)

    def _score_passages(self, doc, seg, mask, batch):
        needed_passages = doc.shape[0]
        maxlen = doc.shape[-1]

        # find instances that contain a document (segment B)
        # for unmasked tokens in seg B, seg+mask=2
        # there are always two SEPs in segment B, so the document is not empty if there are >= 3 tokens where seg+mask=2
        valid = ((seg + mask) == 2).sum(dim=1) > 2
        if not any(valid):
            valid[0] = True
        doc, seg, mask = doc[valid], seg[valid], mask[valid]

        batches = np.ceil(doc.shape[0] / batch).astype(int)

        out = []
        for bi in range(batches):
            start = bi * batch
            stop = (bi + 1) * batch

            sb_doc, sb_seg, sb_mask = doc[start:stop], seg[start:stop], mask[start:stop]

            # find first non-padding token and shorten batch to this length
            idx = (sb_seg + sb_mask).argmax(dim=1).max()
            sb_doc = sb_doc[:, : idx + 1]
            sb_seg = sb_seg[:, : idx + 1]
            sb_mask = sb_mask[:, : idx + 1]
            # for idx in reversed(range(maxlen)):
            #     if any(sb_mask[:, idx]):
            #         sb_doc = sb_doc[:, : idx + 1]
            #         sb_seg = sb_seg[:, : idx + 1]
            #         sb_mask = sb_mask[:, : idx + 1]
            #         break

            sb_scores = self.bert(input_ids=sb_doc, token_type_ids=sb_seg, attention_mask=sb_mask)
            sb_scores = sb_scores[0]  # for new bert output
            out.append(sb_scores)

        real_out = torch.cat(out, dim=0)
        found_passages = real_out.shape[0]
        if found_passages < needed_passages:
            pad_out = torch.min(real_out, dim=0)[0].repeat(needed_passages - found_passages, 1)
            return torch.cat((real_out, pad_out), dim=0).cpu()
        else:
            return real_out


@Reranker.register
class Birch(Reranker):
    module_name = "birch"

    config_spec = [ConfigOption("topk", 3, "top k scores to use")]
    dependencies = [
        Dependency(
            key="extractor",
            module="extractor",
            name="bertpassage",
            default_config_overrides={"tokenizer": {"pretrained": "bert-base-uncased"}},
        ),
        Dependency(key="trainer", module="trainer", name="pytorch"),
    ]

    def build_model(self):
        self.model = Birch_Class(self.extractor, self.config)
        return self.model

    def score(self, d):
        return [
            self.model(d["poskey"], d["pos_bert_input"], d["pos_seg"], d["pos_mask"]).view(-1),
            self.model(d["negkey"], d["neg_bert_input"], d["neg_seg"], d["neg_mask"]).view(-1),
        ]

    def test(self, d):
        return self.model(d["poskey"], d["pos_bert_input"], d["pos_seg"], d["pos_mask"]).view(-1)
