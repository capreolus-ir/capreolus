import contextlib
import numpy as np
import torch
from torch import nn
from transformers import BertForNextSentencePrediction

from capreolus import ConfigOption, Dependency
from capreolus.reranker import Reranker
from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)


# official weights converted with:
# def convert(name):
#     from transformers import BertTokenizer, BertForNextSentencePrediction, TFBertForNextSentencePrediction

#     tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")

#     state = torch.load(f"/GW/NeuralIR/nobackup/birch-emnlp_bert4ir_v2/models/saved.{name}_1", map_location="cpu")

#     model = BertForNextSentencePrediction.from_pretrained("bert-large-uncased")
#     model.load_state_dict(state["model"].state_dict())

#     output = f"/GW/NeuralIR/nobackup/birch-emnlp_bert4ir_v2/models/export/birch-bert-large-{name}"
#     os.makedirs(output, exist_ok=True)
#     model.save_pretrained(output)
#     tokenizer.save_pretrained(output)

#     # tf2 support
#     tf_model = TFBertForNextSentencePrediction.from_pretrained(output, from_pt=True)
#     tf_model.save_pretrained(output)


class Birch_Class(nn.Module):
    def __init__(self, extractor, config):
        super().__init__()

        self.config = config

        if config["hidden"] == 0:
            self.combine = nn.Linear(config["topk"], 1, bias=False)
            with torch.no_grad():
                self.combine.weight = nn.Parameter(torch.ones_like(self.combine.weight) / config["topk"])
        else:
            assert config["hidden"] > 0
            self.combine = nn.Sequential(nn.Linear(config["topk"], config["hidden"]), nn.ReLU(), nn.Linear(config["hidden"], 1))

        # original model file (requires apex):
        # state = torch.load("/GW/NeuralIR/nobackup/birch-emnlp_bert4ir_v2/models/saved.msmarco_mb_1", map_location="cpu")
        # self.bert = state["model"]

        # saved.msmarco_mb_1 weights exported from the official apex model:
        # self.bert = BertForNextSentencePrediction.from_pretrained("bert-large-uncased")
        # self.bert.load_state_dict(torch.load("/GW/NeuralIR/nobackup/birch-emnlp_bert4ir_v2/models/converted"))
        # converted_weights.msmarco_mb

        # kevin's base model:
        # self.bert = BertForNextSentencePrediction.from_pretrained("bert-base-uncased")
        # saved_bert = torch.load("/GW/NeuralIR/nobackup/birch/models/saved.tmp_1")["model"]
        # self.bert.load_state_dict(saved_bert.state_dict())

        # also /GW/NeuralIR/nobackup/birch-emnlp_bert4ir_v2/models/export/birch-bert-base-kevin
        self.bert = BertForNextSentencePrediction.from_pretrained(f"Capreolus/birch-bert-large-{config['pretrained']}")

        if not config["finetune"]:
            self.bert.requires_grad = False
            self.bert_context = torch.no_grad
        else:
            self.bert_context = contextlib.nullcontext

    def forward(self, doc, seg, mask):
        batch = doc.shape[0]

        with self.bert_context():
            bi_scores = [self.score_passages(doc[bi], seg[bi], mask[bi], batch) for bi in range(batch)]
            scores = torch.stack(bi_scores)
            assert scores.shape == (batch, self.config["extractor"]["numpassages"], 2)
            scores = scores[:, :, 1]  # take second output

        topk, _ = torch.topk(scores, dim=1, k=self.config["topk"])
        doc_score = self.combine(topk)
        return doc_score

    def score_passages(self, doc, seg, mask, batch):
        needed_passages = doc.shape[0]
        maxlen = doc.shape[-1]

        # find instances that contain a document (segment B)
        # for unmasked tokens in seg B, seg+mask=2
        # there are always two SEPs in segment B, so the document is not empty if there are >= 3 tokens where seg+mask=2
        valid = ((seg + mask) == 2).sum(dim=1) > 2
        if not any(valid):
            valid[0] = True
        doc, seg, mask = doc[valid], seg[valid], mask[valid]

        out = []
        batches = np.ceil(doc.shape[0] / batch).astype(int)
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
            return torch.cat((real_out, pad_out), dim=0)
        else:
            return real_out


@Reranker.register
class Birch(Reranker):
    module_name = "birch"

    config_spec = [
        ConfigOption("topk", 3, "top k scores to use"),
        ConfigOption("hidden", 0, "size of hidden layer or 0 to take the weighted sum of the topk"),
        ConfigOption("finetune", False, "fine-tune the BERT model"),
        ConfigOption("pretrained", "msmarco_mb", "pretrained Birch model to load: mb, msmarco_mb, or car_mb"),
    ]
    dependencies = [
        Dependency(
            key="extractor",
            module="extractor",
            name="bertpassage",
            default_config_overrides={"tokenizer": {"pretrained": "bert-large-uncased"}},
        ),
        Dependency(key="trainer", module="trainer", name="pytorch"),
    ]

    def build_model(self):
        self.model = Birch_Class(self.extractor, self.config)
        return self.model

    def score(self, d):
        return [
            self.model(d["pos_bert_input"], d["pos_seg"], d["pos_mask"]).view(-1),
            self.model(d["neg_bert_input"], d["neg_seg"], d["neg_mask"]).view(-1),
        ]

    def test(self, d):
        return self.model(d["pos_bert_input"], d["pos_seg"], d["pos_mask"]).view(-1)
