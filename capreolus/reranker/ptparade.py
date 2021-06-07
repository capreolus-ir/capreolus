import torch
from torch import nn
from transformers import BertModel, ElectraModel
from transformers.models.bert.modeling_bert import BertLayer

from capreolus import ConfigOption, Dependency
from capreolus.reranker import Reranker


class PTParade_Class(nn.Module):
    def __init__(self, extractor, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extractor = extractor
        self.config = config

        if config["pretrained"] == "electra-base-msmarco":
            self.bert = ElectraModel.from_pretrained("Capreolus/electra-base-msmarco")
        elif config["pretrained"] == "bert-base-msmarco":
            self.bert = BertModel.from_pretrained("Capreolus/bert-base-msmarco")
        elif config["pretrained"] == "bert-base-uncased":
            self.bert = BertModel.from_pretrained("bert-base-uncased")
        else:
            raise ValueError(
                f"unsupported model: {config['pretrained']}; need to ensure correct tokenizers will be used before arbitrary hgf models are supported"
            )

        self.transformer_layer_1 = BertLayer(self.bert.config)
        self.transformer_layer_2 = BertLayer(self.bert.config)
        self.num_passages = extractor.config["numpassages"]
        self.maxseqlen = extractor.config["maxseqlen"]
        self.linear = nn.Linear(self.bert.config.hidden_size, 1)

        if config["aggregation"] == "max":
            raise NotImplementedError()
        elif config["aggregation"] == "avg":
            raise NotImplementedError()
        elif config["aggregation"] == "attn":
            raise NotImplementedError()
        elif config["aggregation"] == "transformer":
            self.aggregation = self.aggregate_using_transformer
            input_embeddings = self.bert.get_input_embeddings()
            # TODO hardcoded CLS token id
            cls_token_id = torch.tensor([[101]])
            self.initial_cls_embedding = input_embeddings(cls_token_id).view(1, self.bert.config.hidden_size)
            self.full_position_embeddings = torch.zeros(
                (1, self.num_passages + 1, self.bert.config.hidden_size), requires_grad=True, dtype=torch.float
            )
            torch.nn.init.normal_(self.full_position_embeddings, mean=0.0, std=0.02)

            self.initial_cls_embedding = nn.Parameter(self.initial_cls_embedding, requires_grad=True)
            self.full_position_embeddings = nn.Parameter(self.full_position_embeddings, requires_grad=True)
        else:
            raise ValueError(f"unknown aggregation type: {self.config['aggregation']}")

    def aggregate_using_transformer(self, cls):
        expanded_cls = cls.view(-1, self.num_passages, self.bert.config.hidden_size)
        # TODO make sure batch size here is correct
        batch_size = expanded_cls.shape[0]
        tiled_initial_cls = self.initial_cls_embedding.repeat(batch_size, 1)
        merged_cls = torch.cat((tiled_initial_cls.view(batch_size, 1, self.bert.config.hidden_size), expanded_cls), dim=1)
        merged_cls = merged_cls + self.full_position_embeddings

        (transformer_out_1,) = self.transformer_layer_1(merged_cls, None, None, None)
        (transformer_out_2,) = self.transformer_layer_2(transformer_out_1, None, None, None)

        aggregated = transformer_out_2[:, 0, :]
        return aggregated

    def forward(self, doc_input, doc_mask, doc_seg):
        batch_size = doc_input.shape[0]
        doc_input = doc_input.view((batch_size * self.num_passages, self.maxseqlen))
        doc_mask = doc_mask.view((batch_size * self.num_passages, self.maxseqlen))
        doc_seg = doc_seg.view((batch_size * self.num_passages, self.maxseqlen))

        cls = self.bert(doc_input, attention_mask=doc_mask, token_type_ids=doc_seg)[0][:, 0, :]
        aggregated = self.aggregation(cls)

        return self.linear(aggregated)


@Reranker.register
class PTParade(Reranker):
    """
    PyTorch implementation of PARADE.

    PARADE: Passage Representation Aggregation for Document Reranking.
    Canjia Li, Andrew Yates, Sean MacAvaney, Ben He, and Yingfei Sun. arXiv 2020.
    https://arxiv.org/pdf/2008.09093.pdf
    """

    module_name = "ptparade"

    dependencies = [
        Dependency(key="extractor", module="extractor", name="pooledbertpassage"),
        Dependency(key="trainer", module="trainer", name="pytorch"),
    ]
    config_spec = [
        ConfigOption(
            "pretrained", "bert-base-uncased", "Pretrained model: bert-base-uncased, bert-base-msmarco, or electra-base-msmarco"
        ),
        ConfigOption("aggregation", "transformer"),
    ]

    def build_model(self):
        if not hasattr(self, "model"):
            self.model = PTParade_Class(self.extractor, self.config)
        return self.model

    def score(self, d):
        return [
            self.model(d["pos_bert_input"], d["pos_mask"], d["pos_seg"]).view(-1),
            self.model(d["neg_bert_input"], d["neg_mask"], d["neg_seg"]).view(-1),
        ]

    def test(self, d):
        return self.model(d["pos_bert_input"], d["pos_mask"], d["pos_seg"]).view(-1)
