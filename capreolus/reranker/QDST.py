import torch
import pickle
from torch import nn
from transformers import LongformerModel
from capreolus import Reranker, Dependency, ConfigOption


class QDSTPretrained_Class():
    def __init__(self, extractor, config):
        super(QDSTPretrained_Class, self).__init__()
        self.window_size = 64
        self.required_graph = False
        self.construct_components()
        for i, layer in enumerate(self.base.encoder.layer):
            layer.attention.self.attention_window = self.window_size

    def get_name(self):
        return 'QDST'

    def construct_components(self):
        # Base model.
        self.base = LongformerModel.from_pretrained("allenai/longformer-base-4096")
        self.hidden_layer = nn.Linear(
            self.base.config.hidden_size,
            self.base.config.hidden_size, bias=True)
        self.final_layer = nn.Linear(self.base.config.hidden_size, 1, bias=True)
        self.dropout = nn.Dropout(self.base.config.hidden_dropout_prob)

    def forward(self, input_ids, tok_mask):
        '''
            input_ids: torch.long (batch_size, seq_len)
            tok_mask: torch.float (batch_size, seq_len)
            sent_locs: torch.long (batch_size, sent_num)
            sent_mask: torch.float (batch_size, sent_num, 1)
        '''
        # Get token embeddings (batch_size, seq_len, emb_dim).
        emb = self.base(input_ids, attention_mask=tok_mask)[1]

        self.final_features = emb
        # Output Head
        emb = self.dropout(emb)
        hidden = torch.tanh(self.hidden_layer(emb))
        self.final_hidden_features = emb

        hidden = self.dropout(hidden)
        logit = self.final_layer(hidden)

        return logit


@Reranker.register
class QDST(Reranker):
    """
    PyTorch implementation of QDS Transformer.

    Implementation of "Long Document Ranking with Query-Directed Sparse Transformer," Jyun-Yu Jiang, Chenyan Xiong, Chia-Jung Lee and Wei Wang.
    """

    module_name = "qdst"
    pretrained_weights = "/GW/NeuralIR/nobackup/kevin_cache/msmarco_saved/qdst/qdst_model"

    dependencies = [
        Dependency(key="extractor", module="extractor", name="pooledbertpassage"),
        Dependency(key="trainer", module="trainer", name="pytorch"),
    ]

    def build_model(self):
        if not hasattr(self, "model"):
            self.model = QDSTPretrained_Class(self.extractor, self.config)
            with open(self.pretrained_weights, "rb") as f:
               d = pickle.load(f)

            self.model.load_state_dict(d, strict=False)

        return self.model

    def score(self, d):
        raise NotImplementedError("Fine-tuning QDST is not supported yet")

    def test(self, d):
        return self.model(d["bert_input"], d["mask"]).view(-1)
