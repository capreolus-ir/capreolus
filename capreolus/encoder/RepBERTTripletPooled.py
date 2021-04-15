import torch
from capreolus import ConfigOption, Dependency
import torch.nn.functional as F
from torch import nn
import os
from transformers import BertPreTrainedModel, BertModel, BertConfig
from capreolus import get_logger
from capreolus.utils.common import download_file, pack_tensor_2D
from capreolus.encoder import Encoder


class RepBERTTripletPooled_Class(BertPreTrainedModel):
    """
    Adapted from https://github.com/jingtaozhan/RepBERT-Index/
    """
    def __init__(self, config):
        super(RepBERTTripletPooled_Class, self).__init__(config)
        self.bert = BertModel(config)
        self.hidden_size = self.bert.config.hidden_size
        self.init_weights()

    def set_config(self, num_passages, pool_method):
        self.num_passages = num_passages
        self.pool_method = pool_method
        if self.pool_method == "conv":
            self.pool_layer = nn.Conv2d(1, self.hidden_size, (num_passages, self.hidden_size))
        elif self.pool_method == "linear":
            self.pool_layer = nn.Linear(self.num_passages * self.hidden_size, self.hidden_size)
        else:
            raise ValueError("Invalid pool method")

    def get_doc_embedding(self, doc, doc_mask):
        batch_size, num_passages, seq_len = doc.shape
        doc = doc.reshape(batch_size * num_passages, seq_len)
        doc_mask = doc_mask.reshape(batch_size * num_passages, seq_len)
        doc_lengths = torch.sum((doc_mask != 0), dim=1, keepdim=True)

        doc_output = self.bert(doc, attention_mask=doc_mask)
        doc_embedding = torch.sum(doc_output, dim=1) / doc_lengths
        doc_embedding = doc_embedding.reshape(batch_size, num_passages, self.hidden_size)
        if self.pool_method == "conv":
            doc_embedding = doc_embedding.reshape((batch_size, 1, num_passages, self.hidden_size))
        elif self.pool_method == "linear":
            doc_embedding = doc_embedding.reshape((batch_size, num_passages * self.hidden_size))
        else:
            raise ValueError("Unknown pool method")

        doc_embedding = self.pool_layer(doc_embedding)
        doc_embedding = doc_embedding.reshape((batch_size, self.hidden_size))

        return doc_embedding

    def forward(self, query, posdoc, negdoc, query_mask, posdoc_mask, negdoc_mask):
        """
        :param query: has shape (batch_size, seq_length)
        :param everything else: has shape (batch_size, num_passages, seq_len)

        :return:
        """

        batch_size, num_passages, seq_len = posdoc.shape

        # These lengths are required for averaging
        query_lengths = (query_mask != 0).sum(dim=1, keepdim=True)
        query_output = self.bert(query, attention_mask=query_mask)[0]
        query_embedding = torch.sum(query_output, dim=1) / query_lengths
        assert query_embedding.shape == (batch_size, self.hidden_size)

        posdoc_embedding = self.get_doc_embedding(posdoc, posdoc_mask)
        negdoc_embedding = self.get_doc_embedding(negdoc, negdoc_mask)

        posdoc_score = F.cosine_similarity(query_embedding, posdoc_embedding, dim=1)
        negdoc_score = F.cosine_similarity(query_embedding, negdoc_embedding, dim=1)

        return posdoc_score, negdoc_score

    def predict(self, input_ids, valid_mask, is_query=False):
        if is_query:
            query_lengths = (valid_mask != 0).sum(dim=1, keepdim=True)
            sequence_output = self.bert(input_ids, attention_mask=valid_mask)[0]
            sequence_output = torch.sum(sequence_output, dim=1) / query_lengths
        else:
            sequence_output = self.get_doc_embedding(input_ids, valid_mask)

        return sequence_output


@Encoder.register
class RepBERTTripletPooled(Encoder):
    module_name = "repberttripletpooled"
    dependencies = [
        Dependency(key="trainer", module="trainer", name="pytorchann"),
        Dependency(key="sampler", module="sampler", name="triplet"),
        Dependency(key="extractor", module="extractor", name="altpooledbertpassage"),
        Dependency(key="benchmark", module="benchmark")
    ]
    config_spec = [
        ConfigOption("pretrainedweights", "/GW/NeuralIR/nobackup/kevin_cache/msmarco_saved/repbert.ckpt-350000", "By default we use RepBERT MSMarco checkpoint"),
        ConfigOption("poolmethod", "conv")
    ]

    def instantiate_model(self):
        if not hasattr(self, "model"):
            config = BertConfig.from_pretrained(self.config["pretrainedweights"])
            config.pool_method = self.config["poolmethod"]
            self.model = torch.nn.DataParallel(RepBERTTripletPooled_Class.from_pretrained(self.config["pretrainedweights"], config=config))
            self.model.module.set_config(self.extractor.config["numpassages"], self.config["poolmethod"])
            self.hidden_size = self.model.module.hidden_size

    def encode_doc(self, numericalized_text, mask):
        return self.model.module.predict(numericalized_text, mask)

    def encode_query(self, numericalized_text, mask):
        return self.model.module.predict(numericalized_text, mask, is_query=True)

    def score(self, batch):
        return self.model(batch["query"], batch["posdoc"], batch["negdoc"], batch["query_mask"], batch["posdoc_mask"], batch["negdoc_mask"])

