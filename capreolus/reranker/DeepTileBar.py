import copy

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

from capreolus import ConfigOption, Dependency
from capreolus.reranker import Reranker
from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)  # pylint: disable=invalid-name


class DeepTileBar_nn(nn.Module):
    def __init__(self, p, batch_size, number_filter, lstm_hidden_dim, linear_hidden_dim1, linear_hidden_dim2):
        super(DeepTileBar_nn, self).__init__()
        self.p = p
        self.tilechannels = 3
        if not self.p["tfchannel"]:
            self.tilechannels -= 1
        self.batch_size = batch_size
        self.number_filter = number_filter
        self.lstm_hidden_dim = lstm_hidden_dim
        self.conv1 = nn.Conv2d(self.tilechannels, number_filter, (p["maxqlen"], 1), stride=1)
        self.conv2 = nn.Conv2d(self.tilechannels, number_filter, (p["maxqlen"], 2), stride=1)
        self.conv3 = nn.Conv2d(self.tilechannels, number_filter, (p["maxqlen"], 3), stride=1)
        self.conv4 = nn.Conv2d(self.tilechannels, number_filter, (p["maxqlen"], 4), stride=1)
        self.conv5 = nn.Conv2d(self.tilechannels, number_filter, (p["maxqlen"], 5), stride=1)
        self.conv6 = nn.Conv2d(self.tilechannels, number_filter, (p["maxqlen"], 6), stride=1)
        self.conv7 = nn.Conv2d(self.tilechannels, number_filter, (p["maxqlen"], 7), stride=1)
        self.conv8 = nn.Conv2d(self.tilechannels, number_filter, (p["maxqlen"], 8), stride=1)
        self.conv9 = nn.Conv2d(self.tilechannels, number_filter, (p["maxqlen"], 9), stride=1)
        self.conv10 = nn.Conv2d(self.tilechannels, number_filter, (p["maxqlen"], 10), stride=1)
        self.lstm1 = nn.LSTM(input_size=3, hidden_size=lstm_hidden_dim)
        self.lstm2 = nn.LSTM(input_size=3, hidden_size=lstm_hidden_dim)
        self.lstm3 = nn.LSTM(input_size=3, hidden_size=lstm_hidden_dim)
        self.lstm4 = nn.LSTM(input_size=3, hidden_size=lstm_hidden_dim)
        self.lstm5 = nn.LSTM(input_size=3, hidden_size=lstm_hidden_dim)
        self.lstm6 = nn.LSTM(input_size=3, hidden_size=lstm_hidden_dim)
        self.lstm7 = nn.LSTM(input_size=3, hidden_size=lstm_hidden_dim)
        self.lstm8 = nn.LSTM(input_size=3, hidden_size=lstm_hidden_dim)
        self.lstm9 = nn.LSTM(input_size=3, hidden_size=lstm_hidden_dim)
        self.lstm10 = nn.LSTM(input_size=3, hidden_size=lstm_hidden_dim)
        self.W1 = nn.Linear(10 * lstm_hidden_dim, linear_hidden_dim1, bias=True)
        self.W2 = nn.Linear(linear_hidden_dim1, linear_hidden_dim2, bias=True)
        self.W3 = nn.Linear(linear_hidden_dim2, 1, bias=True)
        [
            self.hidden1,
            self.hidden2,
            self.hidden3,
            self.hidden4,
            self.hidden5,
            self.hidden6,
            self.hidden7,
            self.hidden8,
            self.hidden9,
            self.hidden10,
        ] = self.init_hidden()

    def init_hidden(self):
        # first is the hidden h
        # second is the cell c
        # if self.use_gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        l = []
        for j in range(10):
            l.append(
                (
                    Variable(torch.zeros(1, self.batch_size, self.lstm_hidden_dim).to(device)),
                    Variable(torch.zeros(1, self.batch_size, self.lstm_hidden_dim).to(device)),
                )
            )
        return l

    def reset_hidden(self):
        [
            self.hidden1,
            self.hidden2,
            self.hidden3,
            self.hidden4,
            self.hidden5,
            self.hidden6,
            self.hidden7,
            self.hidden8,
            self.hidden9,
            self.hidden10,
        ] = self.init_hidden()

    def forward(self, tile_matrix1):
        tile_matrix2 = torch.transpose(
            torch.transpose(tile_matrix1.view(self.batch_size, self.p["maxqlen"], self.p["passagelen"], -1), 1, 3), 2, 3
        )
        x1 = torch.transpose(torch.transpose(self.conv1(tile_matrix2).view(self.batch_size, self.number_filter, -1), 0, 2), 1, 2)
        x2 = torch.transpose(torch.transpose(self.conv2(tile_matrix2).view(self.batch_size, self.number_filter, -1), 0, 2), 1, 2)
        x3 = torch.transpose(torch.transpose(self.conv3(tile_matrix2).view(self.batch_size, self.number_filter, -1), 0, 2), 1, 2)
        x4 = torch.transpose(torch.transpose(self.conv4(tile_matrix2).view(self.batch_size, self.number_filter, -1), 0, 2), 1, 2)
        x5 = torch.transpose(torch.transpose(self.conv5(tile_matrix2).view(self.batch_size, self.number_filter, -1), 0, 2), 1, 2)
        x6 = torch.transpose(torch.transpose(self.conv6(tile_matrix2).view(self.batch_size, self.number_filter, -1), 0, 2), 1, 2)
        x7 = torch.transpose(torch.transpose(self.conv7(tile_matrix2).view(self.batch_size, self.number_filter, -1), 0, 2), 1, 2)
        x8 = torch.transpose(torch.transpose(self.conv8(tile_matrix2).view(self.batch_size, self.number_filter, -1), 0, 2), 1, 2)
        x9 = torch.transpose(torch.transpose(self.conv9(tile_matrix2).view(self.batch_size, self.number_filter, -1), 0, 2), 1, 2)
        x10 = torch.transpose(
            torch.transpose(self.conv10(tile_matrix2).view(self.batch_size, self.number_filter, -1), 0, 2), 1, 2
        )

        lstm_out1, self.hidden1 = self.lstm1(x1, self.hidden1)
        lstm_out2, self.hidden2 = self.lstm2(x2, self.hidden2)
        lstm_out3, self.hidden3 = self.lstm3(x3, self.hidden3)
        lstm_out4, self.hidden4 = self.lstm4(x4, self.hidden4)
        lstm_out5, self.hidden5 = self.lstm5(x5, self.hidden5)
        lstm_out6, self.hidden6 = self.lstm6(x6, self.hidden6)
        lstm_out7, self.hidden7 = self.lstm7(x7, self.hidden7)
        lstm_out8, self.hidden8 = self.lstm8(x8, self.hidden8)
        lstm_out9, self.hidden9 = self.lstm9(x9, self.hidden9)
        lstm_out10, self.hidden10 = self.lstm10(x10, self.hidden10)
        input_x = torch.cat(
            [
                lstm_out1[-1],
                lstm_out2[-1],
                lstm_out3[-1],
                lstm_out4[-1],
                lstm_out5[-1],
                lstm_out6[-1],
                lstm_out7[-1],
                lstm_out8[-1],
                lstm_out9[-1],
                lstm_out10[-1],
            ],
            1,
        )
        input_x1 = F.relu(self.W1(input_x))
        input_x2 = F.relu(self.W2(input_x1))
        input_x3 = self.W3(input_x2)
        return input_x3.view(-1)


class DeepTileBar_class(nn.Module):
    def __init__(self, extractor, config):
        super(DeepTileBar_class, self).__init__()
        batch_size = config["batch"]
        number_filter = config["numberfilter"]
        lstm_hidden_dim = config["lstmhiddendim"]
        linear_hidden_dim1 = config["linearhiddendim1"]
        linear_hidden_dim2 = config["linearhiddendim2"]
        config = dict(config)
        config.update(dict(extractor.config))

        self.DeepTileBar1 = DeepTileBar_nn(
            config, batch_size, number_filter, lstm_hidden_dim, linear_hidden_dim1, linear_hidden_dim2
        )

    def init_hidden(self):
        return self.DeepTileBar1.init_hidden()

    def reset_hidden(self):
        self.DeepTileBar1.reset_hidden()

    def forward(self, pos_tile_matrix, neg_tile_matrix):
        self.reset_hidden()
        pos_tag_scores = self.DeepTileBar1(pos_tile_matrix)
        self.reset_hidden()
        neg_tag_scores = self.DeepTileBar1(neg_tile_matrix)
        return [pos_tag_scores, neg_tag_scores]

    def test_forward(self, pos_tile_matrix):
        self.reset_hidden()
        pos_tag_scores = self.DeepTileBar1(pos_tile_matrix)
        return pos_tag_scores


@Reranker.register
class DeepTileBar(Reranker):
    """Zhiwen Tang and Grace Hui Yang. 2019. DeepTileBars: Visualizing Term Distribution for Neural Information Retrieval. In AAAI'19."""

    module_name = "DeepTileBar"

    dependencies = [
        Dependency(key="extractor", module="extractor", name="deeptiles"),
        Dependency(key="trainer", module="trainer", name="pytorch"),
    ]

    config_spec = [
        ConfigOption("passagelen", 30),
        ConfigOption("numberfilter", 3),
        ConfigOption("lstmhiddendim", 3),
        ConfigOption("linearhiddendim1", 32),
        ConfigOption("linearhiddendim2", 16),
    ]

    def build_model(self):
        if not hasattr(self, "model"):
            config = copy.copy(dict(self.config))
            config["batch"] = self.trainer.config["batch"]
            self.model = DeepTileBar_class(self.extractor, config)

        return self.model

    def score(self, d):
        pos_tile_matrix = torch.cat([d["posdoc"][i] for i in range(len(d["qid"]))])  # 32 x
        neg_tile_matrix = torch.cat([d["negdoc"][i] for i in range(len(d["qid"]))])
        return self.model(pos_tile_matrix, neg_tile_matrix)

    def test(self, d):
        qids = d["qid"]
        pos_sentence = d["posdoc"]
        pos_tile_matrix = torch.cat([pos_sentence[i] for i in range(len(qids))])

        return self.model.test_forward(pos_tile_matrix)

    def zero_grad(self, *args, **kwargs):
        self.model.zero_grad(*args, **kwargs)
