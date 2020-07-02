import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

from capreolus import ConfigOption, Dependency
from capreolus.reranker import Reranker
from capreolus.reranker.common import create_emb_layer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GRUCell2d(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell2d, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.Wrz = nn.Linear(3 * hidden_size + input_size, 7 * hidden_size, bias=bias)
        self.W = nn.Linear(input_size, hidden_size, bias=bias)
        self.U = nn.Linear(3 * hidden_size, hidden_size, bias=bias)

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def softmax_by_row(self, zi, zl, zt, zd):
        zi, zl, zt, zd = zi.view(-1, 1, 2), zl.view(-1, 1, 2), zt.view(-1, 1, 2), zd.view(-1, 1, 2)
        # each: (B, 1, hidden=2)

        ppp = torch.cat((zi, zl, zt, zd), dim=1)  # (B, 4, hidden)

        pt = F.softmax(ppp, dim=1)  # (B, 4, hidden)

        zi, zl, zt, zd = pt.unbind(dim=1)
        zi, zl, zt, zd = zi.view(-1, 2), zl.view(-1, 2), zt.view(-1, 2), zd.view(-1, 2)

        return zi, zl, zt, zd

    def forward(self, x, hidden_i1_j1, hidden_i1_j, hidden_i_j1):
        q = torch.cat([hidden_i1_j, hidden_i_j1, hidden_i1_j1, x], dim=-1)  # (B, 3*nhidden+input)
        r_z = self.Wrz(q)
        rl, rt, rd, zi, zl, zt, zd = r_z.chunk(7, 1)  # each: (B, hidden)

        rl, rt, rd = torch.sigmoid(rl), torch.sigmoid(rt), torch.sigmoid(rd)
        zi, zl, zt, zd = self.softmax_by_row(zi, zl, zt, zd)

        r = torch.cat([rl, rt, rd], dim=-1)
        t11 = torch.cat([hidden_i1_j, hidden_i_j1, hidden_i1_j1], dim=-1)

        h1 = torch.tanh(self.W(x) + self.U(r * t11))
        h = (zl * hidden_i_j1) + (zt * hidden_i1_j) + (zd * hidden_i1_j1) + (zi * h1)

        return h


class GRUModel2d(nn.Module):
    def __init__(self, input_dim, hidden_dim, bias=True):
        super(GRUModel2d, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru_cell = GRUCell2d(input_dim, hidden_dim).to(device)

    def forward(self, x):
        B, T1, T2, H = x.size()
        last_outs = [(torch.zeros(x.size(0), self.hidden_dim).to(device)) for _ in range(T2 + 1)]
        for seq in range(T1):
            outs_row = [(torch.zeros(x.size(0), self.hidden_dim).to(device))]
            for seq1 in range(1, T2 + 1):
                hn = last_outs[seq1 - 1]
                hn_top = last_outs[seq1]
                hn_left = outs_row[seq1 - 1]

                hn1 = self.gru_cell(x[:, seq, seq1 - 1, :], hn, hn_top, hn_left)
                outs_row.append(hn1)

            last_outs = outs_row
        out = last_outs[-1]
        return out


class HiNT(nn.Module):
    def __init__(self, weights_matrix, p):
        super(HiNT, self).__init__()
        self.p = p
        self.passagelen = int(p["maxdoclen"] / 100)  # 100: windows size
        Ws_dim = 1  # fix to 1, since we assume 1 when creating GRUModel (2*Ws_dim + 1)

        embedding_dim = weights_matrix.shape[1]
        self.batch_size, self.lstm_hidden_dim = p["batch"], self.p["LSTMdim"]

        self.embedding = create_emb_layer(weights_matrix, non_trainable=True)

        self.Ws = nn.Linear(embedding_dim, Ws_dim)
        self.GRU2d1 = GRUModel2d(3, self.p["spatialGRU"]).to(device)
        self.GRU2d3 = GRUModel2d(3, self.p["spatialGRU"]).to(device)

        self.lstm = nn.LSTM(input_size=(4 * self.p["spatialGRU"]), hidden_size=self.lstm_hidden_dim, bidirectional=True)
        self.Wv = nn.Linear((4 * self.p["spatialGRU"]), self.lstm_hidden_dim, bias=True)
        self.fc = nn.Linear(self.lstm_hidden_dim * self.p["kmax"], 1)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return (
            Variable(torch.zeros(2, self.batch_size, self.lstm_hidden_dim).to(device)),
            Variable(torch.zeros(2, self.batch_size, self.lstm_hidden_dim).to(device)),
        )

    def matrix_inv(self, A):
        A1 = torch.randn(self.passagelen * self.batch_size, self.p["maxqlen"], 100, 3).type(torch.FloatTensor).to(device)
        for i in range(self.p["maxqlen"]):
            for j in range(100):
                A1[:, i, j, :] = A[:, self.p["maxqlen"] - i - 1, 99 - j, :]
        return A1

    def forward(self, sentence, query_sentence, M_XOR, M_cos, masks):
        """
            M_XOR or M_cos: (B, Q, D)
            masks: (B, Q, D) have 0 on non-pad positions and 1 on pad positions
        """
        sentence, query_sentence = sentence.to(device), query_sentence.to(device)

        x, query_x = self.embedding(sentence), self.embedding(query_sentence)

        X_i = self.Ws(query_x).view(self.batch_size, -1)
        Y_j = self.Ws(x).view(self.batch_size, -1)

        total_passage_level = torch.randn(self.passagelen, self.batch_size, 8).type(torch.FloatTensor).to(device)
        M_cos_passage = torch.randn(self.passagelen, self.batch_size, self.p["maxqlen"], 100).type(torch.FloatTensor).to(device)
        M_XOR_passage = torch.randn(self.passagelen, self.batch_size, self.p["maxqlen"], 100).type(torch.FloatTensor).to(device)
        Y_j_passage = torch.randn(self.passagelen, self.batch_size, 100).type(torch.FloatTensor).to(device)
        X_i_passage = torch.randn(self.passagelen, self.batch_size, self.p["maxqlen"]).type(torch.FloatTensor).to(device)

        mask_passage = torch.randn(self.passagelen, self.batch_size, self.p["maxqlen"], 100).type(torch.FloatTensor).to(device)

        for number_window in range(self.passagelen):
            mask_passage[number_window] = masks[:, :, (number_window * 100) : ((number_window + 1) * 100)]  # (P, BAT, Q, 100)

            M_cos_passage[number_window] = M_cos[:, :, (number_window * 100) : ((number_window + 1) * 100)]  # (P, BAT, Q, 100)
            M_XOR_passage[number_window] = M_XOR[:, :, (number_window * 100) : ((number_window + 1) * 100)]  # (P, BAT, Q, 100)
            Y_j_passage[number_window] = Y_j[:, (number_window * 100) : ((number_window + 1) * 100)]  # (P, BAT, 100)
            X_i_passage[number_window] = X_i  # (P, BAT, 100)

        S_cos = (
            torch.randn(self.passagelen, self.batch_size, self.p["maxqlen"], 100, 3).type(torch.FloatTensor).to(device)
        )  # (P, BAT, Q, 100, 3)
        S_xor = torch.randn(self.passagelen, self.batch_size, self.p["maxqlen"], 100, 3).type(torch.FloatTensor).to(device)
        S_cos[:, :, :, :, 0] = X_i_passage.reshape(self.passagelen, self.batch_size, self.p["maxqlen"], 1).expand(
            self.passagelen, self.batch_size, self.p["maxqlen"], 100
        )
        S_cos[:, :, :, :, 1] = Y_j_passage.reshape(self.passagelen, self.batch_size, 1, 100).expand(
            self.passagelen, self.batch_size, self.p["maxqlen"], 100
        )
        S_cos[:, :, :, :, 2] = M_cos_passage
        S_xor[:, :, :, :, 0] = X_i_passage.reshape(self.passagelen, self.batch_size, self.p["maxqlen"], 1).expand(
            self.passagelen, self.batch_size, self.p["maxqlen"], 100
        )
        S_xor[:, :, :, :, 1] = Y_j_passage.reshape(self.passagelen, self.batch_size, 1, 100).expand(
            self.passagelen, self.batch_size, self.p["maxqlen"], 100
        )
        S_xor[:, :, :, :, 2] = M_XOR_passage

        # add mask on X and Y
        S_cos = S_cos * (1 - mask_passage).unsqueeze(-1)
        S_xor = S_xor * (1 - mask_passage).unsqueeze(-1)

        # S_xor, S_cos: (P, B, Q, 100, 3) -> (P*B, Q, 100, 3)
        S_xor1 = S_xor.view(self.passagelen * self.batch_size, self.p["maxqlen"], 100, 3)
        S_cos1 = S_cos.view(self.passagelen * self.batch_size, self.p["maxqlen"], 100, 3)
        S_xor1_cos1 = torch.cat([S_xor1, S_cos1], dim=0)  # (2*P*B, Q, 100, 3)
        H_xor_cos = self.GRU2d1(S_xor1_cos1)  # (2*P*B, 2)

        H_xor = H_xor_cos[: (self.passagelen * self.batch_size)]  # (P*B, 2)
        H_cos = H_xor_cos[(self.passagelen * self.batch_size) :]  # (P*B, 2)
        e = torch.cat([H_xor, H_cos], dim=-1)  # (P*B, 4)

        S_xor_inv = self.matrix_inv(S_xor1)  # (P*B, Q, 100, 3)
        S_cos_inv = self.matrix_inv(S_cos1)
        S_xor_cos_inv = torch.cat([S_xor_inv, S_cos_inv], dim=0)
        H_xor_cos_inv = self.GRU2d3(S_xor_cos_inv)
        H_xor_inv = H_xor_cos_inv[: (self.passagelen * self.batch_size)]
        H_cos_inv = H_xor_cos_inv[(self.passagelen * self.batch_size) :]

        e_inv = torch.cat([H_xor_inv, H_cos_inv], dim=-1)  # (P*B, 4)
        passage_level_e = torch.cat([e, e_inv], dim=-1)  # (P*B, 8)
        for number_window in range(self.passagelen):  # (P, B, 8)
            total_passage_level[number_window] = passage_level_e[
                (self.batch_size * number_window) : (self.batch_size * (number_window + 1))
            ]

        lstm_out, self.hidden = self.lstm(total_passage_level, self.hidden)
        # lstm_out: (P, B, 2 * self.hidden), where P is the timestep dimension

        lstm_out_forward = lstm_out[:, :, :6]  # (P, B, self.hidden)
        lstm_out_backward = lstm_out[:, :, 6:]  # (P, B, self.hidden)

        # added
        lstm_out = lstm_out_forward + lstm_out_backward

        vt = torch.tanh(self.Wv(total_passage_level))  # (P, B, 8) -> (P, B, 6)

        # evidence = torch.cat((vt, lstm_out_forward, lstm_out_backward), 0)    # (3P, B, 6)
        evidence = torch.cat((vt, lstm_out), 0)  # (2P, B, 6)

        evidence1 = torch.transpose(evidence, 0, 2)  # (6, B, 3P) / (6, B, 2P)
        anss, _ = torch.topk(evidence1, self.p["kmax"], largest=True, sorted=True, dim=2)  # (6, B, 10)
        ans = torch.transpose(anss, 0, 1)  # (B, 6, 10)
        ans1 = ans.contiguous().view(self.batch_size, -1)  # (B, 60)
        score = self.fc(ans1)
        return score.view(-1)


class HiNT_main(nn.Module):
    def __init__(self, extractor, config):
        super(HiNT_main, self).__init__()
        self.HiNT1 = HiNT(extractor.embeddings, config).to(device)
        self.batch_size = config["batch"]

    def init_hidden(self):
        return self.HiNT1.init_hidden()

    def forward(self, query_sentence, query_idf, pos_sentence, neg_sentence):
        self.HiNT1.hidden = self.HiNT1.init_hidden()
        query_sentence = query_sentence
        query_idf = query_idf
        pos_sentence = pos_sentence
        neg_sentence = neg_sentence

        x = self.HiNT1.embedding(pos_sentence)
        query_x = self.HiNT1.embedding(query_sentence)
        BAT = query_sentence.shape[0]
        A = query_sentence.shape[1]
        B = pos_sentence.shape[1]

        x1 = x.norm(dim=2)[:, :, None] + 1e-7
        query_x1 = query_x.norm(dim=2)[:, :, None] + 1e-7
        x_norm = x / x1  # (BAT, B, H)
        query_x_norm = query_x / query_x1  # (BAT, A, H)

        M_cos_pos = torch.matmul(query_x_norm, torch.transpose(x_norm, 1, 2))  # (BAT, A, B)

        nul = torch.zeros_like(M_cos_pos)
        one = torch.ones_like(M_cos_pos)
        XOR_matrix_pos = torch.where(
            query_sentence.reshape(BAT, A, 1).expand(BAT, A, B) == pos_sentence.reshape(BAT, 1, B).expand(BAT, A, B), one, nul
        )

        # add padding for both matrix
        query_masks, sentence_masks = (query_sentence == 0), (pos_sentence == 0)  # (B, Q), (B, D)
        pos_masks = query_masks[:, :, None] * sentence_masks[:, None, :]  # (B, Q, D)
        # add mask
        # XOR_matrix_pos = torch.where(pos_masks, nul, XOR_matrix_pos)
        # M_cos_pos = torch.where(pos_masks, nul, M_cos_pos)
        # mask would be applied in HiNT1
        pos_scores = self.HiNT1(pos_sentence, query_sentence, XOR_matrix_pos, M_cos_pos, pos_masks)

        self.HiNT1.hidden = self.HiNT1.init_hidden()
        x = self.HiNT1.embedding(neg_sentence)
        query_x = self.HiNT1.embedding(query_sentence)
        BAT = query_sentence.shape[0]
        A = query_sentence.shape[1]
        B = neg_sentence.shape[1]

        x1 = x.norm(dim=2)[:, :, None] + 1e-7
        query_x1 = query_x.norm(dim=2)[:, :, None] + 1e-7
        x_norm = x / x1
        query_x_norm = query_x / query_x1
        M_cos_neg = torch.matmul(query_x_norm, torch.transpose(x_norm, 1, 2))
        nul = torch.zeros_like(M_cos_pos)
        one = torch.ones_like(M_cos_pos)
        XOR_matrix_neg = torch.where(
            query_sentence.reshape(BAT, A, 1).expand(BAT, A, B) == neg_sentence.reshape(BAT, 1, B).expand(BAT, A, B), one, nul
        )

        # add mask for both matrix
        query_masks, sentence_masks = (query_sentence == 0), (neg_sentence == 0)  # (B, Q), (B, D)
        neg_masks = query_masks[:, :, None] * sentence_masks[:, None, :]  # (B, Q, D)
        # add mask
        # XOR_matrix_neg = torch.where(neg_masks, nul, XOR_matrix_neg)
        # M_cos_neg = torch.where(neg_masks, nul, M_cos_neg)

        # mask would be applied in HiNT1
        neg_scores = self.HiNT1(neg_sentence, query_sentence, XOR_matrix_neg, M_cos_neg, neg_masks)

        return [pos_scores, neg_scores]

    def test_forward(self, query_sentence, query_idf, pos_sentence):
        self.HiNT1.hidden = self.HiNT1.init_hidden()
        query_sentence = query_sentence
        query_idf = query_idf
        pos_sentence = pos_sentence

        x = self.HiNT1.embedding(pos_sentence)
        query_x = self.HiNT1.embedding(query_sentence)
        BAT = query_sentence.shape[0]
        A = query_sentence.shape[1]
        B = pos_sentence.shape[1]
        x1 = x.norm(dim=2)[:, :, None] + 1e-7
        query_x1 = query_x.norm(dim=2)[:, :, None] + 1e-7
        x_norm = x / x1
        query_x_norm = query_x / query_x1
        M_cos_pos = torch.matmul(query_x_norm, torch.transpose(x_norm, 1, 2))

        nul = torch.zeros_like(M_cos_pos)
        one = torch.ones_like(M_cos_pos)
        XOR_matrix_pos = torch.where(
            query_sentence.reshape(BAT, A, 1).expand(BAT, A, B) == pos_sentence.reshape(BAT, 1, B).expand(BAT, A, B), one, nul
        )

        query_masks, sentence_masks = (query_sentence == 0), (pos_sentence == 0)  # (B, Q), (B, D)
        pos_masks = query_masks[:, :, None] * sentence_masks[:, None, :]  # (B, Q, D)
        # add mask
        # XOR_matrix_pos = torch.where(pos_masks, nul, XOR_matrix_pos)
        # M_cos_pos = torch.where(pos_masks, nul, M_cos_pos)

        # mask would be applied in HiNT1
        pos_scores = self.HiNT1(pos_sentence, query_sentence, XOR_matrix_pos, M_cos_pos, pos_masks)
        return pos_scores


@Reranker.register
class HINT(Reranker):
    """Yixing Fan, Jiafeng Guo, Yanyan Lan, Jun Xu, Chengxiang Zhai, and Xueqi Cheng. 2018. Modeling Diverse Relevance Patterns in Ad-hoc Retrieval. In SIGIR'18."""

    module_name = "HINT"

    dependencies = [
        Dependency(key="extractor", module="extractor", name="slowembedtext"),
        Dependency(key="trainer", module="trainer", name="pytorch"),
    ]
    config_spec = [ConfigOption("spatialGRU", 2), ConfigOption("LSTMdim", 6), ConfigOption("kmax", 10)]

    def test(self, query_sentence, query_idf, pos_sentence, *args, **kwargs):
        return self.model.test_forward(query_sentence, query_idf, pos_sentence)

    def score(self, d):
        query_idf = d["query_idf"]
        query_sentence = d["query"]
        pos_sentence, neg_sentence = d["posdoc"], d["negdoc"]
        return self.model(query_sentence, query_idf, pos_sentence, neg_sentence)

    def test(self, d):
        query_idf = d["query_idf"]
        query_sentence = d["query"]
        pos_sentence = d["posdoc"]

        return self.model.test_forward(query_sentence, query_idf, pos_sentence)

    def zero_grad(self, *args, **kwargs):
        self.model.zero_grad(*args, **kwargs)

    def build_model(self):
        if not hasattr(self, "model"):
            config = dict(self.config)
            config.update(self.extractor.config)
            config["batch"] = self.trainer.config["batch"]
            self.model = HiNT_main(self.extractor, config)

        return self.model
