import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from net_utils import run_lstm, col_name_encode


class MultiSqlPredictor(nn.Module):
    '''Predict if the next token is (multi SQL key words):
        NONE, EXCEPT, INTERSECT, or UNION.'''
    def __init__(self, N_word, N_col, N_h, N_depth, dropout, gpu, use_hs):
        super(MultiSqlPredictor, self).__init__()
        self.N_h = N_h
        self.gpu = gpu
        self.use_hs = use_hs

        self.q_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=dropout, bidirectional=True)

        if N_col:
            N_word = N_col

        self.hs_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=dropout, bidirectional=True)

        self.mkw_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=dropout, bidirectional=True)

        self.q_att = nn.Linear(N_h, N_h)
        self.hs_att = nn.Linear(N_h, N_h)
        self.multi_out_q = nn.Linear(N_h, N_h)
        self.multi_out_hs = nn.Linear(N_h, N_h)
        self.multi_out_c = nn.Linear(N_h, N_h)
        self.multi_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 1))

        self.softmax = nn.Softmax() #dim=1
        self.CE = nn.CrossEntropyLoss()
        self.log_softmax = nn.LogSoftmax()
        self.mlsml = nn.MultiLabelSoftMarginLoss()
        self.bce_logit = nn.BCEWithLogitsLoss()
        self.sigm = nn.Sigmoid()

        if gpu:
            self.cuda()

    def forward(self, q_emb_var, q_len, hs_emb_var, hs_len, mkw_emb_var, mkw_len):
        # print("q_emb_shape:{} hs_emb_shape:{}".format(q_emb_var.size(), hs_emb_var.size()))
        max_q_len = max(q_len)
        max_hs_len = max(hs_len)
        B = len(q_len)

        # q_enc: (B, max_q_len, hid_dim)
        # hs_enc: (B, max_hs_len, hid_dim)
        # mkw: (B, 4, hid_dim)
        q_enc, _ = run_lstm(self.q_lstm, q_emb_var, q_len)
        hs_enc, _ = run_lstm(self.hs_lstm, hs_emb_var, hs_len)
        mkw_enc, _ = run_lstm(self.mkw_lstm, mkw_emb_var, mkw_len)

        # Compute attention values between multi SQL key words and question tokens.
        # qmkw_att(q_enc).transpose(1, 2): (B, hid_dim, max_q_len)
        # att_val_qmkw: (B, 4, max_q_len)
        # print("mkw_enc {} q_enc {}".format(mkw_enc.size(), self.q_att(q_enc).transpose(1, 2).size()))
        att_val_qmkw = torch.bmm(mkw_enc, self.q_att(q_enc).transpose(1, 2))
        # assign appended positions values -100
        for idx, num in enumerate(q_len):
            if num < max_q_len:
                att_val_qmkw[idx, :, num:] = -100
        # att_prob_qmkw: (B, 4, max_q_len)
        att_prob_qmkw = self.softmax(att_val_qmkw.view((-1, max_q_len))).view(B, -1, max_q_len)
        # q_enc.unsqueeze(1): (B, 1, max_q_len, hid_dim)
        # att_prob_qmkw.unsqueeze(3): (B, 4, max_q_len, 1)
        # q_weighted: (B, 4, hid_dim)
        q_weighted = (q_enc.unsqueeze(1) * att_prob_qmkw.unsqueeze(3)).sum(2)

        # Same as the above, compute SQL history embedding weighted by key words attentions
        att_val_hsmkw = torch.bmm(mkw_enc, self.hs_att(hs_enc).transpose(1, 2))
        for idx, num in enumerate(hs_len):
            if num < max_hs_len:
                att_val_hsmkw[idx, :, num:] = -100
        att_prob_hsmkw = self.softmax(att_val_hsmkw.view((-1, max_hs_len))).view(B, -1, max_hs_len)
        hs_weighted = (hs_enc.unsqueeze(1) * att_prob_hsmkw.unsqueeze(3)).sum(2)

        # Compute prediction scores
        # self.multi_out.squeeze(): (B, 4, 1) -> (B, 4)
        mulit_score = self.multi_out(self.multi_out_q(q_weighted) + int(self.use_hs)* self.multi_out_hs(hs_weighted) + self.multi_out_c(mkw_enc)).view(B,-1)

        return mulit_score


    def loss(self, score, truth):
        data = torch.from_numpy(np.array(truth))
        truth_var = Variable(data.cuda())
        loss = self.CE(score, truth_var)

        return loss


    def check_acc(self, score, truth):
        err = 0
        B = len(score)
        pred = []
        for b in range(B):
            pred.append(np.argmax(score[b].data.cpu().numpy()))
        for b, (p, t) in enumerate(zip(pred, truth)):
            if p != t:
                err += 1

        return err
