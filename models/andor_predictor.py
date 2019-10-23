import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from net_utils import run_lstm, col_name_encode


class AndOrPredictor(nn.Module):
    def __init__(self, N_word, N_col, N_h, N_depth, dropout, gpu, use_hs):
        super(AndOrPredictor, self).__init__()
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

        self.q_att = nn.Linear(N_h, N_h)
        self.hs_att = nn.Linear(N_h, N_h)
        self.ao_out_q = nn.Linear(N_h, N_h)
        self.ao_out_hs = nn.Linear(N_h, N_h)
        self.ao_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 2)) #for and/or

        self.softmax = nn.Softmax() #dim=1
        self.CE = nn.CrossEntropyLoss()
        self.log_softmax = nn.LogSoftmax()
        self.mlsml = nn.MultiLabelSoftMarginLoss()
        self.bce_logit = nn.BCEWithLogitsLoss()
        self.sigm = nn.Sigmoid()
        if gpu:
            self.cuda()

    def forward(self, q_emb_var, q_len, hs_emb_var, hs_len):
        max_q_len = max(q_len)
        max_hs_len = max(hs_len)
        B = len(q_len)

        q_enc, _ = run_lstm(self.q_lstm, q_emb_var, q_len)
        hs_enc, _ = run_lstm(self.hs_lstm, hs_emb_var, hs_len)

        att_np_q = np.ones((B, max_q_len))
        att_val_q = torch.from_numpy(att_np_q).float()
        att_val_q = Variable(att_val_q.cuda())
        for idx, num in enumerate(q_len):
            if num < max_q_len:
                att_val_q[idx, num:] = -100
        att_prob_q = self.softmax(att_val_q)
        q_weighted = (q_enc * att_prob_q.unsqueeze(2)).sum(1)

        # Same as the above, compute SQL history embedding weighted by column attentions
        att_np_h = np.ones((B, max_hs_len))
        att_val_h = torch.from_numpy(att_np_h).float()
        att_val_h = Variable(att_val_h.cuda())
        for idx, num in enumerate(hs_len):
            if num < max_hs_len:
                att_val_h[idx, num:] = -100
        att_prob_h = self.softmax(att_val_h)
        hs_weighted = (hs_enc * att_prob_h.unsqueeze(2)).sum(1)
        # ao_score: (B, 2)
        ao_score = self.ao_out(self.ao_out_q(q_weighted) + int(self.use_hs)* self.ao_out_hs(hs_weighted))

        return ao_score


    def loss(self, score, truth):
        loss = 0
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
