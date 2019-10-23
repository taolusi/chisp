import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from net_utils import run_lstm, col_name_encode


class KeyWordPredictor(nn.Module):
    '''Predict if the next token is (SQL key words):
        WHERE, GROUP BY, ORDER BY. excluding SELECT (it is a must)'''
    def __init__(self, N_word, N_col, N_h, N_depth, dropout, gpu, use_hs):
        super(KeyWordPredictor, self).__init__()
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

        self.kw_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=dropout, bidirectional=True)

        self.q_num_att = nn.Linear(N_h, N_h)
        self.hs_num_att = nn.Linear(N_h, N_h)
        self.kw_num_out_q = nn.Linear(N_h, N_h)
        self.kw_num_out_hs = nn.Linear(N_h, N_h)
        self.kw_num_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 4)) # num of key words: 0-3

        self.q_att = nn.Linear(N_h, N_h)
        self.hs_att = nn.Linear(N_h, N_h)
        self.kw_out_q = nn.Linear(N_h, N_h)
        self.kw_out_hs = nn.Linear(N_h, N_h)
        self.kw_out_kw = nn.Linear(N_h, N_h)
        self.kw_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 1))

        self.softmax = nn.Softmax() #dim=1
        self.CE = nn.CrossEntropyLoss()
        self.log_softmax = nn.LogSoftmax()
        self.mlsml = nn.MultiLabelSoftMarginLoss()
        self.bce_logit = nn.BCEWithLogitsLoss()
        self.sigm = nn.Sigmoid()
        if gpu:
            self.cuda()

    def forward(self, q_emb_var, q_len, hs_emb_var, hs_len, kw_emb_var, kw_len):
        max_q_len = max(q_len)
        max_hs_len = max(hs_len)
        B = len(q_len)

        q_enc, _ = run_lstm(self.q_lstm, q_emb_var, q_len)
        hs_enc, _ = run_lstm(self.hs_lstm, hs_emb_var, hs_len)
        kw_enc, _ = run_lstm(self.kw_lstm, kw_emb_var, kw_len)

        # Predict key words number: 0-3
        att_val_qkw_num = torch.bmm(kw_enc, self.q_num_att(q_enc).transpose(1, 2))
        for idx, num in enumerate(q_len):
            if num < max_q_len:
                att_val_qkw_num[idx, :, num:] = -100
        att_prob_qkw_num = self.softmax(att_val_qkw_num.view((-1, max_q_len))).view(B, -1, max_q_len)
        # q_weighted: (B, hid_dim)
        q_weighted_num = (q_enc.unsqueeze(1) * att_prob_qkw_num.unsqueeze(3)).sum(2).sum(1)

        # Same as the above, compute SQL history embedding weighted by key words attentions
        att_val_hskw_num = torch.bmm(kw_enc, self.hs_num_att(hs_enc).transpose(1, 2))
        for idx, num in enumerate(hs_len):
            if num < max_hs_len:
                att_val_hskw_num[idx, :, num:] = -100
        att_prob_hskw_num = self.softmax(att_val_hskw_num.view((-1, max_hs_len))).view(B, -1, max_hs_len)
        hs_weighted_num = (hs_enc.unsqueeze(1) * att_prob_hskw_num.unsqueeze(3)).sum(2).sum(1)
        # Compute prediction scores
        # self.kw_num_out: (B, 4)
        kw_num_score = self.kw_num_out(self.kw_num_out_q(q_weighted_num) + int(self.use_hs)* self.kw_num_out_hs(hs_weighted_num))

        # Predict key words: WHERE, GROUP BY, ORDER BY.
        att_val_qkw = torch.bmm(kw_enc, self.q_att(q_enc).transpose(1, 2))
        for idx, num in enumerate(q_len):
            if num < max_q_len:
                att_val_qkw[idx, :, num:] = -100
        att_prob_qkw = self.softmax(att_val_qkw.view((-1, max_q_len))).view(B, -1, max_q_len)
        # q_weighted: (B, 3, hid_dim)
        q_weighted = (q_enc.unsqueeze(1) * att_prob_qkw.unsqueeze(3)).sum(2)

        # Same as the above, compute SQL history embedding weighted by key words attentions
        att_val_hskw = torch.bmm(kw_enc, self.hs_att(hs_enc).transpose(1, 2))
        for idx, num in enumerate(hs_len):
            if num < max_hs_len:
                att_val_hskw[idx, :, num:] = -100
        att_prob_hskw = self.softmax(att_val_hskw.view((-1, max_hs_len))).view(B, -1, max_hs_len)
        hs_weighted = (hs_enc.unsqueeze(1) * att_prob_hskw.unsqueeze(3)).sum(2)
        # Compute prediction scores
        # self.kw_out.squeeze(): (B, 3)
        kw_score = self.kw_out(self.kw_out_q(q_weighted) + int(self.use_hs)* self.kw_out_hs(hs_weighted) + self.kw_out_kw(kw_enc)).view(B,-1)

        score = (kw_num_score, kw_score)

        return score

    def loss(self, score, truth):
        loss = 0
        B = len(truth)
        kw_num_score, kw_score = score
        #loss for the key word number
        truth_num = [len(t) for t in truth] # double check to exclude select
        data = torch.from_numpy(np.array(truth_num))
        truth_num_var = Variable(data.cuda())
        loss += self.CE(kw_num_score, truth_num_var)
        #loss for the key words
        T = len(kw_score[0])
        truth_prob = np.zeros((B, T), dtype=np.float32)
        for b in range(B):
            truth_prob[b][truth[b]] = 1
        data = torch.from_numpy(truth_prob)
        truth_var = Variable(data.cuda())
        #loss += self.mlsml(kw_score, truth_var)
        #loss += self.bce_logit(kw_score, truth_var) # double check no sigmoid for kw
        pred_prob = self.sigm(kw_score)
        bce_loss = -torch.mean( 3*(truth_var * \
                torch.log(pred_prob+1e-10)) + \
                (1-truth_var) * torch.log(1-pred_prob+1e-10) )
        loss += bce_loss

        return loss


    def check_acc(self, score, truth):
        num_err, err, tot_err = 0, 0, 0
        B = len(truth)
        pred = []
        kw_num_score, kw_score = [x.data.cpu().numpy() for x in score]
        for b in range(B):
            cur_pred = {}
            kw_num = np.argmax(kw_num_score[b])
            cur_pred['kw_num'] = kw_num
            cur_pred['kw'] = np.argsort(-kw_score[b])[:kw_num]
            pred.append(cur_pred)

        for b, (p, t) in enumerate(zip(pred, truth)):
            kw_num, kw = p['kw_num'], p['kw']
            flag = True
            if kw_num != len(t): # double check to excluding select
                num_err += 1
                flag = False
            if flag and set(kw) != set(t):
                err += 1
                flag = False
            if not flag:
                tot_err += 1

        return np.array((num_err, err, tot_err))
