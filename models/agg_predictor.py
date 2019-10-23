import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from net_utils import run_lstm, col_name_encode


class AggPredictor(nn.Module):
    def __init__(self, N_word, N_col, N_h, N_depth, dropout, gpu, use_hs):
        super(AggPredictor, self).__init__()
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

        self.col_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=dropout, bidirectional=True)

        self.q_num_att = nn.Linear(N_h, N_h)
        self.hs_num_att = nn.Linear(N_h, N_h)
        self.agg_num_out_q = nn.Linear(N_h, N_h)
        self.agg_num_out_hs = nn.Linear(N_h, N_h)
        self.agg_num_out_c = nn.Linear(N_h, N_h)
        self.agg_num_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 4)) #for 0-3 agg num

        self.q_att = nn.Linear(N_h, N_h)
        self.hs_att = nn.Linear(N_h, N_h)
        self.agg_out_q = nn.Linear(N_h, N_h)
        self.agg_out_hs = nn.Linear(N_h, N_h)
        self.agg_out_c = nn.Linear(N_h, N_h)
        self.agg_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 5)) #for 1-5 aggregators

        self.softmax = nn.Softmax() #dim=1
        self.CE = nn.CrossEntropyLoss()
        self.log_softmax = nn.LogSoftmax()
        self.mlsml = nn.MultiLabelSoftMarginLoss()
        self.bce_logit = nn.BCEWithLogitsLoss()
        self.sigm = nn.Sigmoid()
        if gpu:
            self.cuda()


    def forward(self, q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, col_name_len, gt_col):
        max_q_len = max(q_len)
        max_hs_len = max(hs_len)
        max_col_len = max(col_len)
        B = len(q_len)

        q_enc, _ = run_lstm(self.q_lstm, q_emb_var, q_len)
        hs_enc, _ = run_lstm(self.hs_lstm, hs_emb_var, hs_len)
        col_enc, _ = col_name_encode(col_emb_var, col_name_len, col_len, self.col_lstm)

        col_emb = []
        for b in range(B):
            col_emb.append(col_enc[b, gt_col[b]])
        col_emb = torch.stack(col_emb)

        # Predict agg number
        att_val_qc_num = torch.bmm(col_emb.unsqueeze(1), self.q_num_att(q_enc).transpose(1, 2)).view(B, -1)
        for idx, num in enumerate(q_len):
            if num < max_q_len:
                att_val_qc_num[idx, num:] = -100
        att_prob_qc_num = self.softmax(att_val_qc_num)
        q_weighted_num = (q_enc * att_prob_qc_num.unsqueeze(2)).sum(1)

        # Same as the above, compute SQL history embedding weighted by column attentions
        att_val_hc_num = torch.bmm(col_emb.unsqueeze(1), self.hs_num_att(hs_enc).transpose(1, 2)).view(B, -1)
        for idx, num in enumerate(hs_len):
            if num < max_hs_len:
                att_val_hc_num[idx, num:] = -100
        att_prob_hc_num = self.softmax(att_val_hc_num)
        hs_weighted_num = (hs_enc * att_prob_hc_num.unsqueeze(2)).sum(1)
        # agg_num_score: (B, 4)
        agg_num_score = self.agg_num_out(self.agg_num_out_q(q_weighted_num) + int(self.use_hs)* self.agg_num_out_hs(hs_weighted_num) + self.agg_num_out_c(col_emb))

        # Predict aggregators
        att_val_qc = torch.bmm(col_emb.unsqueeze(1), self.q_att(q_enc).transpose(1, 2)).view(B, -1)
        for idx, num in enumerate(q_len):
            if num < max_q_len:
                att_val_qc[idx, num:] = -100
        att_prob_qc = self.softmax(att_val_qc)
        q_weighted = (q_enc * att_prob_qc.unsqueeze(2)).sum(1)

        # Same as the above, compute SQL history embedding weighted by column attentions
        att_val_hc = torch.bmm(col_emb.unsqueeze(1), self.hs_att(hs_enc).transpose(1, 2)).view(B, -1)
        for idx, num in enumerate(hs_len):
            if num < max_hs_len:
                att_val_hc[idx, num:] = -100
        att_prob_hc = self.softmax(att_val_hc)
        hs_weighted = (hs_enc * att_prob_hc.unsqueeze(2)).sum(1)
        # agg_score: (B, 5)
        agg_score = self.agg_out(self.agg_out_q(q_weighted) + int(self.use_hs)* self.agg_out_hs(hs_weighted) + self.agg_out_c(col_emb))

        score = (agg_num_score, agg_score)

        return score


    def loss(self, score, truth):
        loss = 0
        B = len(truth)
        agg_num_score, agg_score = score
        #loss for the column number
        truth_num = [len(t) for t in truth] # double check truth format and for test cases
        data = torch.from_numpy(np.array(truth_num))
        truth_num_var = Variable(data.cuda())
        loss += self.CE(agg_num_score, truth_num_var)
        #loss for the key words
        T = len(agg_score[0])
        truth_prob = np.zeros((B, T), dtype=np.float32)
        for b in range(B):
            truth_prob[b][truth[b]] = 1
        data = torch.from_numpy(truth_prob)
        truth_var = Variable(data.cuda())
        #loss += self.mlsml(agg_score, truth_var)
        #loss += self.bce_logit(agg_score, truth_var) # double check no sigmoid
        pred_prob = self.sigm(agg_score)
        bce_loss = -torch.mean( 3*(truth_var * \
                torch.log(pred_prob+1e-10)) + \
                (1-truth_var) * torch.log(1-pred_prob+1e-10) )
        loss += bce_loss

        return loss


    def check_acc(self, score, truth):
        num_err, err, tot_err = 0, 0, 0
        B = len(truth)
        pred = []
        agg_num_score, agg_score = [x.data.cpu().numpy() for x in score]
        for b in range(B):
            cur_pred = {}
            agg_num = np.argmax(agg_num_score[b]) #double check
            cur_pred['agg_num'] = agg_num
            cur_pred['agg'] = np.argsort(-agg_score[b])[:agg_num]
            pred.append(cur_pred)

        for b, (p, t) in enumerate(zip(pred, truth)):
            agg_num, agg = p['agg_num'], p['agg']
            flag = True
            if agg_num != len(t): # double check truth format and for test cases
                num_err += 1
                flag = False
            if flag and set(agg) != set(t):
                err += 1
                flag = False
            if not flag:
                tot_err += 1

        return np.array((num_err, err, tot_err))
