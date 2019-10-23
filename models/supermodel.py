import json
import torch
import datetime
import time
import argparse
import numpy as np
import torch.nn as nn
import traceback
from collections import defaultdict

from utils.word_embedding import WordEmbedding
from models.agg_predictor import AggPredictor
from models.col_predictor import ColPredictor
from models.desasc_limit_predictor import DesAscLimitPredictor
from models.having_predictor import HavingPredictor
from models.keyword_predictor import KeyWordPredictor
from models.multisql_predictor import MultiSqlPredictor
from models.root_teminal_predictor import RootTeminalPredictor
from models.andor_predictor import AndOrPredictor
from models.op_predictor import OpPredictor
from preprocess_data import index_to_column_name


SQL_OPS = ('none', 'intersect', 'union', 'except')
KW_OPS = ('where', 'groupBy', 'orderBy')
AGG_OPS = ('max', 'min', 'count', 'sum', 'avg')
ROOT_TERM_OPS = ("root", "terminal")
COND_OPS = ("and", "or")
DEC_ASC_OPS = (("asc", True), ("asc", False), ("desc", True), ("desc", False))
NEW_WHERE_OPS = (
    '=',
    '>',
    '<',
    '>=',
    '<=',
    '!=',
    'like',
    'not in',
    'in',
    'between')
KW_WITH_COL = ("select", "where", "groupBy", "orderBy", "having")


class Stack:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def peek(self):
        return self.items[len(self.items) - 1]

    def size(self):
        return len(self.items)

    def insert(self, i, x):
        return self.items.insert(i, x)


def to_batch_tables(tables, B, table_type):
    # col_lens = []
    col_seq = []
    ts = [
        tables["table_names"],
        tables["column_names"],
        tables["column_types"]]
    tname_toks = [x.split(" ") for x in ts[0]]
    col_type = ts[2]
    cols = [x.split(" ") for xid, x in ts[1]]
    tab_seq = [xid for xid, x in ts[1]]
    cols_add = []
    for tid, col, ct in zip(tab_seq, cols, col_type):
        col_one = [ct]
        if tid == -1:
            tabn = ["all"]
        else:
            if table_type == "no":
                tabn = []
            else:
                tabn = tname_toks[tid]
        for t in tabn:
            if t not in col:
                col_one.append(t)
        col_one.extend(col)
        cols_add.append(col_one)

    col_seq = [cols_add] * B

    return col_seq


class SuperModel(nn.Module):
    def __init__(
            self,
            word_emb,
            col_emb,
            N_word,
            N_col,
            N_h,
            N_depth,
            dropout,
            gpu=True,
            trainable_emb=False,
            table_type="std",
            use_hs=True):
        super(SuperModel, self).__init__()
        self.gpu = gpu
        self.N_h = N_h
        self.N_depth = N_depth
        self.dropout = dropout
        self.trainable_emb = trainable_emb
        self.table_type = table_type
        self.use_hs = use_hs
        self.SQL_TOK = [
            '<UNK>',
            '<END>',
            'WHERE',
            'AND',
            'EQL',
            'GT',
            'LT',
            '<BEG>']

        # word embedding layer
        self.embed_layer = WordEmbedding(word_emb, N_word, gpu,
                                         self.SQL_TOK, trainable=trainable_emb)
        self.q_embed_layer = self.embed_layer

        if not col_emb:
            N_col = None
        else:
            self.embed_layer = WordEmbedding(
                col_emb, N_col, gpu, self.SQL_TOK, trainable=trainable_emb)

        # initial all modules
        self.multi_sql = MultiSqlPredictor(
            N_word=N_word,
            N_col=N_col,
            N_h=N_h,
            N_depth=N_depth,
            dropout=dropout,
            gpu=gpu,
            use_hs=use_hs)
        self.multi_sql.eval()

        self.key_word = KeyWordPredictor(
            N_word=N_word,
            N_col=N_col,
            N_h=N_h,
            N_depth=N_depth,
            dropout=dropout,
            gpu=gpu,
            use_hs=use_hs)
        self.key_word.eval()

        self.col = ColPredictor(
            N_word=N_word,
            N_col=N_col,
            N_h=N_h,
            N_depth=N_depth,
            dropout=dropout,
            gpu=gpu,
            use_hs=use_hs)
        self.col.eval()

        self.op = OpPredictor(
            N_word=N_word,
            N_col=N_col,
            N_h=N_h,
            N_depth=N_depth,
            dropout=dropout,
            gpu=gpu,
            use_hs=use_hs)
        self.op.eval()

        self.agg = AggPredictor(
            N_word=N_word,
            N_col=N_col,
            N_h=N_h,
            N_depth=N_depth,
            dropout=dropout,
            gpu=gpu,
            use_hs=use_hs)
        self.agg.eval()

        self.root_teminal = RootTeminalPredictor(
            N_word=N_word,
            N_col=N_col,
            N_h=N_h,
            N_depth=N_depth,
            dropout=dropout,
            gpu=gpu,
            use_hs=use_hs)
        self.root_teminal.eval()

        self.des_asc = DesAscLimitPredictor(
            N_word=N_word,
            N_col=N_col,
            N_h=N_h,
            N_depth=N_depth,
            dropout=dropout,
            gpu=gpu,
            use_hs=use_hs)
        self.des_asc.eval()

        self.having = HavingPredictor(
            N_word=N_word,
            N_col=N_col,
            N_h=N_h,
            N_depth=N_depth,
            dropout=dropout,
            gpu=gpu,
            use_hs=use_hs)
        self.having.eval()

        self.andor = AndOrPredictor(
            N_word=N_word,
            N_col=N_col,
            N_h=N_h,
            N_depth=N_depth,
            dropout=dropout,
            gpu=gpu,
            use_hs=use_hs)
        self.andor.eval()

        self.softmax = nn.Softmax()  # dim=1
        self.CE = nn.CrossEntropyLoss()
        self.log_softmax = nn.LogSoftmax()
        self.mlsml = nn.MultiLabelSoftMarginLoss()
        self.bce_logit = nn.BCEWithLogitsLoss()
        self.sigm = nn.Sigmoid()
        if gpu:
            self.cuda()
        self.path_not_found = 0

    def forward(self, q_seq, history, tables):
        # if self.part:
        #     return self.part_forward(q_seq,history,tables)
        # else:
        return self.full_forward(q_seq, history, tables)

    def full_forward(self, q_seq, history, tables):
        B = len(q_seq)
        # print("q_seq:{}".format(q_seq))
        # print("Batch size:{}".format(B))
        q_emb_var, q_len = self.q_embed_layer.gen_x_q_batch(q_seq)
        col_seq = to_batch_tables(tables, B, self.table_type)

        col_emb_var, col_name_len, col_len = self.embed_layer.gen_col_batch(
            col_seq)

        mkw_emb_var = self.embed_layer.gen_word_list_embedding(
            ["none", "except", "intersect", "union"], (B))
        mkw_len = np.full(q_len.shape, 4, dtype=np.int64)
        kw_emb_var = self.embed_layer.gen_word_list_embedding(
            ["where", "group by", "order by"], (B))
        kw_len = np.full(q_len.shape, 3, dtype=np.int64)

        stack = Stack()
        stack.push(("root", None))
        history = [["root"]] * B
        andor_cond = ""
        has_limit = False
        # sql = {}
        current_sql = {}
        sql_stack = []
        idx_stack = []
        kw_stack = []
        kw = ""
        nested_label = ""
        has_having = False

        timeout = time.time() + 2  # set timer to prevent infinite recursion in SQL generation
        failed = False
        while not stack.isEmpty():
            if time.time() > timeout:
                failed = True
                break
            vet = stack.pop()
            # print(vet)
            hs_emb_var, hs_len = self.embed_layer.gen_x_history_batch(history)
            if len(idx_stack) > 0 and stack.size() < idx_stack[-1]:
                # print("pop!!!!!!!!!!!!!!!!!!!!!!")
                idx_stack.pop()
                current_sql = sql_stack.pop()
                kw = kw_stack.pop()
                # current_sql = current_sql["sql"]
            # history.append(vet)
            # print("hs_emb:{} hs_len:{}".format(hs_emb_var.size(),hs_len.size()))
            if isinstance(vet, tuple) and vet[0] == "root":
                if history[0][-1] != "root":
                    history[0].append("root")
                    hs_emb_var, hs_len = self.embed_layer.gen_x_history_batch(
                        history)
                if vet[1] != "original":
                    idx_stack.append(stack.size())
                    sql_stack.append(current_sql)
                    kw_stack.append(kw)
                else:
                    idx_stack.append(stack.size())
                    sql_stack.append(sql_stack[-1])
                    kw_stack.append(kw)
                if "sql" in current_sql:
                    current_sql["nested_sql"] = {}
                    current_sql["nested_label"] = nested_label
                    current_sql = current_sql["nested_sql"]
                elif isinstance(vet[1], dict):
                    vet[1]["sql"] = {}
                    current_sql = vet[1]["sql"]
                elif vet[1] != "original":
                    current_sql["sql"] = {}
                    current_sql = current_sql["sql"]
                # print("q_emb_var:{} hs_emb_var:{} mkw_emb_var:{}".format(q_emb_var.size(),hs_emb_var.size(),mkw_emb_var.size()))
                if vet[1] == "nested" or vet[1] == "original":
                    stack.push("none")
                    history[0].append("none")
                else:
                    score = self.multi_sql.forward(
                        q_emb_var, q_len, hs_emb_var, hs_len, mkw_emb_var, mkw_len)
                    label = np.argmax(score[0].data.cpu().numpy())
                    label = SQL_OPS[label]
                    history[0].append(label)
                    stack.push(label)
                if label != "none":
                    nested_label = label

            elif vet in ('intersect', 'except', 'union'):
                stack.push(("root", "nested"))
                stack.push(("root", "original"))
                # history[0].append("root")
            elif vet == "none":
                score = self.key_word.forward(
                    q_emb_var, q_len, hs_emb_var, hs_len, kw_emb_var, kw_len)
                kw_num_score, kw_score = [x.data.cpu().numpy() for x in score]
                # print("kw_num_score:{}".format(kw_num_score))
                # print("kw_score:{}".format(kw_score))
                num_kw = np.argmax(kw_num_score[0])
                kw_score = list(np.argsort(-kw_score[0])[:num_kw])
                kw_score.sort(reverse=True)
                # print("num_kw:{}".format(num_kw))
                for kw in kw_score:
                    stack.push(KW_OPS[kw])
                stack.push("select")
            elif vet in ("select", "orderBy", "where", "groupBy", "having"):
                kw = vet
                current_sql[kw] = []
                history[0].append(vet)
                stack.push(("col", vet))
                # score = self.andor.forward(q_emb_var,q_len,hs_emb_var,hs_len)
                # label = score[0].data.cpu().numpy()
                # andor_cond = COND_OPS[label]
                # history.append("")
            # elif vet == "groupBy":
            #     score = self.having.forward(q_emb_var,q_len,hs_emb_var,hs_len,col_emb_var,col_len,)
            elif isinstance(vet, tuple) and vet[0] == "col":
                # print("q_emb_var:{} hs_emb_var:{} col_emb_var:{}".format(q_emb_var.size(), hs_emb_var.size(),col_emb_var.size()))
                score = self.col.forward(
                    q_emb_var,
                    q_len,
                    hs_emb_var,
                    hs_len,
                    col_emb_var,
                    col_len,
                    col_name_len)
                col_num_score, col_score = [
                    x.data.cpu().numpy() for x in score]
                col_num = np.argmax(col_num_score[0]) + 1  # double check
                cols = np.argsort(-col_score[0])[:col_num]
                # print(col_num)
                # print("col_num_score:{}".format(col_num_score))
                # print("col_score:{}".format(col_score))
                for col in cols:
                    if vet[1] == "where":
                        stack.push(("op", "where", col))
                    elif vet[1] != "groupBy":
                        stack.push(("agg", vet[1], col))
                    elif vet[1] == "groupBy":
                        history[0].append(index_to_column_name(col, tables))
                        current_sql[kw].append(
                            index_to_column_name(col, tables))
                # predict and or or when there is multi col in where condition
                if col_num > 1 and vet[1] == "where":
                    score = self.andor.forward(
                        q_emb_var, q_len, hs_emb_var, hs_len)
                    label = np.argmax(score[0].data.cpu().numpy())
                    andor_cond = COND_OPS[label]
                    current_sql[kw].append(andor_cond)
                if vet[1] == "groupBy" and col_num > 0:
                    score = self.having.forward(
                        q_emb_var,
                        q_len,
                        hs_emb_var,
                        hs_len,
                        col_emb_var,
                        col_len,
                        col_name_len,
                        np.full(
                            B,
                            cols[0],
                            dtype=np.int64))
                    label = np.argmax(score[0].data.cpu().numpy())
                    if label == 1:
                        has_having = (label == 1)
                        # stack.insert(-col_num,"having")
                        stack.push("having")
                # history.append(index_to_column_name(cols[-1], tables[0]))
            elif isinstance(vet, tuple) and vet[0] == "agg":
                history[0].append(index_to_column_name(vet[2], tables))
                if vet[1] not in ("having", "orderBy"):  # DEBUG-ed 20180817
                    try:
                        current_sql[kw].append(
                            index_to_column_name(vet[2], tables))
                    except Exception as e:
                        # print(e)
                        traceback.print_exc()
                        print(
                            "history:{},current_sql:{} stack:{}".format(
                                history[0], current_sql, stack.items))
                        print("idx_stack:{}".format(idx_stack))
                        print("sql_stack:{}".format(sql_stack))
                        exit(1)
                hs_emb_var, hs_len = self.embed_layer.gen_x_history_batch(
                    history)

                score = self.agg.forward(
                    q_emb_var,
                    q_len,
                    hs_emb_var,
                    hs_len,
                    col_emb_var,
                    col_len,
                    col_name_len,
                    np.full(
                        B,
                        vet[2],
                        dtype=np.int64))
                agg_num_score, agg_score = [
                    x.data.cpu().numpy() for x in score]
                agg_num = np.argmax(agg_num_score[0])  # double check
                agg_idxs = np.argsort(-agg_score[0])[:agg_num]
                # print("agg:{}".format([AGG_OPS[agg] for agg in agg_idxs]))
                if len(agg_idxs) > 0:
                    history[0].append(AGG_OPS[agg_idxs[0]])
                    if vet[1] not in ("having", "orderBy"):
                        current_sql[kw].append(AGG_OPS[agg_idxs[0]])
                    elif vet[1] == "orderBy":
                        # DEBUG-ed 20180817
                        stack.push(("des_asc", vet[2], AGG_OPS[agg_idxs[0]]))
                    else:
                        stack.push(
                            ("op", "having", vet[2], AGG_OPS[agg_idxs[0]]))
                for agg in agg_idxs[1:]:
                    history[0].append(index_to_column_name(vet[2], tables))
                    history[0].append(AGG_OPS[agg])
                    if vet[1] not in ("having", "orderBy"):
                        current_sql[kw].append(
                            index_to_column_name(vet[2], tables))
                        current_sql[kw].append(AGG_OPS[agg])
                    elif vet[1] == "orderBy":
                        stack.push(("des_asc", vet[2], AGG_OPS[agg]))
                    else:
                        stack.push(("op", "having", vet[2], agg_idxs))
                if len(agg_idxs) == 0:
                    if vet[1] not in ("having", "orderBy"):
                        current_sql[kw].append("none_agg")
                    elif vet[1] == "orderBy":
                        stack.push(("des_asc", vet[2], "none_agg"))
                    else:
                        stack.push(("op", "having", vet[2], "none_agg"))
                # current_sql[kw].append([AGG_OPS[agg] for agg in agg_idxs])
                # if vet[1] == "having":
                #     stack.push(("op","having",vet[2],agg_idxs))
                # if vet[1] == "orderBy":
                #     stack.push(("des_asc",vet[2],agg_idxs))
                # if vet[1] == "groupBy" and has_having:
                #     stack.push("having")
            elif isinstance(vet, tuple) and vet[0] == "op":
                if vet[1] == "where":
                    # current_sql[kw].append(index_to_column_name(vet[2], tables))
                    history[0].append(index_to_column_name(vet[2], tables))
                    hs_emb_var, hs_len = self.embed_layer.gen_x_history_batch(
                        history)

                score = self.op.forward(
                    q_emb_var,
                    q_len,
                    hs_emb_var,
                    hs_len,
                    col_emb_var,
                    col_len,
                    col_name_len,
                    np.full(
                        B,
                        vet[2],
                        dtype=np.int64))

                op_num_score, op_score = [x.data.cpu().numpy() for x in score]
                # num_score 0 maps to 1 in truth, must have at least one op
                op_num = np.argmax(op_num_score[0]) + 1
                ops = np.argsort(-op_score[0])[:op_num]
                # current_sql[kw].append([NEW_WHERE_OPS[op] for op in ops])
                if op_num > 0:
                    history[0].append(NEW_WHERE_OPS[ops[0]])
                    if vet[1] == "having":
                        stack.push(("root_teminal", vet[2], vet[3], ops[0]))
                    else:
                        stack.push(("root_teminal", vet[2], ops[0]))
                    # current_sql[kw].append(NEW_WHERE_OPS[ops[0]])
                for op in ops[1:]:
                    history[0].append(index_to_column_name(vet[2], tables))
                    history[0].append(NEW_WHERE_OPS[op])
                    # current_sql[kw].append(index_to_column_name(vet[2], tables))
                    # current_sql[kw].append(NEW_WHERE_OPS[op])
                    if vet[1] == "having":
                        stack.push(("root_teminal", vet[2], vet[3], op))
                    else:
                        stack.push(("root_teminal", vet[2], op))
                # stack.push(("root_teminal",vet[2]))
            elif isinstance(vet, tuple) and vet[0] == "root_teminal":
                score = self.root_teminal.forward(
                    q_emb_var,
                    q_len,
                    hs_emb_var,
                    hs_len,
                    col_emb_var,
                    col_len,
                    col_name_len,
                    np.full(
                        B,
                        vet[1],
                        dtype=np.int64))

                label = np.argmax(score[0].data.cpu().numpy())
                label = ROOT_TERM_OPS[label]
                if len(vet) == 4:
                    current_sql[kw].append(
                        index_to_column_name(vet[1], tables))
                    current_sql[kw].append(vet[2])
                    current_sql[kw].append(NEW_WHERE_OPS[vet[3]])
                else:
                    # print("kw:{}".format(kw))
                    try:
                        current_sql[kw].append(
                            index_to_column_name(vet[1], tables))
                    except Exception as e:
                        # print(e)
                        traceback.print_exc()
                        print(
                            "history:{},current_sql:{} stack:{}".format(
                                history[0], current_sql, stack.items))
                        print("idx_stack:{}".format(idx_stack))
                        print("sql_stack:{}".format(sql_stack))
                        exit(1)
                    current_sql[kw].append(NEW_WHERE_OPS[vet[2]])
                if label == "root":
                    history[0].append("root")
                    current_sql[kw].append({})
                    # current_sql = current_sql[kw][-1]
                    stack.push(("root", current_sql[kw][-1]))
                else:
                    current_sql[kw].append("terminal")
            elif isinstance(vet, tuple) and vet[0] == "des_asc":
                current_sql[kw].append(index_to_column_name(vet[1], tables))
                current_sql[kw].append(vet[2])
                score = self.des_asc.forward(
                    q_emb_var,
                    q_len,
                    hs_emb_var,
                    hs_len,
                    col_emb_var,
                    col_len,
                    col_name_len,
                    np.full(
                        B,
                        vet[1],
                        dtype=np.int64))
                label = np.argmax(score[0].data.cpu().numpy())
                dec_asc, has_limit = DEC_ASC_OPS[label]
                history[0].append(dec_asc)
                current_sql[kw].append(dec_asc)
                current_sql[kw].append(has_limit)
        # print("{}".format(current_sql))

        if failed:
            return None
        print("history:{}".format(history[0]))
        if len(sql_stack) > 0:
            current_sql = sql_stack[0]
        # print("{}".format(current_sql))
        return current_sql

    def gen_col(self, col, table, table_alias_dict):
        colname = table["column_names_original"][col[2]][1]
        table_idx = table["column_names_original"][col[2]][0]
        if table_idx not in table_alias_dict:
            return colname
        return "T{}.{}".format(table_alias_dict[table_idx], colname)

    def gen_group_by(self, sql, kw, table, table_alias_dict):
        ret = []
        for i in range(0, len(sql)):
            # if len(sql[i+1]) == 0:
            # if sql[i+1] == "none_agg":
            ret.append(self.gen_col(sql[i], table, table_alias_dict))
            # else:
            #     ret.append("{}({})".format(sql[i+1], self.gen_col(sql[i], table, table_alias_dict)))
            # for agg in sql[i+1]:
            #     ret.append("{}({})".format(agg,gen_col(sql[i],table,table_alias_dict)))
        return "{} {}".format(kw, ",".join(ret))

    def gen_select(self, sql, kw, table, table_alias_dict):
        ret = []
        for i in range(0, len(sql), 2):
            # if len(sql[i+1]) == 0:
            if sql[i +
                   1] == "none_agg" or not isinstance(sql[i +
                                                          1], basestring):  # DEBUG-ed 20180817
                ret.append(self.gen_col(sql[i], table, table_alias_dict))
            else:
                ret.append("{}({})".format(
                    sql[i + 1], self.gen_col(sql[i], table, table_alias_dict)))
            # for agg in sql[i+1]:
            #     ret.append("{}({})".format(agg,gen_col(sql[i],table,table_alias_dict)))
        return "{} {}".format(kw, ",".join(ret))

    def gen_where(self, sql, table, table_alias_dict):
        if len(sql) == 0:
            return ""
        start_idx = 0
        andor = "and"
        if isinstance(sql[0], basestring):
            start_idx += 1
            andor = sql[0]
        ret = []
        for i in range(start_idx, len(sql), 3):
            # rewrite to stop a bug
            if i + 2 < len(sql):
                col = self.gen_col(sql[i], table, table_alias_dict)
                op = sql[i + 1]
                val = sql[i + 2]
                where_item = ""
            else:
                break
            if val == "terminal":
                where_item = "{} {} '{}'".format(col, op, val)
            else:
                val = self.gen_sql(val, table)
                where_item = "{} {} ({})".format(col, op, val)
            if op == "between":
                # TODO temprarily fixed
                where_item += " and 'terminal'"
            ret.append(where_item)
        return "where {}".format(" {} ".format(andor).join(ret))

    def gen_orderby(self, sql, table, table_alias_dict):
        ret = []
        limit = ""
        if sql[-1]:
            limit = "limit 1"
        for i in range(0, len(sql), 4):
            if sql[i +
                   1] == "none_agg" or not isinstance(sql[i +
                                                          1], basestring):  # DEBUG-ed 20180817
                ret.append("{} {}".format(self.gen_col(
                    sql[i], table, table_alias_dict), sql[i + 2]))
            else:
                ret.append("{}({}) {}".format(
                    sql[i + 1], self.gen_col(sql[i], table, table_alias_dict), sql[i + 2]))
        return "order by {} {}".format(",".join(ret), limit)

    def gen_having(self, sql, table, table_alias_dict):
        ret = []
        for i in range(0, len(sql), 4):
            if sql[i + 1] == "none_agg":
                col = self.gen_col(sql[i], table, table_alias_dict)
            else:
                col = "{}({})".format(
                    sql[i + 1], self.gen_col(sql[i], table, table_alias_dict))
            op = sql[i + 2]
            val = sql[i + 3]
            if val == "terminal":
                ret.append("{} {} '{}'".format(col, op, val))
            else:
                val = self.gen_sql(val, table)
                ret.append("{} {} ({})".format(col, op, val))
        return "having {}".format(",".join(ret))

    def find_shortest_path(self, start, end, graph):
        stack = [[start, []]]
        visited = set()
        while len(stack) > 0:
            ele, history = stack.pop()
            if ele == end:
                return history
            for node in graph[ele]:
                if node[0] not in visited:
                    stack.append((node[0], history + [(node[0], node[1])]))
                    visited.add(node[0])
        print("table {} table {}".format(start, end))
        # print("could not find path!!!!!{}".format(self.path_not_found))
        self.path_not_found += 1
        # return []

    def gen_from(self, candidate_tables, table):
        def find(d, col):
            if d[col] == -1:
                return col
            return find(d, d[col])

        def union(d, c1, c2):
            r1 = find(d, c1)
            r2 = find(d, c2)
            if r1 == r2:
                return
            d[r1] = r2

        ret = ""
        if len(candidate_tables) <= 1:
            if len(candidate_tables) == 1:
                ret = "from {}".format(
                    table["table_names_original"][list(candidate_tables)[0]])
            else:
                ret = "from {}".format(table["table_names_original"][0])
            # TODO: temporarily settings
            return {}, ret
        # print("candidate:{}".format(candidate_tables))
        table_alias_dict = {}
        uf_dict = {}
        for t in candidate_tables:
            uf_dict[t] = -1
        idx = 1
        graph = defaultdict(list)
        for acol, bcol in table["foreign_keys"]:
            t1 = table["column_names"][acol][0]
            t2 = table["column_names"][bcol][0]
            graph[t1].append((t2, (acol, bcol)))
            graph[t2].append((t1, (bcol, acol)))
            # if t1 in candidate_tables and t2 in candidate_tables:
            #     r1 = find(uf_dict,t1)
            #     r2 = find(uf_dict,t2)
            #     if r1 == r2:
            #         continue
            #     union(uf_dict,t1,t2)
            #     if len(ret) == 0:
            #         ret = "from {} as T{} join {} as T{} on T{}.{}=T{}.{}".format(table["table_names"][t1],idx,table["table_names"][t2],
            #                                                                       idx+1,idx,table["column_names_original"][acol][1],idx+1,
            #                                                                       table["column_names_original"][bcol][1])
            #         table_alias_dict[t1] = idx
            #         table_alias_dict[t2] = idx+1
            #         idx += 2
            #     else:
            #         if t1 in table_alias_dict:
            #             old_t = t1
            #             new_t = t2
            #             acol,bcol = bcol,acol
            #         elif t2 in table_alias_dict:
            #             old_t = t2
            #             new_t = t1
            #         else:
            #             ret = "{} join {} as T{} join {} as T{} on T{}.{}=T{}.{}".format(ret,table["table_names"][t1], idx,
            #                                                                           table["table_names"][t2],
            #                                                                           idx + 1, idx,
            #                                                                           table["column_names_original"][acol][1],
            #                                                                           idx + 1,
            #                                                                           table["column_names_original"][bcol][1])
            #             table_alias_dict[t1] = idx
            #             table_alias_dict[t2] = idx + 1
            #             idx += 2
            #             continue
            #         ret = "{} join {} as T{} on T{}.{}=T{}.{}".format(ret,new_t,idx,idx,table["column_names_original"][acol][1],
            #                                                        table_alias_dict[old_t],table["column_names_original"][bcol][1])
            #         table_alias_dict[new_t] = idx
            #         idx += 1
        # visited = set()
        candidate_tables = list(candidate_tables)
        start = candidate_tables[0]
        table_alias_dict[start] = idx
        idx += 1
        ret = "from {} as T1".format(table["table_names_original"][start])
        try:
            for end in candidate_tables[1:]:
                if end in table_alias_dict:
                    continue
                path = self.find_shortest_path(start, end, graph)
                prev_table = start
                if not path:
                    table_alias_dict[end] = idx
                    idx += 1
                    ret = "{} join {} as T{}".format(
                        ret, table["table_names_original"][end], table_alias_dict[end], )
                    continue
                for node, (acol, bcol) in path:
                    if node in table_alias_dict:
                        prev_table = node
                        continue
                    table_alias_dict[node] = idx
                    idx += 1
                    ret = "{} join {} as T{} on T{}.{} = T{}.{}".format(
                        ret,
                        table["table_names_original"][node],
                        table_alias_dict[node],
                        table_alias_dict[prev_table],
                        table["column_names_original"][acol][1],
                        table_alias_dict[node],
                        table["column_names_original"][bcol][1])
                    prev_table = node
        except BaseException:
            traceback.print_exc()
            print("db:{}".format(table["db_id"]))
            # print(table["db_id"])
            return table_alias_dict, ret
        # if len(candidate_tables) != len(table_alias_dict):
        #     print("error in generate from clause!!!!!")
        return table_alias_dict, ret

    def gen_sql(self, sql, table):
        select_clause = ""
        from_clause = ""
        groupby_clause = ""
        orderby_clause = ""
        having_clause = ""
        where_clause = ""
        nested_clause = ""
        cols = {}
        candidate_tables = set()
        nested_sql = {}
        nested_label = ""
        parent_sql = sql
        # if "sql" in sql:
        #     sql = sql["sql"]
        if "nested_label" in sql:
            nested_label = sql["nested_label"]
            nested_sql = sql["nested_sql"]
            sql = sql["sql"]
        elif "sql" in sql:
            sql = sql["sql"]
        for key in sql:
            if key not in KW_WITH_COL:
                continue
            for item in sql[key]:
                if isinstance(item, tuple) and len(item) == 3:
                    if table["column_names"][item[2]][0] != -1:
                        candidate_tables.add(table["column_names"][item[2]][0])
        table_alias_dict, from_clause = self.gen_from(candidate_tables, table)
        ret = []
        if "select" in sql:
            select_clause = self.gen_select(
                sql["select"], "select", table, table_alias_dict)
            if len(select_clause) > 0:
                ret.append(select_clause)
            else:
                print("select not found:{}".format(parent_sql))
        else:
            print("select not found:{}".format(parent_sql))
        if len(from_clause) > 0:
            ret.append(from_clause)
        if "where" in sql:
            where_clause = self.gen_where(
                sql["where"], table, table_alias_dict)
            if len(where_clause) > 0:
                ret.append(where_clause)
        if "groupBy" in sql:  # DEBUG-ed order
            groupby_clause = self.gen_group_by(
                sql["groupBy"], "group by", table, table_alias_dict)
            if len(groupby_clause) > 0:
                ret.append(groupby_clause)
        if "orderBy" in sql:
            orderby_clause = self.gen_orderby(
                sql["orderBy"], table, table_alias_dict)
            if len(orderby_clause) > 0:
                ret.append(orderby_clause)
        if "having" in sql:
            having_clause = self.gen_having(
                sql["having"], table, table_alias_dict)
            if len(having_clause) > 0:
                ret.append(having_clause)
        if len(nested_label) > 0:
            nested_clause = "{} {}".format(
                nested_label, self.gen_sql(
                    nested_sql, table))
            if len(nested_clause) > 0:
                ret.append(nested_clause)
        return " ".join(ret)

    def check_acc(self, pred_sql, gt_sql):
        pass
