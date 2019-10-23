# _*_ coding: utf_8 _*

import argparse
import codecs
import json
import os
import sys
from collections import defaultdict

defaultencoding = 'utf-8'
if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)


OLD_WHERE_OPS = (
    'not',
    'between',
    '=',
    '>',
    '<',
    '>=',
    '<=',
    '!=',
    'in',
    'like',
    'is',
    'exists')
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
    'between',
    'is')
NEW_WHERE_DICT = {
    '=': 0,
    '>': 1,
    '<': 2,
    '>=': 3,
    '<=': 4,
    '!=': 5,
    'like': 6,
    'not in': 7,
    'in': 8,
    'between': 9,
    'is': 10
}
# SQL_OPS = ('none','intersect', 'union', 'except')
SQL_OPS = {
    'none': 0,
    'intersect': 1,
    'union': 2,
    'except': 3
}
KW_DICT = {
    'where': 0,
    'groupBy': 1,
    'orderBy': 2
}
ORDER_OPS = {
    'desc': 0,
    'asc': 1}
AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')

COND_OPS = {
    'and': 0,
    'or': 1
}


def convert_to_op_index(is_not, op):
    op = OLD_WHERE_OPS[op]
    if is_not and op == "in":
        return 7
    try:
        return NEW_WHERE_DICT[op]
    except BaseException:
        print("Unsupport op: {}".format(op))
        return -1


def index_to_column_name(index, table):
    column_name = table["column_names"][index][1]
    table_index = table["column_names"][index][0]
    table_name = table["table_names"][table_index]
    return table_name, column_name, index


def get_label_cols(with_join, fk_dict, labels):
    # list(set([l[1][i][0][2] for i in range(min(len(l[1]), 3))]))
    cols = set()
    ret = []
    for i in range(len(labels)):
        cols.add(labels[i][0][2])
        if len(cols) > 3:
            break
    for col in cols:
        # ret.append([col])
        if with_join and len(fk_dict[col]) > 0:
            ret.append([col] + fk_dict[col])
        else:
            ret.append(col)
    return ret


class MultiSqlPredictor:
    def __init__(self, question, sql, history):
        self.sql = sql
        self.question = question
        self.history = history
        self.keywords = ('intersect', 'except', 'union')

    def generate_output(self):
        for key in self.sql:
            if key in self.keywords and self.sql[key]:
                return self.history + ['root'], key, self.sql[key]
        return self.history + ['root'], 'none', self.sql


class KeyWordPredictor:
    def __init__(self, question, sql, history):
        self.sql = sql
        self.question = question
        self.history = history
        self.keywords = (
            'select',
            'where',
            'groupBy',
            'orderBy',
            'limit',
            'having')

    def generate_output(self):
        sql_keywords = []
        for key in self.sql:
            if key in self.keywords and self.sql[key]:
                sql_keywords.append(key)
        return self.history, [len(sql_keywords), sql_keywords], self.sql


class ColPredictor:
    def __init__(self, question, sql, table, history, kw=None):
        self.sql = sql
        self.question = question
        self.history = history
        self.table = table
        self.keywords = ('select', 'where', 'groupBy', 'orderBy', 'having')
        self.kw = kw

    def generate_output(self):
        ret = []
        candidate_keys = self.sql.keys()
        if self.kw:
            candidate_keys = [self.kw]
        for key in candidate_keys:
            if key in self.keywords and self.sql[key]:
                cols = []
                sqls = []
                if key == 'groupBy':
                    sql_cols = self.sql[key]
                    for col in sql_cols:
                        cols.append(
                            (index_to_column_name(
                                col[1], self.table), col[2]))
                        sqls.append(col)
                elif key == 'orderBy':
                    sql_cols = self.sql[key][1]
                    for col in sql_cols:
                        cols.append(
                            (index_to_column_name(
                                col[1][1], self.table), col[1][2]))
                        sqls.append(col)
                elif key == 'select':
                    sql_cols = self.sql[key][1]
                    for col in sql_cols:
                        cols.append(
                            (index_to_column_name(
                                col[1][1][1],
                                self.table),
                                col[1][1][2]))
                        sqls.append(col)
                elif key == 'where' or key == 'having':
                    sql_cols = self.sql[key]
                    for col in sql_cols:
                        if not isinstance(col, list):
                            continue
                        try:
                            cols.append(
                                (index_to_column_name(
                                    col[2][1][1],
                                    self.table),
                                    col[2][1][2]))
                        except BaseException:
                            print(
                                "Key:{} Col:{} Question:{}".format(
                                    key, col, self.question))
                        sqls.append(col)
                ret.append((
                    self.history + [key], (len(cols), cols), sqls
                ))
        return ret
        # ret.append(history+[key],)


class OpPredictor:
    def __init__(self, question, sql, history):
        self.sql = sql
        self.question = question
        self.history = history
        # self.keywords = ('select', 'where', 'groupBy', 'orderBy', 'having')

    def generate_output(self):
        return self.history, convert_to_op_index(
            self.sql[0], self.sql[1]), (self.sql[3], self.sql[4])


class AggPredictor:
    def __init__(self, question, sql, history, kw=None):
        self.sql = sql
        self.question = question
        self.history = history
        self.kw = kw

    def generate_output(self):
        label = -1
        if self.kw:
            key = self.kw
        else:
            key = self.history[-2]
        if key == 'select':
            label = self.sql[0]
        elif key == 'orderBy':
            label = self.sql[1][0]
        elif key == 'having':
            label = self.sql[2][1][0]
        return self.history, label


# class RootTemPredictor:
#     def __init__(self, question, sql):
#         self.sql = sql
#         self.question = question
#         self.keywords = ('intersect', 'except', 'union')
#
#     def generate_output(self):
#         for key in self.sql:
#             if key in self.keywords:
#                 return ['ROOT'], key, self.sql[key]
#         return ['ROOT'], 'none', self.sql


class DesAscPredictor:
    def __init__(self, question, sql, table, history):
        self.sql = sql
        self.question = question
        self.history = history
        self.table = table

    def generate_output(self):
        for key in self.sql:
            if key == "orderBy" and self.sql[key]:
                # self.history.append(key)
                try:
                    col = self.sql[key][1][0][1][1]
                except BaseException:
                    print("question:{} sql:{}".format(self.question, self.sql))
                # self.history.append(index_to_column_name(col, self.table))
                # self.history.append(self.sql[key][1][0][1][0])
                if self.sql[key][0] == "asc" and self.sql["limit"]:
                    label = 0
                elif self.sql[key][0] == "asc" and not self.sql["limit"]:
                    label = 1
                elif self.sql[key][0] == "desc" and self.sql["limit"]:
                    label = 2
                else:
                    label = 3
                return self.history + \
                    [index_to_column_name(col, self.table), self.sql[key][1][0][1][0]], label


class AndOrPredictor:
    def __init__(self, question, sql, table, history):
        self.sql = sql
        self.question = question
        self.history = history
        self.table = table

    def generate_output(self):
        if 'where' in self.sql and self.sql['where'] and len(
                self.sql['where']) > 1:
            return self.history, COND_OPS[self.sql['where'][1]]
        return self.history, -1


def parser_item_with_long_history(
        question_tokens,
        sql,
        table,
        history,
        dataset):
    table_schema = [
        table["table_names"],
        table["column_names"],
        table["column_types"]
    ]
    stack = [("root", sql)]
    with_join = False
    fk_dict = defaultdict(list)
    for fk in table["foreign_keys"]:
        fk_dict[fk[0]].append(fk[1])
        fk_dict[fk[1]].append(fk[0])
    while len(stack) > 0:
        node = stack.pop()
        if node[0] == "root":
            history, label, sql = MultiSqlPredictor(
                question_tokens, node[1], history).generate_output()
            dataset['multi_sql_dataset'].append({
                "question_tokens": question_tokens,
                "ts": table_schema,
                "history": history[:],
                "label": SQL_OPS[label]
            })
            history.append(label)
            if label == "none":
                stack.append((label, sql))
            else:
                node[1][label] = None
                stack.append((label, node[1], sql))
            # if label != "none":
                # stack.append(("none",node[1]))
        elif node[0] in ('intersect', 'except', 'union'):
            stack.append(("root", node[1]))
            stack.append(("root", node[2]))
        elif node[0] == "none":
            with_join = len(node[1]["from"]["table_units"]) > 1
            history, label, sql = KeyWordPredictor(
                question_tokens, node[1], history).generate_output()
            label_idxs = []
            for item in label[1]:
                if item in KW_DICT:
                    label_idxs.append(KW_DICT[item])
            label_idxs.sort()
            dataset['keyword_dataset'].append({
                "question_tokens": question_tokens,
                "ts": table_schema,
                "history": history[:],
                "label": label_idxs
            })
            if "having" in label[1]:
                stack.append(("having", node[1]))
            if "orderBy" in label[1]:
                stack.append(("orderBy", node[1]))
            if "groupBy" in label[1]:
                if "having" in label[1]:
                    dataset['having_dataset'].append({
                        "question_tokens": question_tokens,
                        "ts": table_schema,
                        "history": history[:],
                        "gt_col": node[1]["groupBy"][0][1],
                        "label": 1
                    })
                else:
                    dataset['having_dataset'].append({
                        "question_tokens": question_tokens,
                        "ts": table_schema,
                        "history": history[:],
                        "gt_col": node[1]["groupBy"][0][1],
                        "label": 0
                    })
                stack.append(("groupBy", node[1]))
            if "where" in label[1]:
                stack.append(("where", node[1]))
            if "select" in label[1]:
                stack.append(("select", node[1]))
        elif node[0] in ("select", "having", "orderBy"):
            # if node[0] != "orderBy":
            history.append(node[0])
            if node[0] == "orderBy":
                orderby_ret = DesAscPredictor(
                    question_tokens, node[1], table, history).generate_output()
                if orderby_ret:
                    dataset['des_asc_dataset'].append({
                        "question_tokens": question_tokens,
                        "ts": table_schema,
                        "history": orderby_ret[0],
                        "gt_col": node[1]["orderBy"][1][0][1][1],
                        "label": orderby_ret[1]
                    })
                    # history.append(orderby_ret[1])
            col_ret = ColPredictor(
                question_tokens,
                node[1],
                table,
                history,
                node[0]).generate_output()
            agg_col_dict = dict()
            op_col_dict = dict()
            for h, l, s in col_ret:
                if l[0] == 0:
                    print("Warning: predicted 0 columns!")
                    continue
                dataset['col_dataset'].append({
                    "question_tokens": question_tokens,
                    "ts": table_schema,
                    "history": history[:],
                    "label": get_label_cols(with_join, fk_dict, l[1])
                })
                for col, sql_item in zip(l[1], s):
                    key = "{}{}{}".format(col[0][0], col[0][1], col[0][2])
                    if key not in agg_col_dict:
                        agg_col_dict[key] = [(sql_item, col[0])]
                    else:
                        agg_col_dict[key].append((sql_item, col[0]))
                    if key not in op_col_dict:
                        op_col_dict[key] = [(sql_item, col[0])]
                    else:
                        op_col_dict[key].append((sql_item, col[0]))
                for key in agg_col_dict:
                    stack.append(
                        ("col", node[0], agg_col_dict[key], op_col_dict[key]))
        elif node[0] == "col":
            history.append(node[2][0][1])
            if node[1] == "where":
                stack.append(("op", node[2], "where"))
            else:
                labels = []
                for sql_item, col in node[2]:
                    _, label = AggPredictor(
                        question_tokens, sql_item, history, node[1]).generate_output()
                    if label - 1 >= 0:
                        labels.append(label - 1)

                # print(node[2][0][1][2])
                dataset['agg_dataset'].append({
                    "question_tokens": question_tokens,
                    "ts": table_schema,
                    "history": history[:],
                    "gt_col": node[2][0][1][2],
                    "label": labels[:min(len(labels), 3)]
                })
                if node[1] == "having":
                    stack.append(("op", node[2], "having"))
                # if len(labels) == 0:
                #     history.append("none")
                # else:
                if len(labels) > 0:
                    history.append(AGG_OPS[labels[0] + 1])
        elif node[0] == "op":
            # history.append(node[1][0][1])
            labels = []
            # if len(labels) > 2:
            #     print(question_tokens)
            dataset['op_dataset'].append({
                "question_tokens": question_tokens,
                "ts": table_schema,
                "history": history[:],
                "gt_col": node[1][0][1][2],
                "label": labels
            })

            for sql_item, col in node[1]:
                _, label, s = OpPredictor(
                    question_tokens, sql_item, history).generate_output()
                if label != -1:
                    labels.append(label)
                    history.append(NEW_WHERE_OPS[label])
                if isinstance(s[0], dict):
                    stack.append(("root", s[0]))
                    # history.append("root")
                    dataset['root_tem_dataset'].append({
                        "question_tokens": question_tokens,
                        "ts": table_schema,
                        "history": history[:],
                        "gt_col": node[1][0][1][2],
                        "label": 0
                    })
                else:
                    dataset['root_tem_dataset'].append({
                        "question_tokens": question_tokens,
                        "ts": table_schema,
                        "history": history[:],
                        "gt_col": node[1][0][1][2],
                        "label": 1
                    })
                    # history.append("terminal")
            if len(labels) > 2:
                print(question_tokens)
            dataset['op_dataset'][-1]["label"] = labels
        elif node[0] == "where":
            history.append(node[0])
            hist, label = AndOrPredictor(
                question_tokens, node[1], table, history).generate_output()
            if label != -1:
                dataset['andor_dataset'].append({
                    "question_tokens": question_tokens,
                    "ts": table_schema,
                    "history": history[:],
                    "label": label
                })
            col_ret = ColPredictor(
                question_tokens,
                node[1],
                table,
                history,
                "where").generate_output()
            op_col_dict = dict()
            for h, l, s in col_ret:
                if l[0] == 0:
                    print("Warning: predicted 0 columns!")
                    continue
                dataset['col_dataset'].append({
                    "question_tokens": question_tokens,
                    "ts": table_schema,
                    "history": history[:],
                    "label": get_label_cols(with_join, fk_dict, l[1])
                })
                for col, sql_item in zip(l[1], s):
                    key = "{}{}{}".format(col[0][0], col[0][1], col[0][2])
                    if key not in op_col_dict:
                        op_col_dict[key] = [(sql_item, col[0])]
                    else:
                        op_col_dict[key].append((sql_item, col[0]))
                for key in op_col_dict:
                    stack.append(("col", "where", op_col_dict[key]))
        elif node[0] == "groupBy":
            history.append(node[0])
            col_ret = ColPredictor(
                question_tokens,
                node[1],
                table,
                history,
                node[0]).generate_output()
            agg_col_dict = dict()
            for h, l, s in col_ret:
                if l[0] == 0:
                    print("Warning: predicted 0 columns!")
                    continue
                dataset['col_dataset'].append({
                    "question_tokens": question_tokens,
                    "ts": table_schema,
                    "history": history[:],
                    "label": get_label_cols(with_join, fk_dict, l[1])
                })
                for col, sql_item in zip(l[1], s):
                    key = "{}{}{}".format(col[0][0], col[0][1], col[0][2])
                    if key not in agg_col_dict:
                        agg_col_dict[key] = [(sql_item, col[0])]
                    else:
                        agg_col_dict[key].append((sql_item, col[0]))
                for key in agg_col_dict:
                    stack.append(("col", node[0], agg_col_dict[key]))


def parser_item(question_tokens, sql, table, history, dataset):
    # try:
    #     question_tokens = item['question_toks']
    # except:
    #     print(item)
    # sql = item['sql']
    table_schema = [
        table["table_names"],
        table["column_names"],
        table["column_types"]
    ]
    history, label, sql = MultiSqlPredictor(
        question_tokens, sql, history).generate_output()
    dataset['multi_sql_dataset'].append({
        "question_tokens": question_tokens,
        "ts": table_schema,
        "history": history[:],
        "label": SQL_OPS[label]
    })
    history.append(label)
    history, label, sql = KeyWordPredictor(
        question_tokens, sql, history).generate_output()
    label_idxs = []
    for item in label[1]:
        if item in KW_DICT:
            label_idxs.append(KW_DICT[item])
    label_idxs.sort()
    dataset['keyword_dataset'].append({
        "question_tokens": question_tokens,
        "ts": table_schema,
        "history": history[:],
        "label": label_idxs
    })
    hist, label = AndOrPredictor(
        question_tokens, sql, table, history).generate_output()
    if label != -1:
        dataset['andor_dataset'].append({
            "question_tokens": question_tokens,
            "ts": table_schema,
            "history": hist[:] + ["where"],
            "label": label
        })
    orderby_ret = DesAscPredictor(
        question_tokens,
        sql,
        table,
        history).generate_output()
    if orderby_ret:
        dataset['des_asc_dataset'].append({
            "question_tokens": question_tokens,
            "ts": table_schema,
            "history": orderby_ret[0][:],
            "label": orderby_ret[1]
        })
    col_ret = ColPredictor(
        question_tokens,
        sql,
        table,
        history).generate_output()
    agg_candidates = []
    op_candidates = []
    for h, l, s in col_ret:
        if l[0] == 0:
            print("Warning: predicted 0 columns!")
            continue
        dataset['col_dataset'].append({
            "question_tokens": question_tokens,
            "ts": table_schema,
            "history": h[:],
            "label": list(set([l[1][i][0][2] for i in range(min(len(l[1]), 3))]))
        })
        for col, sql_item in zip(l[1], s):
            if h[-1] in ('where', 'having'):
                op_candidates.append((h + [col[0]], sql_item))
            if h[-1] in ('select', 'orderBy', 'having'):
                agg_candidates.append((h + [col[0]], sql_item))
            if h[-1] == "groupBy":
                label = 0
                if sql["having"]:
                    label = 1
                dataset['having_dataset'].append({
                    "question_tokens": question_tokens,
                    "ts": table_schema,
                    "history": h[:] + [col[0]],
                    "label": label
                })

    op_col_dict = dict()
    for h, sql_item in op_candidates:
        _, label, s = OpPredictor(
            question_tokens, sql_item, h).generate_output()
        if label == -1:
            continue
        key = "{}{}".format(h[-2], h[-1][2])
        label = NEW_WHERE_OPS[label]
        if key in op_col_dict:
            op_col_dict[key][1].append(label)
        else:
            op_col_dict[key] = [h[:], [label]]
        # dataset['op_dataset'].append({
        #     "question_tokens": question_tokens,
        #     "ts": table_schema,
        #     "history": h[:],
        #     "label": label
        # })
        if isinstance(s[0], dict):
            dataset['root_tem_dataset'].append({
                "question_tokens": question_tokens,
                "ts": table_schema,
                "history": h[:] + [label],
                "label": 0
            })
            parser_item(question_tokens, s[0], table, h[:] + [label], dataset)
        else:
            dataset['root_tem_dataset'].append({
                "question_tokens": question_tokens,
                "ts": table_schema,
                "history": h[:] + [label],
                "label": 1
            })
    for key in op_col_dict:
        # if len(op_col_dict[key][1]) > 1:
        #     print("same col has mult op ")
        dataset['op_dataset'].append({
            "question_tokens": question_tokens,
            "ts": table_schema,
            "history": op_col_dict[key][0],
            "label": op_col_dict[key][1]
        })
    agg_col_dict = dict()
    for h, sql_item in agg_candidates:
        _, label = AggPredictor(question_tokens, sql_item, h).generate_output()
        if label != 5:
            key = "{}{}".format(h[-2], h[-1][2])
            if key in agg_col_dict:
                agg_col_dict[key][1].append(label)
            else:
                agg_col_dict[key] = [h[:], [label]]
    for key in agg_col_dict:
        # if 5 in agg_col_dict[key][1]:
        #     print("none in agg label!!!")
        dataset['agg_dataset'].append({
            "question_tokens": question_tokens,
            "ts": table_schema,
            "history": agg_col_dict[key][0],
            "label": agg_col_dict[key][1]
        })


def get_table_dict(table_data_path):
    data = json.load(open(table_data_path))
    table = dict()
    for item in data:
        table[item["db_id"]] = item
    return table


def parse_data(data, table_path, gen_data_path, history_option, part):
    dataset = {
        "multi_sql_dataset": [],
        "keyword_dataset": [],
        "col_dataset": [],
        "op_dataset": [],
        "agg_dataset": [],
        "root_tem_dataset": [],
        "des_asc_dataset": [],
        "having_dataset": [],
        "andor_dataset": []
    }
    table_dict = get_table_dict(table_path)
    for item in data:
        if history_option == "full":
            # parser_item(item["question_toks"], item["sql"], table_dict[item["db_id"]], [], dataset)
            parser_item_with_long_history(
                item["question_toks"], item["sql"], table_dict[item["db_id"]], [], dataset)
        else:
            parser_item(item["question_toks"], item["sql"],
                        table_dict[item["db_id"]], [], dataset)
    print("\nfinished preprocess %s part" % (part))
    for key in dataset:
        print("dataset:{} size:{}".format(key, len(dataset[key])))
        json.dump(
            dataset[key],
            open(
                os.path.join(
                    gen_data_path,
                    "{}_{}_{}.json".format(
                        history_option,
                        part,
                        key)),
                "w"),
            indent=2,
            encoding="UTF-8",
            ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s',
        '--schema',
        type=str,
        default='char',
        choices=[
            'char',
            'word'],
        help='char for char-based schema and word for word-based schema.')

    parser.add_argument(
        '--history_option',
        type=str,
        default='full',
        choices=[
            'full',
            'part',
            'no'],
        help='full, part, or no history')

    args = parser.parse_args()

    table_path = "./data/tables.json"

    schema = args.schema
    history_option = args.history_option
    train_dev_test = ['train', 'dev', 'test']
    for part in train_dev_test:
        data_path = os.path.join('data', args.schema, part + '.json')
        data = json.load(codecs.open(data_path, 'r', encoding='utf-8'))
        gen_data_path = os.path.join('data', args.schema, 'generated_datasets')
        if not os.path.exists(gen_data_path):
            os.mkdir(gen_data_path)
        parse_data(data, table_path, gen_data_path, history_option, part)
