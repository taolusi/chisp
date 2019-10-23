import re
import io
import json
import numpy as np
import os
import signal


def get_table_dict(table_data_path):
    data = json.load(open(table_data_path))
    table = dict()
    for item in data:
        table[item["db_id"]] = item
    return table


def load_dataset(component, train_dev, history, root):
    return json.load(
        open("{}/{}_{}_{}_dataset.json".format(root, history, train_dev, component)))


def to_batch_seq(data, idxes, st, ed):
    q_seq = []
    history = []
    label = []
    for i in range(st, ed):
        q_seq.append(data[idxes[i]]['question_tokens'])
        history.append(data[idxes[i]]["history"])
        label.append(data[idxes[i]]["label"])
    return q_seq, history, label

# CHANGED


def to_batch_tables(data, idxes, st, ed, table_type):
    # col_lens = []
    col_seq = []
    for i in range(st, ed):
        ts = data[idxes[i]]["ts"]
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
        col_seq.append(cols_add)

    return col_seq

# used for training in train.py


def epoch_train(
        model,
        optimizer,
        batch_size,
        component,
        data,
        table_type,
        q_embed_layer,
        embed_layer):
    model.train()
    perm = np.random.permutation(len(data))
    cum_loss = 0.0
    st = 0

    while st < len(data):
        ed = st + batch_size if st + batch_size < len(perm) else len(perm)
        q_seq, history, label = to_batch_seq(data, perm, st, ed)
        q_emb_var, q_len = q_embed_layer.gen_x_q_batch(q_seq)
        hs_emb_var, hs_len = embed_layer.gen_x_history_batch(history)
        score = 0.0
        loss = 0.0
        if component == "multi_sql":
            mkw_emb_var = embed_layer.gen_word_list_embedding(
                ["none", "except", "intersect", "union"], (ed - st))
            mkw_len = np.full(q_len.shape, 4, dtype=np.int64)
            # print("mkw_emb:{}".format(mkw_emb_var.size()))
            score = model.forward(
                q_emb_var,
                q_len,
                hs_emb_var,
                hs_len,
                mkw_emb_var=mkw_emb_var,
                mkw_len=mkw_len)
        elif component == "keyword":
            # where group by order by
            # [[0,1,2]]
            kw_emb_var = embed_layer.gen_word_list_embedding(
                ["where", "group by", "order by"], (ed - st))
            mkw_len = np.full(q_len.shape, 3, dtype=np.int64)
            score = model.forward(
                q_emb_var,
                q_len,
                hs_emb_var,
                hs_len,
                kw_emb_var=kw_emb_var,
                kw_len=mkw_len)
        elif component == "col":
            # col word embedding
            # [[0,1,3]]
            col_seq = to_batch_tables(data, perm, st, ed, table_type)
            col_emb_var, col_name_len, col_len = embed_layer.gen_col_batch(
                col_seq)
            score = model.forward(
                q_emb_var,
                q_len,
                hs_emb_var,
                hs_len,
                col_emb_var,
                col_len,
                col_name_len)

        elif component == "op":
            # B*index
            gt_col = np.zeros(q_len.shape, dtype=np.int64)
            index = 0
            for i in range(st, ed):
                # print(i)
                gt_col[index] = data[perm[i]]["gt_col"]
                index += 1

            col_seq = to_batch_tables(data, perm, st, ed, table_type)
            col_emb_var, col_name_len, col_len = embed_layer.gen_col_batch(
                col_seq)
            score = model.forward(
                q_emb_var,
                q_len,
                hs_emb_var,
                hs_len,
                col_emb_var,
                col_len,
                col_name_len,
                gt_col=gt_col)

        elif component == "agg":
            # [[0,1,3]]
            col_seq = to_batch_tables(data, perm, st, ed, table_type)
            col_emb_var, col_name_len, col_len = embed_layer.gen_col_batch(
                col_seq)
            gt_col = np.zeros(q_len.shape, dtype=np.int64)
            # print(ed)
            index = 0
            for i in range(st, ed):
                # print(i)
                gt_col[index] = data[perm[i]]["gt_col"]
                index += 1
            score = model.forward(
                q_emb_var,
                q_len,
                hs_emb_var,
                hs_len,
                col_emb_var,
                col_len,
                col_name_len,
                gt_col=gt_col)

        elif component == "root_tem":
            # B*0/1
            col_seq = to_batch_tables(data, perm, st, ed, table_type)
            col_emb_var, col_name_len, col_len = embed_layer.gen_col_batch(
                col_seq)
            gt_col = np.zeros(q_len.shape, dtype=np.int64)
            # print(ed)
            index = 0
            for i in range(st, ed):
                # print(data[perm[i]]["history"])
                gt_col[index] = data[perm[i]]["gt_col"]
                index += 1
            score = model.forward(
                q_emb_var,
                q_len,
                hs_emb_var,
                hs_len,
                col_emb_var,
                col_len,
                col_name_len,
                gt_col=gt_col)

        elif component == "des_asc":
            # B*0/1
            col_seq = to_batch_tables(data, perm, st, ed, table_type)
            col_emb_var, col_name_len, col_len = embed_layer.gen_col_batch(
                col_seq)
            gt_col = np.zeros(q_len.shape, dtype=np.int64)
            # print(ed)
            index = 0
            for i in range(st, ed):
                # print(i)
                gt_col[index] = data[perm[i]]["gt_col"]
                index += 1
            score = model.forward(
                q_emb_var,
                q_len,
                hs_emb_var,
                hs_len,
                col_emb_var,
                col_len,
                col_name_len,
                gt_col=gt_col)

        elif component == 'having':
            col_seq = to_batch_tables(data, perm, st, ed, table_type)
            col_emb_var, col_name_len, col_len = embed_layer.gen_col_batch(
                col_seq)
            gt_col = np.zeros(q_len.shape, dtype=np.int64)
            # print(ed)
            index = 0
            for i in range(st, ed):
                # print(i)
                gt_col[index] = data[perm[i]]["gt_col"]
                index += 1
            score = model.forward(
                q_emb_var,
                q_len,
                hs_emb_var,
                hs_len,
                col_emb_var,
                col_len,
                col_name_len,
                gt_col=gt_col)

        elif component == "andor":
            score = model.forward(q_emb_var, q_len, hs_emb_var, hs_len)
        # score = model.forward(q_seq, col_seq, col_num, pred_entry,
        #         gt_where=gt_where_seq, gt_cond=gt_cond_seq, gt_sel=gt_sel_seq)
        # print("label {}".format(label))
        loss = model.loss(score, label)
        # print("loss {}".format(loss.data.cpu().numpy()))
        cum_loss += loss.data.cpu().numpy()[0] * (ed - st)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        st = ed

    return cum_loss / len(data)

# used for development evaluation in train.py


def epoch_acc(
        model,
        batch_size,
        component,
        data,
        table_type,
        q_embed_layer,
        embed_layer,
        error_print=False,
        train_flag=False):
    model.eval()
    perm = list(range(len(data)))
    st = 0
    total_number_error = 0.0
    total_p_error = 0.0
    total_error = 0.0
    print("dev data size {}".format(len(data)))
    while st < len(data):
        ed = st + batch_size if st + batch_size < len(perm) else len(perm)

        q_seq, history, label = to_batch_seq(data, perm, st, ed)
        q_emb_var, q_len = q_embed_layer.gen_x_q_batch(q_seq)
        hs_emb_var, hs_len = embed_layer.gen_x_history_batch(history)
        score = 0.0

        if component == "multi_sql":
            # none, except, intersect,union
            # truth B*index(0,1,2,3)
            # print("hs_len:{}".format(hs_len))
            # print("q_emb_shape:{} hs_emb_shape:{}".format(q_emb_var.size(), hs_emb_var.size()))
            mkw_emb_var = embed_layer.gen_word_list_embedding(
                ["none", "except", "intersect", "union"], (ed - st))
            mkw_len = np.full(q_len.shape, 4, dtype=np.int64)
            # print("mkw_emb:{}".format(mkw_emb_var.size()))
            score = model.forward(
                q_emb_var,
                q_len,
                hs_emb_var,
                hs_len,
                mkw_emb_var=mkw_emb_var,
                mkw_len=mkw_len)
        elif component == "keyword":
            # where group by order by
            # [[0,1,2]]
            kw_emb_var = embed_layer.gen_word_list_embedding(
                ["where", "group by", "order by"], (ed - st))
            mkw_len = np.full(q_len.shape, 3, dtype=np.int64)
            score = model.forward(
                q_emb_var,
                q_len,
                hs_emb_var,
                hs_len,
                kw_emb_var=kw_emb_var,
                kw_len=mkw_len)
        elif component == "col":
            # col word embedding
            # [[0,1,3]]
            col_seq = to_batch_tables(data, perm, st, ed, table_type)
            col_emb_var, col_name_len, col_len = embed_layer.gen_col_batch(
                col_seq)
            score = model.forward(
                q_emb_var,
                q_len,
                hs_emb_var,
                hs_len,
                col_emb_var,
                col_len,
                col_name_len)
        elif component == "op":
            # B*index
            col_seq = to_batch_tables(data, perm, st, ed, table_type)
            col_emb_var, col_name_len, col_len = embed_layer.gen_col_batch(
                col_seq)
            gt_col = np.zeros(q_len.shape, dtype=np.int64)
            # print(ed)
            index = 0
            for i in range(st, ed):
                # print(i)
                gt_col[index] = data[perm[i]]["gt_col"]
                index += 1
            score = model.forward(
                q_emb_var,
                q_len,
                hs_emb_var,
                hs_len,
                col_emb_var,
                col_len,
                col_name_len,
                gt_col=gt_col)

        elif component == "agg":
            # [[0,1,3]]
            col_seq = to_batch_tables(data, perm, st, ed, table_type)
            col_emb_var, col_name_len, col_len = embed_layer.gen_col_batch(
                col_seq)
            gt_col = np.zeros(q_len.shape, dtype=np.int64)
            # print(ed)
            index = 0
            for i in range(st, ed):
                # print(i)
                gt_col[index] = data[perm[i]]["gt_col"]
                index += 1

            score = model.forward(
                q_emb_var,
                q_len,
                hs_emb_var,
                hs_len,
                col_emb_var,
                col_len,
                col_name_len,
                gt_col=gt_col)

        elif component == "root_tem":
            # B*0/1
            col_seq = to_batch_tables(data, perm, st, ed, table_type)
            col_emb_var, col_name_len, col_len = embed_layer.gen_col_batch(
                col_seq)
            gt_col = np.zeros(q_len.shape, dtype=np.int64)
            # print(ed)
            index = 0
            for i in range(st, ed):
                # print(data[perm[i]]["history"])
                gt_col[index] = data[perm[i]]["gt_col"]
                index += 1
            score = model.forward(
                q_emb_var,
                q_len,
                hs_emb_var,
                hs_len,
                col_emb_var,
                col_len,
                col_name_len,
                gt_col=gt_col)

        elif component == "des_asc":
            # B*0/1
            col_seq = to_batch_tables(data, perm, st, ed, table_type)
            col_emb_var, col_name_len, col_len = embed_layer.gen_col_batch(
                col_seq)
            gt_col = np.zeros(q_len.shape, dtype=np.int64)
            # print(ed)
            index = 0
            for i in range(st, ed):
                # print(i)
                gt_col[index] = data[perm[i]]["gt_col"]
                index += 1
            score = model.forward(
                q_emb_var,
                q_len,
                hs_emb_var,
                hs_len,
                col_emb_var,
                col_len,
                col_name_len,
                gt_col=gt_col)

        elif component == 'having':
            col_seq = to_batch_tables(data, perm, st, ed, table_type)
            col_emb_var, col_name_len, col_len = embed_layer.gen_col_batch(
                col_seq)
            gt_col = np.zeros(q_len.shape, dtype=np.int64)
            # print(ed)
            index = 0
            for i in range(st, ed):
                # print(i)
                gt_col[index] = data[perm[i]]["gt_col"]
                index += 1
            score = model.forward(
                q_emb_var,
                q_len,
                hs_emb_var,
                hs_len,
                col_emb_var,
                col_len,
                col_name_len,
                gt_col=gt_col)

        elif component == "andor":
            score = model.forward(q_emb_var, q_len, hs_emb_var, hs_len)
        # print("label {}".format(label))
        if component in ("agg", "col", "keyword", "op"):
            num_err, p_err, err = model.check_acc(score, label)
            total_number_error += num_err
            total_p_error += p_err
            total_error += err
        else:
            err = model.check_acc(score, label)
            total_error += err
        st = ed

    if component in ("agg", "col", "keyword", "op"):
        print(
            "Dev {} acc number predict acc:{} partial acc: {} total acc: {}".format(
                component,
                1 -
                total_number_error *
                1.0 /
                len(data),
                1 -
                total_p_error *
                1.0 /
                len(data),
                1 -
                total_error *
                1.0 /
                len(data)))
        return 1 - total_error * 1.0 / len(data)
    else:
        print(
            "Dev {} acc total acc: {}".format(
                component,
                1 -
                total_error *
                1.0 /
                len(data)))
        return 1 - total_error * 1.0 / len(data)


def timeout_handler(num, stack):
    print("Received SIGALRM")
    raise Exception("Timeout")

## used in test.py


def test_acc(model, batch_size, data, output_path):
    table_dict = get_table_dict("./data/tables.json")
    f = open(output_path, "w")
    for item in data[:]:
        db_id = item["db_id"]
        if db_id not in table_dict:
            print "Error %s not in table_dict" % db_id
        # signal.signal(signal.SIGALRM, timeout_handler)
        # signal.alarm(2) # set timer to prevent infinite recursion in SQL
        # generation
        sql = model.forward([item["question_toks"]] *
                            batch_size, [], table_dict[db_id])
        if sql is not None:
            print(sql)
            sql = model.gen_sql(sql, table_dict[db_id])
        else:
            sql = "select a from b"
        print(sql)
        print("")
        f.write("{}\n".format(sql))
    f.close()


def load_emb(file_name, load_used=False, use_small=False):
    if not load_used:
        print('Loading word embedding from %s' % file_name)
        ret = {}
        with open(file_name) as inf:
            for idx, line in enumerate(inf):
                if (use_small and idx >= 5000):
                    break
                info = line.strip().split(' ')
                info[0] = info[0].decode('utf-8')
                if info[0].lower() not in ret:
                    ret[info[0]] = np.array(map(lambda x: float(x), info[1:]))
        return ret
    else:
        print('Load used word embedding')
        with open('../alt/glove/word2idx.json') as inf:
            w2i = json.load(inf)
        with open('../alt/glove/usedwordemb.npy') as inf:
            word_emb_val = np.load(inf)
        return w2i, word_emb_val
