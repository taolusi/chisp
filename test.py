import torch
import argparse
from utils.util import *
from models.supermodel import SuperModel

from config import global_config as cfg

if __name__ == '__main__':
    N_word = cfg.emb_size
    N_col = cfg.col_emb_size
    N_h = cfg.hidden_size
    N_depth = cfg.num_layers
    dropout = cfg.dropout
    BATCH_SIZE = cfg.batch_size
    learning_rate = cfg.learning_rate
    train_emb = cfg.train_emb

    history_type = cfg.history_type
    table_type = cfg.table_type

    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=str, help='path to saved model')
    parser.add_argument('--test_data_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument(
        '--emb_path',
        type=str,
        default='',
        help='embedding path, multi-lingual or monolingual')
    parser.add_argument(
        '--col_emb_path',
        type=str,
        default='',
        help='column embedding path')
    parser.add_argument(
        '--toy',
        action='store_true',
        help='If set, use small data; used for fast debugging.')
    args = parser.parse_args()

    use_hs = True
    if history_type == "no":
        use_hs = False

    # default to use GPU, but have to check if GPU exists
    GPU = True
    if not cfg.nogpu:
        if torch.cuda.device_count() == 0:
            GPU = False

    toy = args.toy
    if toy:
        USE_SMALL = True
    else:
        USE_SMALL = False

    data = json.load(open(args.test_data_path))

    emb_path = args.emb_path
    word_emb = load_emb(emb_path, load_used=train_emb, use_small=USE_SMALL)
    col_emb_path = args.col_emb_path
    col_emb = None
    if col_emb_path != 'None':
        col_emb = load_emb(
            col_emb_path,
            load_used=train_emb,
            use_small=USE_SMALL)
    print("Finished load word embedding")

    model = SuperModel(
        word_emb,
        col_emb,
        N_word=N_word,
        N_col = N_col,
        N_h=N_h,
        N_depth=N_depth,
        dropout=dropout,
        gpu=GPU,
        trainable_emb=train_emb,
        table_type=table_type,
        use_hs=use_hs)

    print "Loading from modules..."
    model.multi_sql.load_state_dict(torch.load(
        "{}/multi_sql_models.dump".format(args.models)))
    model.key_word.load_state_dict(torch.load(
        "{}/keyword_models.dump".format(args.models)))
    model.col.load_state_dict(torch.load(
        "{}/col_models.dump".format(args.models)))
    model.op.load_state_dict(torch.load(
        "{}/op_models.dump".format(args.models)))
    model.agg.load_state_dict(torch.load(
        "{}/agg_models.dump".format(args.models)))
    model.root_teminal.load_state_dict(torch.load(
        "{}/root_tem_models.dump".format(args.models)))
    model.des_asc.load_state_dict(torch.load(
        "{}/des_asc_models.dump".format(args.models)))
    model.having.load_state_dict(torch.load(
        "{}/having_models.dump".format(args.models)))

    test_acc(model, BATCH_SIZE, data, args.output_path)
    # test_exec_acc()
