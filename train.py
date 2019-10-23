import argparse
import datetime
import torch
from config import global_config as cfg
from utils.util import *
from utils.word_embedding import WordEmbedding
from models.agg_predictor import AggPredictor
from models.col_predictor import ColPredictor
from models.desasc_limit_predictor import DesAscLimitPredictor
from models.having_predictor import HavingPredictor
from models.keyword_predictor import KeyWordPredictor
from models.multisql_predictor import MultiSqlPredictor
from models.op_predictor import OpPredictor
from models.root_teminal_predictor import RootTeminalPredictor
from models.andor_predictor import AndOrPredictor


TRAIN_COMPONENTS = (
    'multi_sql',
    'keyword',
    'col',
    'op',
    'agg',
    'root_tem',
    'des_asc',
    'having',
    'andor')
SQL_TOK = ['<UNK>', '<END>', 'WHERE', 'AND', 'EQL', 'GT', 'LT', '<BEG>']


def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr * ((1 - decay_rate)**epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


if __name__ == '__main__':
    np.random.seed(100)
    torch.manual_seed(100)
    torch.cuda.manual_seed(100)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    N_word = cfg.emb_size
    N_col = cfg.col_emb_size
    N_h = cfg.hidden_size
    N_depth = cfg.num_layers
    dropout = cfg.dropout
    BATCH_SIZE = cfg.batch_size
    learning_rate = cfg.learning_rate
    train_emb = cfg.train_emb
    epoch = cfg.epoch

    history_type = cfg.history_type
    table_type = cfg.table_type

    use_hs = True
    if history_type == "no":
        use_hs = False

    # default to use GPU, but have to check if GPU exists
    GPU = True
    if not cfg.nogpu:
        if torch.cuda.device_count() == 0:
            GPU = False

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_root',
        type=str,
        default='',
        help='root path for generated_data')
    parser.add_argument(
        '--save_dir',
        type=str,
        default='',
        help='set model save directory.')
    parser.add_argument(
        '--train_component',
        type=str,
        default='',
        help='set train components,available:[multi_sql,keyword,col,op,agg,root_tem,des_asc,having,andor].')
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

    toy = args.toy
    if toy:
        USE_SMALL = True
    else:
        USE_SMALL = False

    data_root = args.data_root
    save_dir = args.save_dir

    train_component = args.train_component
    if train_component not in TRAIN_COMPONENTS:
        print("Invalid train component")
        exit(1)
    train_data = load_dataset(
        train_component,
        "train",
        history_type,
        data_root)
    dev_data = load_dataset(
        train_component,
        "dev",
        history_type,
        data_root)

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

    embed_layer = WordEmbedding(
        word_emb,
        N_word,
        gpu=GPU,
        SQL_TOK=SQL_TOK,
        trainable=train_emb)
    q_embed_layer = embed_layer

    if not col_emb:
        N_col = None
    else:
        embed_layer = WordEmbedding(
            col_emb, N_col, gpu=GPU, SQL_TOK=SQL_TOK, trainable=train_emb)

    model = None
    if train_component == "multi_sql":
        model = MultiSqlPredictor(
            N_word=N_word,
            N_col=N_col,
            N_h=N_h,
            N_depth=N_depth,
            gpu=GPU,
            dropout=dropout,
            use_hs=use_hs)
    elif train_component == "keyword":
        model = KeyWordPredictor(
            N_word=N_word,
            N_col=N_col,
            N_h=N_h,
            N_depth=N_depth,
            dropout=dropout,
            gpu=GPU,
            use_hs=use_hs)
    elif train_component == "col":
        model = ColPredictor(
            N_word=N_word,
            N_col=N_col,
            N_h=N_h,
            N_depth=N_depth,
            dropout=dropout,
            gpu=GPU,
            use_hs=use_hs)
    elif train_component == "op":
        model = OpPredictor(
            N_word=N_word,
            N_col=N_col,
            N_h=N_h,
            N_depth=N_depth,
            dropout=dropout,
            gpu=GPU,
            use_hs=use_hs)
    elif train_component == "agg":
        model = AggPredictor(
            N_word=N_word,
            N_col=N_col,
            N_h=N_h,
            N_depth=N_depth,
            dropout=dropout,
            gpu=GPU,
            use_hs=use_hs)
    elif train_component == "root_tem":
        model = RootTeminalPredictor(
            N_word=N_word,
            N_col=N_col,
            N_h=N_h,
            N_depth=N_depth,
            dropout=dropout,
            gpu=GPU,
            use_hs=use_hs)
    elif train_component == "des_asc":
        model = DesAscLimitPredictor(
            N_word=N_word,
            N_col=N_col,
            N_h=N_h,
            N_depth=N_depth,
            dropout=dropout,
            gpu=GPU,
            use_hs=use_hs)
    elif train_component == "having":
        model = HavingPredictor(
            N_word=N_word,
            N_col=N_col,
            N_h=N_h,
            N_depth=N_depth,
            dropout=dropout,
            gpu=GPU,
            use_hs=use_hs)
    elif train_component == "andor":
        model = AndOrPredictor(
            N_word=N_word,
            N_col=N_col,
            N_h=N_h,
            N_depth=N_depth,
            dropout=dropout,
            gpu=GPU,
            use_hs=use_hs)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0)
    print("Finished build model")

    print_flag = False

    print("Start training")
    best_acc = 0.0
    for i in range(epoch):
        print('Epoch %d @ %s' % (i + 1, datetime.datetime.now()))
        print(
            ' Loss = %s' %
            epoch_train(
                model,
                optimizer,
                BATCH_SIZE,
                train_component,
                train_data,
                table_type,
                q_embed_layer,
                embed_layer))
        acc = epoch_acc(
            model,
            BATCH_SIZE,
            train_component,
            dev_data,
            table_type,
            q_embed_layer,
            embed_layer)
        if acc > best_acc:
            best_acc = acc
            print("Save model...")
            torch.save(
                model.state_dict(),
                save_dir + "/{}_models.dump".format(
                    train_component))
