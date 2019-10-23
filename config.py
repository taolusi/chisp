class _Config:
    def __init__(self):
        self.emb_size = 200
        self.col_emb_size = 300
        self.hidden_size = 400
        self.batch_size = 10
        self.epoch = 600
        self.dropout = 0.5
        self.num_layers = 2
        self.learning_rate = 1e-4
        self.toy = False
        self.train_emb = False
        self.history_type = 'full'  # full, part or no
        self.nogpu = False
        self.table_type = 'std' # choices=['std','no'], help='standard, hierarchical, or no table info'

    def _char_init(self):
        self.data_root = "./data/char/generated_datasets"
        self.sep_emb = "./embedding/char/separate_emb.txt"
        self.comb_emb = "./embedding/char/combine_emb.txt"
    def _word_init(self):
        self.data_root = "./data/word/generated_datasets"
        self.sep_emb = "./embedding/word/separate_emb.txt"
        self.comb_emb = "./embedding/word/combine_emb.txt"

global_config = _Config()
