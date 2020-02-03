# CSpider: A Large-Scale Chinese Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task

CSpider is a large Chinese dataset for complex and cross-domain semantic parsing and text-to-SQL task (natural language interfaces for relational databases). It is released with our EMNLP 2019 paper: [A Pilot Study for Chinese SQL Semantic Parsing](https://arxiv.org/abs/1909.13293). This repo contains all code for evaluation, preprocessing, and all baselines used in our paper. Please refer to [the task site](https://taolusi.github.io/CSpider-explorer/) for more general introduction and the leaderboard.

### Changelog
- `10/2019` We start a Chinese text-to-SQL task with the full dataset translated from [Spider](https://yale-lily.github.io/spider). The submission tutorial and our dataset can be found at our [task site](https://taolusi.github.io/CSpider-explorer/). Please follow it to get your results on the unreleased test data. Thank [Tao Yu](https://taoyds.github.io/) for sharing the test set with us.
- `9/2019` The dataset used in our EMNLP 2019 paper is redivided based on the training and deveploment sets from Spider. The dataset can be downloaded from [here](https://drive.google.com/drive/folders/1SVAdUQqZ2UjjcSCSxhVXRPcXxIMu1r_C?usp=sharing). This dataset is just released to reproduce the results in our paper. To join the CSpider leaderboard and better compare with the original English results, please refer to our [task site](https://taolusi.github.io/CSpider-explorer/) for full dataset.

### Citation
When you use the CSpider dataset, we would appreciate it if you cite the following:
```
@inproceedings{min2019pilot,
  title={A Pilot Study for Chinese SQL Semantic Parsing},
  author={Min, Qingkai and Shi, Yuefeng and Zhang, Yue},
  booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
  pages={3643--3649},
  year={2019}
}
```
Our dataset is based on [Spider](https://github.com/taoyds/spider/), please cite it too.

### Baseline models

#### Environment Setup

1. The code uses Python 2.7 and [Pytorch 0.2.0](https://pytorch.org/get-started/previous-versions/) GPU, and will update python and Pytorch soon.
2. Install Pytorch via conda: `conda install pytorch=0.2.0 -c pytorch`
3. Install Python dependency: `pip install -r requirements.txt`

#### Prepare Data, Embeddings, and Pretrained Models
1. Download the data, embedding and database:
  - To use the full dataset(recommended), download train/dev data from [Google Drive](https://drive.google.com/drive/folders/1TxCUq1ydPuBdDdHF3MkHT-8zixluQuLa?usp=sharing) or [BaiduNetDisk](https://pan.baidu.com/s/1Dxj38wRbbTOe0t3mQ3qhMg) and evaluate on the unreleased test data based on the submission tutorial on our [task site](https://taolusi.github.io/CSpider-explorer/). Specifically, 
    - Put the downloaded `train.json` and `dev.json` under `chisp/data/char/` directory. To use word-based methods, please do the word segmentation first and put the json files under `chisp/data/word/` directory.
    - Put the downloaded `char_emb.txt` under `chisp/embedding/` directory. This is generated from the Tencent multilingual embeddings for the cross-lingual word embeddings schema. To use monolingual embedding schema, step 2 is necessary.
    - Put the downloaded `database` directory under `chisp/` directory.
    - Put the downloaded `train_gold.sql` and `dev_glod.sql` under `chisp/data/` directory.
  - To use the dataset redivided based on the original train and dev data in our paper, download the train/dev/test data from [here](https://drive.google.com/drive/folders/1SVAdUQqZ2UjjcSCSxhVXRPcXxIMu1r_C?usp=sharing). This dataset is released just to reproduce the results in our paper and results based on this dataset cannot join the leaderboard. Specifically,
    - Put the downloaded `data`, `database` and `embedding` directory under `chisp/` directory. And you can run all the experiments(step 2 is necessary) shown in our paper.
    - `models` directory contains all the pretrained models.
2. (optional) Download the pretrained [Glove](https://nlp.stanford.edu/data/wordvecs/glove.42B.300d.zip), and put it as `chisp/embedding/glove.%dB.%dd.txt`
3. Generate training files for each module: `python preprocess_data.py -s char|word`

#### Folder/File Description
- ``data/`` contains:
    - ``char/`` for character-based raw train/dev/test data, corresponding processed dataset and saved models can be found at ``char/generated_datasets``. 
    - ``word/`` for word-based raw train/dev/test data, corresponding processed dataset and saved models can be found at ``word/generated_datasets``.
- ``train.py`` is the main file for training. Use ``train_all.sh`` to train all the modules (see below).
- ``test.py`` is the main file for testing. It uses ``supermodel.py`` to call the trained modules and generate SQL queries. In practice, use ``test_gen.sh`` to generate SQL queries.
- ``evaluation.py`` is for evaluation. It uses ``process_sql.py``. In practice, use ``evaluation.sh`` to evaluate the generated SQL queries.


#### Training
Run ``train_all.sh`` to train all the modules.
It looks like:
```
python train.py \
    --data_root       path/to/char/or/word/based/generated_data \
    --save_dir        path/to/save/trained/module \
    --train_component <module_name> \
    --emb_path        path/to/embeddings 
    --col_emb_path    path/to/corresponding/embeddings/for/column
```

#### Testing
Run ``test_gen.sh`` to generate SQL queries.
``test_gen.sh`` looks like:
```
python test.py \
    --test_data_path  path/to/char/or/word/based/raw/dev/or/test/data \
    --models          path/to/trained/module \
    --output_path     path/to/print/generated/SQL \
    --emb_path        path/to/embeddings 
    --col_emb_path    path/to/corresponding/embeddings/for/column
```

#### Evaluation
Run ``evaluation.sh`` to evaluate generated SQL queries.
``evaluation.sh`` looks like:
```
python evaluation.py \
    --gold            path/to/gold/dev/or/test/queries \ 
    --pred            path/to/predicted/dev/or/test/queries \
    --etype           evaluation/metric \
    --db              path/to/database \
    --table           path/to/tables \
```
``evalution.py`` is from the general evaluation process in [the Spider github page](https://github.com/taoyds/spider).

#### Acknowledgement

The implementation is based on [SyntaxSQLNet](https://github.com/taoyds/syntaxSQL). Please cite it too if you use this code.
