#!/bin/bash

schema="char"  # char or word
#schema="word"

embedding="multi"  # multi for multi-lingual or mono for monolingual
#embedding="mono"

SAVE_PATH="data/${schema}/generated_datasets/saved_models_${embedding}"

# evaluation
python evaluation.py \
    --gold "data/test_gold.sql" \
    --pred "${SAVE_PATH}/test_result.txt" \
    --etype "match" \
    --db "database" \
    --table "data/tables.json" \
   > "result_${schema}_${embedding}.log" \
   2>&1 &
