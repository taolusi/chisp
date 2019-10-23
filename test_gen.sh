#!/bin/bash

#export CUDA_VISIBLE_DEVICES=1
schema="char"  # char or word
#schema="word"

toy=""

embedding="multi"  # multi for multi-lingual or mono for monolingual
#embedding="mono"

emb_path="embedding/${schema}_emb.txt"
col_emb_path="embedding/glove.42B.300d.txt"
if [[ ${embedding} == "multi" ]]; then col_emb_path="None"; fi

TEST_DATA="data/${schema}/test.json"

SAVE_PATH="data/${schema}/generated_datasets/saved_models_${embedding}"
python -u test.py \
   --test_data_path  ${TEST_DATA} \
   --models          ${SAVE_PATH} \
   --output_path     ${SAVE_PATH}/test_result.txt \
   --emb_path    ${emb_path} \
   --col_emb_path    ${col_emb_path} \
   ${toy} \
   > "${SAVE_PATH}/test_result.out.txt" \
   2>&1 &
