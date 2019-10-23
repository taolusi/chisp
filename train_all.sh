#!/bin/bash

DATE=`date '+%Y-%m-%d-%H:%M:%S'`

schema="char"  # char or word
#schema="word"

toy=""

embedding="multi"  # multi for multi-lingual or mono for monolingual
#embedding="mono"

data_root="data/${schema}/generated_datasets"
emb_path="embedding/${schema}_emb.txt"
col_emb_path="embedding/glove.42B.300d.txt"
if [[ ${embedding} == "multi" ]]; then col_emb_path="None"; fi
save_dir="${data_root}/saved_models_${embedding}_${DATE}"
log_dir="${save_dir}/train_log"
mkdir -p ${save_dir}
mkdir -p ${log_dir}

export CUDA_VISIBLE_DEVICES=0
for module in col
do
  nohup python -u train.py \
    --data_root    ${data_root} \
    --save_dir     ${save_dir} \
    --train_component ${module} \
    --emb_path    ${emb_path} \
    --col_emb_path    ${col_emb_path} \
    ${toy} \
    > "${log_dir}/train_${module}_${DATE}.txt" \
    2>&1 &
done

export CUDA_VISIBLE_DEVICES=1
for module in keyword op des_asc multi_sql agg having root_tem andor
do
  nohup python -u train.py \
    --data_root    ${data_root} \
    --save_dir     ${save_dir} \
    --train_component ${module} \
    --emb_path    ${emb_path} \
    --col_emb_path    ${col_emb_path} \
    ${toy} \
    > "${log_dir}/train_${module}_${DATE}.txt" \
    2>&1 &
done

