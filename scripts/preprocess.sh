#!/bin/bash

DATASET=USPTO_50k
MODEL=g2s_series_rel
TASK=retrosynthesis
REPR_START=smiles
REPR_END=smiles
N_WORKERS=8

PREFIX=${DATASET}_${MODEL}_${REPR_START}_${REPR_END}

python preprocess.py \
  --model="$MODEL" \
  --data_name="$DATASET" \
  --task="$TASK" \
  --representation_start=$REPR_START \
  --representation_end=$REPR_END \
  --train_src="./data/$DATASET/src-train.txt" \
  --train_tgt="./data/$DATASET/tgt-train.txt" \
  --val_src="./data/$DATASET/src-val.txt" \
  --val_tgt="./data/$DATASET/tgt-val.txt" \
  --test_src="./data/$DATASET/src-test.txt" \
  --test_tgt="./data/$DATASET/tgt-test.txt" \
  --log_file="$PREFIX.preprocess.log" \
  --preprocess_output_path="./preprocessed/$PREFIX/" \
  --seed=42 \
  --max_src_len=1024 \
  --max_tgt_len=1024 \
  --num_workers="$N_WORKERS"
