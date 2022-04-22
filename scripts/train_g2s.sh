#!/bin/bash

LOAD_FROM=""
MODEL=g2s_series_rel
TASK=retrosynthesis
DATASET=USPTO_50k
MPN_TYPE=dgat
MAX_REL_POS=4
ACCUM_COUNT=4
ENC_PE=none
ENC_H=256
BATCH_SIZE=4096
ENC_EMB_SCALE=sqrt
MAX_STEP=500000
ENC_LAYER=4
BATCH_TYPE=tokens
REL_BUCKETS=11

EXP_NO=1
REL_POS=emb_only
ATTN_LAYER=6
LR=4
DROPOUT=0.3

REPR_START=smiles
REPR_END=smiles

PREFIX=${DATASET}_${MODEL}_${REPR_START}_${REPR_END}


python train.py \
  --model="$MODEL" \
  --data_name="$DATASET" \
  --task="$TASK" \
  --representation_end=$REPR_END \
  --load_from="$LOAD_FROM" \
  --train_bin="./preprocessed/$PREFIX/train_0.npz" \
  --valid_bin="./preprocessed/$PREFIX/val_0.npz" \
  --log_file="$PREFIX.train.$EXP_NO.log" \
  --vocab_file="./preprocessed/$PREFIX/vocab_$REPR_END.txt" \
  --save_dir="./checkpoints/$PREFIX.$EXP_NO" \
  --embed_size=256 \
  --mpn_type="$MPN_TYPE" \
  --encoder_num_layers="$ENC_LAYER" \
  --encoder_hidden_size="$ENC_H" \
  --encoder_norm="$ENC_NORM" \
  --encoder_skip_connection="$ENC_SC" \
  --encoder_positional_encoding="$ENC_PE" \
  --encoder_emb_scale="$ENC_EMB_SCALE" \
  --attn_enc_num_layers="$ATTN_LAYER" \
  --attn_enc_hidden_size=256 \
  --attn_enc_heads=8 \
  --attn_enc_filter_size=2048 \
  --rel_pos="$REL_POS" \
  --rel_pos_buckets="$REL_BUCKETS" \
  --decoder_num_layers=6 \
  --decoder_hidden_size=256 \
  --decoder_attn_heads=8 \
  --decoder_filter_size=2048 \
  --dropout="$DROPOUT" \
  --attn_dropout="$DROPOUT" \
  --max_relative_positions="$MAX_REL_POS" \
  --seed=42 \
  --epoch=2000 \
  --max_steps="$MAX_STEP" \
  --warmup_steps=8000 \
  --lr="$LR" \
  --weight_decay=0.0 \
  --clip_norm=20.0 \
  --batch_type="$BATCH_TYPE" \
  --train_batch_size="$BATCH_SIZE" \
  --valid_batch_size="$BATCH_SIZE" \
  --predict_batch_size="$BATCH_SIZE" \
  --accumulation_count="$ACCUM_COUNT" \
  --num_workers=0 \
  --beam_size=5 \
  --predict_min_len=1 \
  --predict_max_len=512 \
  --log_iter=100 \
  --eval_iter=2000 \
  --save_iter=5000 \
  --compute_graph_distance
