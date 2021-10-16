#!/bin/bash

export MODEL=g2s_series_rel
export TASK=retrosynthesis
export BS=30
export T=1.0
export NBEST=30

export EXP_NO=1
export DATASET=schneider50k
export CHECKPOINT=checkpoints/checkpoint_foo/model_bar.pt

export REPR_START=smiles
export REPR_END=smiles

PREFIX=${DATASET}_${MODEL}_${REPR_START}_${REPR_END}

python predict.py \
  --do_predict \
  --do_score \
  --model="$MODEL" \
  --test_bin="./preprocessed/$PREFIX/test_0.npz" \
  --test_tgt="./data/$DATASET/tgt-test.txt" \
  --result_file="./results/$PREFIX.$EXP_NO.result.txt" \
  --log_file="$PREFIX.predict.$EXP_NO.log" \
  --vocab_file="./preprocessed/$PREFIX/vocab_$REPR_END.txt" \
  --load_from="$CHECKPOINT" \
  --mpn_type="$MPN_TYPE" \
  --rel_pos="$REL_POS" \
  --seed=42 \
  --batch_type=tokens \
  --predict_batch_size=4096 \
  --beam_size="$BS" \
  --n_best="$NBEST" \
  --temperature="$T" \
  --predict_min_len=1 \
  --predict_max_len=512 \
  --log_iter=100
