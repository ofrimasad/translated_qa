#!/bin/bash
export BASEPATH=.
export PYTHONPATH=${BASEPATH}/src
export DATAPATH=${BASEPATH}/data
export CUDA_VISIBLE_DEVICES=0,1


python ${PYTHONPATH}/train/run_qa.py \
    --model_name_or_path=bert-base-multilingual-cased \
    --train_file=$1 \
    --validation_file=$2 \
    --test_file=$2 \
    --output_dir=${BASEPATH}/matcher_exp/train_matcher_$3 \
    --run_name=train_matcher_$3 \
    --do_train \
    --do_eval \
    --do_predict \
    --doc_stride=128 \
    --max_seq_length=384 \
    --per_gpu_train_batch_size=36 \
    --gradient_accumulation_steps=8 \
    --learning_rate=3e-5 \
    --num_train_epochs=4.0 \
    --warmup_steps=500 \
    --evaluation_strategy=steps \
    --weight_decay=0.0001 \
    --overwrite_output_dir \
    --version_2_with_negative \
    --metric_for_best_model=eval_exact \
    --save_total_limit=1 \
    --load_best_model_at_end=True
