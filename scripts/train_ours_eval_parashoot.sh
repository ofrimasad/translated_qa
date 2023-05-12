export BASEPATH=.
export PYTHONPATH=${BASEPATH}/src
export DATAPATH=${BASEPATH}/data
export CUDA_VISIBLE_DEVICES=0,1


python ${PYTHONPATH}/train/run_qa.py \
  --model_name_or_path=bert-base-multilingual-cased \
  --train_file=${DATAPATH}/squad_translated/train-v1.1-hf_iw.json \
  --validation_file=${DATAPATH}/squad_translated/dev-v1.1-hf_iw.json \
  --test_file ${DATAPATH}/parashoot/test.json \
  --output_dir=${BASEPATH}/exp/train_ours_eval_parashoot \
  --run_name=train_ours_eval_parashoot \
  --do_train \
  --do_eval \
  --do_predict \
  --doc_stride=128 \
  --max_seq_length=384 \
  --per_gpu_train_batch_size=4 \
  --gradient_accumulation_steps=8 \
  --learning_rate=3e-5 \
  --save_steps=-1 \
  --num_train_epochs=6.0 \
  --warmup_steps=500 \
  --evaluation_strategy steps \
  --weight_decay=0.0001 \
  --overwrite_output_dir






