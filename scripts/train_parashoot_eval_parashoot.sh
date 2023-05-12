export BASEPATH=.
export PYTHONPATH=${BASEPATH}/src
export DATAPATH=${BASEPATH}/data
export CUDA_VISIBLE_DEVICES=0,1

python ${PYTHONPATH}/train/run_qa.py \
  --model_name_or_path=bert-base-multilingual-cased \
  --train_file=${DATAPATH}/parashoot/train.json \
  --validation_file=${DATAPATH}/parashoot/dev.json \
  --test_file=${DATAPATH}/parashoot/test.json \
  --output_dir=${BASEPATH}/exp/train_parashoot_eval_parashoot \
  --run_name=train_parashoot_eval_parashoot \
  --do_train \
  --do_eval \
  --do_predict \
  --per_device_train_batch_size=8 \
  --per_device_eval_batch_size=4 \
  --gradient_accumulation_steps=8 \
  --learning_rate=1e-04 \
  --doc_stride=128 \
  --max_seq_length=512 \
  --logging_steps=5 \
  --seed=200 \
  --warmup_ratio=0.06 \
  --eval_steps=40 \
  --evaluation_strategy=steps \
  --overwrite_output_dir \
  --overwrite_cache \
  --num_train_epochs=5