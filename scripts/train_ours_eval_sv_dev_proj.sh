export BASEPATH=.
export PYTHONPATH=${BASEPATH}/src
export DATAPATH=${BASEPATH}/data
export CUDA_VISIBLE_DEVICES=0,1

python ${PYTHONPATH}/train/run_qa.py \
  --model_name_or_path bert-base-multilingual-cased \
  --train_file=${DATAPATH}/squad_translated/train-v1.1-hf_sv.json \
  --validation_file=${DATAPATH}/squad_translated/dev-v1.1-hf_sv.json \
  --test_file=${DATAPATH}/swedish/swedish_squad_dev_hf.json \
  --output_dir=${BASEPATH}/exp_xquad_new/train_ours_test_sv_dev_proj \
  --run_name=train_ours_test_sv_dev_proj \
  --do_train \
  --do_eval \
  --do_predict \
  --doc_stride=128 \
  --max_seq_length=384 \
  --per_gpu_train_batch_size=4 \
  --gradient_accumulation_steps=16 \
  --learning_rate=3e-5 \
  --save_steps=-1 \
  --num_train_epochs=3.0 \
  --warmup_steps=500 \
  --evaluation_strategy=steps \
  --weight_decay=0.0001 \
  --overwrite_output_dir



