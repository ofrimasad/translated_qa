# batch (total on all devices = 16)
export BASEPATH=/home/ofri/qa_translate
export PYTHONPATH=${BASEPATH}/src
export DATAPATH=${BASEPATH}/data
export CUDA_VISIBLE_DEVICES=2,3

python ${PYTHONPATH}/train/run_qa.py \
  --model_name_or_path bert-base-multilingual-cased \
  --train_file=${DATAPATH}/squad/enq_25_12_2022/train_v1.0hf_cs_0.05_enq_both.json \
  --validation_file=${DATAPATH}/czech-squad/squad-1.1-cs/dev-v1.1.json \
  --test_file=${DATAPATH}/czech-squad/squad-1.1-cs/dev-v1.1.json \
  --output_dir=${BASEPATH}/exp_xquad_new/train_squad_enq_25_12_2022_test_squad_cs \
  --run_name=train_squad_both_matcher_aug_test_xquad_cs \
  --do_train \
  --do_eval \
  --do_predict \
  --doc_stride=128 \
  --max_seq_length=384 \
  --per_gpu_train_batch_size=4 \
  --gradient_accumulation_steps=8 \
  --learning_rate=3e-5 \
  --save_steps=-1 \
  --num_train_epochs=3.0 \
  --warmup_steps=500 \
  --evaluation_strategy steps \
  --weight_decay=0.0001 \
  --overwrite_output_dir



