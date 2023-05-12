export BASEPATH=.
export PYTHONPATH=${BASEPATH}/src
export DATAPATH=${BASEPATH}/data
export CUDA_VISIBLE_DEVICES=0,1

for lang in "ar" "de" "el" "es" "hi" "ru" "th" "tr" "vi" "zh"
do
  python ${PYTHONPATH}/train/run_qa.py \
    --model_name_or_path=bert-base-multilingual-cased \
    --train_file=${DATAPATH}/xquad_Translated_train/squad.translate.train.en-${lang}-hf.json \
    --validation_file=${DATAPATH}/xquad/xquad.${lang}-hf.json \
    --test_file=${DATAPATH}/xquad/xquad.${lang}-hf.json \
    --output_dir=${BASEPATH}/exp/train_xquad_test_xquad_${lang} \
    --run_name=train_xquad_test_xquad_${lang} \
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
done


