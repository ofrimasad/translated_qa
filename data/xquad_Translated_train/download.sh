#!/bin/bash

for lang in "ar" "de" "el" "es" "hi" "ru" "th" "tr" "vi" "zh"
do
  wget  https://translated-qa.s3.amazonaws.com/squad.translate.train.en-${lang}-hf.json --no-check-certificate -P $( dirname -- "$0"; )
done