#!/bin/bash

for lang in "ar" "de" "el" "es" "hi" "ru" "th" "tr" "vi" "zh" "cs" "fa" "fr" "iw" "nl" "ru" "sv"
do
  wget  https://translated-qa.s3.amazonaws.com/dev-v1.1-hf_${lang}.json -P $( dirname -- "$0"; )
  wget  https://translated-qa.s3.amazonaws.com/train-v1.1-hf_${lang}.json -P $( dirname -- "$0"; )
done