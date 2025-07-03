#!/bin/bash

for dataset in 'XQUAD_open'
do
	for lang in 'en' 'ar' 'de' 'el' 'es' 'hi' 'ro' 'ru' 'th' 'tr' 'vi' 'zh'
	do
		echo $lang
		python check_LLM_eval.py --lang $lang --dataset $dataset
	done
done
