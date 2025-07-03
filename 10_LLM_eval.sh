#!/bin/bash

for dataset in 'XQUAD_open'
do
	for lang in 'en' 'ar' 'de' 'el' 'es' 'hi' 'ro' 'ru' 'th' 'tr' 'vi' 'zh'
	# for lang in 'ar' 'el' 'es' 'hi' 'ro' 'ru' 'th' 'tr' 'vi' 'zh'
	do
		echo $lang
		python sample_instances.py --lang $lang --dataset $dataset --num 50
	done
done
