#!/bin/bash
for lang in 'ar' 'bn' 'de' 'en' 'es' 'fr' 'hi' 'id' 'it' 'ja' 'ko' 'pt' 'sw' 'yo' 'zh' \
	'cs' 'fa' 'tr' 'ro' 'si' 'am' 'te' 'uk' 'vi' 'ru' 'ms' \
	'el' 'fil' 'ha' 'he' 'ig' 'ky' 'lt' 'mg' 'ne' 'nl' 'ny' 'pl' 'sn' 'so' 'sr' 'sv'
do
	python retrieval_inlang_GMMLU.py --lang $lang
done
python fix_data_GMMLU.py # Output the missing values in GMMLU due to poor dataset quality



# Below part for cross-lingual retrieval on GMMLU
#for lang in 'en' 'ar' 'zh' 'si' 'yo'
#do
	#python retrieval_cross_GMMLU.py --lang $lang
#done
#python fix_data_cross_GMMLU.py
