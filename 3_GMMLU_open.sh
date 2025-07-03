#!/bin/bash

for mname in "CohereForAI/aya-expanse-8b" "Qwen/Qwen2.5-7B-Instruct" "meta-llama/Llama-3.2-3B-Instruct" "google/gemma-2-9b-it"
do
	for lang in 'ar' 'bn' 'de' 'en' 'es' 'fr' 'hi' 'id' 'it' 'ja' 'ko' 'pt' 'sw' 'yo' 'zh' \
		'cs' 'fa' 'tr' 'ro' 'si' 'am' 'te' 'uk' 'vi' 'ru' 'ms' \
		'el' 'fil' 'ha' 'he' 'ig' 'ky' 'lt' 'mg' 'ne' 'nl' 'ny' 'pl' 'sn' 'so' 'sr' 'sv'
	do
                sbatch --time=6:00:00 --ntasks=4 --cpus-per-task=4 --mem=180G --partition=gpu_h100 --nodes=1 --gpus-per-node=1 --wrap="conda run -n RAGConsis CUDA_VISIBLE_DEVICES=0 python GMMLU_open.py --mname $mname --lang $lang" --output=log/GMMLU_open/$mname/$lang.%j.out              
	done
done


