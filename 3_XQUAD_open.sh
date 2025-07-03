#!/bin/bash

for mname in "CohereForAI/aya-expanse-8b" "Qwen/Qwen2.5-7B-Instruct" "meta-llama/Llama-3.2-3B-Instruct" "google/gemma-2-9b-it"
do
	for lang in 'en' 'ar' 'de' 'el' 'es' 'hi' 'ro' 'ru' 'th' 'tr' 'vi' 'zh'
	do
                sbatch --time=6:00:00 --ntasks=4 --cpus-per-task=4 --mem=180G --partition=gpu_h100 --nodes=1 --gpus-per-node=1 --wrap="conda run -n RAGConsis CUDA_VISIBLE_DEVICES=0 python XQUAD_open.py --mname $mname --lang $lang" --output=log/XQUAD_open/$mname/$lang.%j.out              
	done
done


