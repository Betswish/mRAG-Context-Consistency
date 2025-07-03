#!/bin/bash

#for mname in "CohereForAI/aya-expanse-8b" "Qwen/Qwen2.5-7B-Instruct" "meta-llama/Llama-3.2-3B-Instruct" "google/gemma-2-9b-it"
for mname in "CohereForAI/aya-expanse-8b"
do
	for lang in 'en' 'ar' 'de' 'el' 'es' 'hi' 'ro' 'ru' 'th' 'tr' 'vi' 'zh'
	#for lang in 'el'
	do
		sbatch --time=2-00:00:00 --ntasks=4 --cpus-per-task=4 --mem=180G --partition=gpu_h100 --nodes=1 --gpus-per-node=1 --wrap="conda run -n RAGConsis CUDA_VISIBLE_DEVICES=0 python attribute_XQUAD_open.py --mname $mname --lang $lang" --output=log_attri/XQUAD_open/$mname/$lang.%j.out
	done
done




