#!/bin/bash

for mname in "CohereForAI/aya-expanse-8b" "Qwen/Qwen2.5-7B-Instruct" "meta-llama/Llama-3.2-3B-Instruct" "google/gemma-2-9b-it"
# for mname in "CohereForAI/aya-expanse-8b"
do
	# for lang in 'en'
	# for lang in ar da de es fi fr he hu it ja ko km ms nl no pl pt ru sv th tr vi zh
	# for lang in ar da de en es fi fr he hu it ja ko km ms nl no pl pt ru sv th tr vi zh
	do
                sbatch --time=6:00:00 --ntasks=4 --cpus-per-task=4 --mem=180G --partition=gpu_h100 --nodes=1 --gpus-per-node=1 --wrap="conda run -n RAGConsis CUDA_VISIBLE_DEVICES=0 python MKQA_open_multi.py --mname $mname --lang $lang --batch_size 3000" --output=log/MKQA_open_multi/$mname/$lang.%j.out --job-name=${mname:0:2}\_$lang            
	done
done


