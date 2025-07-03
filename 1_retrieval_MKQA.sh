#!/bin/bash

for lang in ar da de en es fi fr he hu it ja ko km ms nl no pl pt ru sv th tr vi zh
do
	sbatch --time=2-00:00:00 --ntasks=4 --cpus-per-task=4 --mem=180G --partition=gpu_h100 --nodes=1 --gpus-per-node=1 --wrap="conda run -n RAGConsis CUDA_VISIBLE_DEVICES=0 python retrieval_inlang_MKQA.py --lang $lang" --output=log/MKQA/$lang.%j.out
done


# Below is used for cross-lingual retrieval on MKQA 
#for lang in 'en' 'ar' 'zh' 'ms' 'km'
#do
	#sbatch --time=3-00:00:00 --ntasks=4 --cpus-per-task=4 --mem=180G --partition=gpu_h100 --nodes=1 --gpus-per-node=1 --wrap="conda run -n RAGConsis CUDA_VISIBLE_DEVICES=0 python retrieval_cross_MKQA.py --lang $lang" --output=log/MKQA-cross/$lang.%j.out
#done

