#!/bin/bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"

cd_repo_root

for lang in ar bn de en es fr hi id it ja ko pt sw yo zh \
    cs fa tr ro si am te uk vi ru ms \
    el fil ha he ig ky lt mg ne nl ny pl sn so sr sv
do
    python pipeline/retrieval/retrieval_inlang_GMMLU.py --lang "$lang"
done
python pipeline/retrieval/fix_data_GMMLU.py # Output the missing values in GMMLU due to poor dataset quality



# Below part for cross-lingual retrieval on GMMLU
#for lang in 'en' 'ar' 'zh' 'si' 'yo'
#do
    #python pipeline/retrieval/retrieval_cross_GMMLU.py --lang "$lang"
#done
#python pipeline/retrieval/fix_data_cross_GMMLU.py
