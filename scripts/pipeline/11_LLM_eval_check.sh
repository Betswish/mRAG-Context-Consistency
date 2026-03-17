#!/bin/bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"

cd_repo_root
output_file="$REPO_ROOT/reports/checks/check_LLM_eval.txt"
ensure_dir "$(dirname "$output_file")"
: > "$output_file"

for dataset in XQUAD_open
do
    for lang in en ar de el es hi ro ru th tr vi zh
    do
        echo "$lang" | tee -a "$output_file"
        python pipeline/evaluation/check_LLM_eval.py --lang "$lang" --dataset "$dataset" | tee -a "$output_file"
    done
done
