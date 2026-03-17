#!/bin/bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"

cd_repo_root
output_file="$REPO_ROOT/reports/checks/check_performance.txt"
ensure_dir "$(dirname "$output_file")"
: > "$output_file"

python pipeline/evaluation/check_XQUAD.py | tee -a "$output_file"
python pipeline/evaluation/check_MKQA.py | tee -a "$output_file"
python pipeline/evaluation/check_GMMLU_open.py | tee -a "$output_file"
python pipeline/evaluation/check_GMMLU_choice.py | tee -a "$output_file"
