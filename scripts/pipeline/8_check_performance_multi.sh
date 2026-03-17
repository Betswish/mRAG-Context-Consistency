#!/bin/bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"

cd_repo_root
mkqa_output="$REPO_ROOT/reports/checks/check_MKQA_multi.txt"
gmmlu_output="$REPO_ROOT/reports/checks/check_GMMLU_choice_multi.txt"
ensure_dir "$(dirname "$mkqa_output")"
: > "$mkqa_output"
: > "$gmmlu_output"

python pipeline/evaluation/check_MKQA_multi.py | tee -a "$mkqa_output"
python pipeline/evaluation/check_GMMLU_choice_multi.py | tee -a "$gmmlu_output"
