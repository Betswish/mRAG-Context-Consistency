#!/bin/bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"

cd_repo_root
stats_output="$REPO_ROOT/reports/statistics/statistics_GMMLU.txt"
ensure_dir "$(dirname "$stats_output")"
: > "$stats_output"

python pipeline/filtering/filter_subset_inlang_GMMLU.py # Filter subset
python pipeline/filtering/filter_golddis_inlang_GMMLU.py # Split gold passages and distractors
python pipeline/filtering/statistics_GMMLU.py | tee -a "$stats_output" # Check statistics

# For filter the cross-lingual retrieval results, 
# which is *CORRESPONDING TO THE FILTERING RESULTS OF IN-LANG RETRIEVAL*
#python pipeline/filtering/filter_subset_cross_GMMLU.py
