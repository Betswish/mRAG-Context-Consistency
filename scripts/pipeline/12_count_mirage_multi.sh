#!/bin/bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"

cd_repo_root

python pipeline/attribution/count_mirage_multi.py --task MKQA_open_multi
python pipeline/attribution/count_mirage_multi.py --task GMMLU_choice_multi
