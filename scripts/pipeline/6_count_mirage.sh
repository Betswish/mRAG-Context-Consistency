#!/bin/bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"

cd_repo_root

python pipeline/attribution/count_mirage.py --task XQUAD_open --CTI 1
python pipeline/attribution/count_mirage.py --task MKQA_open --CTI 1

python pipeline/attribution/count_mirage.py --task XQUAD_open --CTI 1.5
python pipeline/attribution/count_mirage.py --task MKQA_open --CTI 1.5

python pipeline/attribution/count_mirage.py --task XQUAD_open --CTI 2
python pipeline/attribution/count_mirage.py --task MKQA_open --CTI 2

python pipeline/attribution/count_mirage.py --task XQUAD_open --CTI 2.5
python pipeline/attribution/count_mirage.py --task MKQA_open --CTI 2.5

python pipeline/attribution/count_mirage.py --task XQUAD_open --CTI 3
python pipeline/attribution/count_mirage.py --task MKQA_open --CTI 3
