#!/bin/bash

python count_mirage.py --task XQUAD_open --thres 1
python count_mirage.py --task MKQA_open --thres 1

python count_mirage.py --task XQUAD_open --thres 1.5
python count_mirage.py --task MKQA_open --thres 1.5

python count_mirage.py --task XQUAD_open --thres 2
python count_mirage.py --task MKQA_open --thres 2

python count_mirage.py --task XQUAD_open --thres 2.5
python count_mirage.py --task MKQA_open --thres 2.5

python count_mirage.py --task XQUAD_open --thres 3
python count_mirage.py --task MKQA_open --thres 3