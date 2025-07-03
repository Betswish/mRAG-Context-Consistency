#!/bin/bash
rm check_MKQA_multi.txt
rm check_GMMLU_choice_multi.txt

python check_MKQA_multi.py >> check_MKQA_multi.txt
python check_GMMLU_choice_multi.py >> check_GMMLU_choice_multi.txt

