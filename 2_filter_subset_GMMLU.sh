#!/bin/bash
python filter_subset_inlang_GMMLU.py # Filter subset
python filter_golddis_inlang_GMMLU.py # Split gold passages and distractors
python statistics_GMMLU.py >> statistics_GMMLU.txt # Chech statistics

# For filter the cross-lingual retrieval results, 
# which is *CORRESPONDING TO THE FILTERING RESULTS OF IN-LANG RETRIEVAL*
#python filter_subset_cross_GMMLU.py

