#!/bin/bash
python filter_subset_inlang_MKQA.py # Filter subset
python filter_golddis_inlang_MKQA.py # Split gold passages and distractors
python statistics_MKQA.py >> statistics_MKQA.txt # Chech statistics


# For filter the cross-lingual retrieval results,
# which is *CORRESPONDING TO THE FILTERING RESULTS OF IN-LANG RETRIEVAL*
#python filter_subset_cross_MKQA.py
