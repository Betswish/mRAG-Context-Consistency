# On the Consistency of Multilingual Context Utilization in Retrieval-Augmented Generation

<div align="center">

**Authors:** Jirui Qi, Raquel Fernández, Arianna Bisazza
[Jirui Qi](https://betswish.github.io/) • [Raquel Fernández](https://staff.fnwi.uva.nl/r.fernandezrovira/) • [[Arianna Bisazza](https://www.cs.rug.nl/~bisazza/)  


</div>

> [!Note]
> Our paper won the Best Paper Award at the [MRL 2025 workshop](https://sigtyp.github.io/ws2025-mrl.html)! 🎉


## Overview

Retrieval-augmented generation (RAG) in multilingual settings often retrieves relevant evidence in a language different from the user query. This repository accompanies our study of how consistently multilingual LLMs can make use of such multilingual context, independently of retrieval quality.

We focus on three core questions:

1. Can multilingual LLMs make consistent use of a relevant passage regardless of its language?
2. Can they produce the final answer in the expected target language?
3. Can they still focus on the relevant evidence when multiple distracting passages in different languages are present?

Our experiments cover **4 multilingual LLMs**, **3 multilingual QA benchmarks**, and **48 languages**.

## Main Findings

- Multilingual LLMs are often able to extract relevant information from passages written in a different language than the query.
- A larger bottleneck is generating the final answer in the correct target language.
- Distracting passages reduce answer quality regardless of their language.
- Distractors in the query language tend to have a slightly stronger negative influence.
- In some settings, several out-language gold passages can be more useful than a single in-language gold passage, highlighting the value of cross-lingual retrieval.

## Citation

If you use this repository, please cite:

```bibtex
@inproceedings{qi-etal-2025-consistency,
  title = {On the Consistency of Multilingual Context Utilization in Retrieval-Augmented Generation},
  author = {Qi, Jirui and Fern{\'a}ndez, Raquel and Bisazza, Arianna},
  booktitle = {Proceedings of the 5th Workshop on Multilingual Representation Learning (MRL 2025)},
  year = {2025},
  pages = {199--225},
  publisher = {Association for Computational Linguistics},
  url = {https://aclanthology.org/2025.mrl-main.15/},
  doi = {10.18653/v1/2025.mrl-main.15}
}
```

## Benchmarks

We evaluate on three multilingual QA datasets:

| Dataset | Task type | Languages | Queries | Queries with gold passages used in experiments |
| --- | --- | ---: | ---: | ---: |
| XQUAD | Extractive QA | 12 | 1,190 | 1,190 |
| MKQA | Open-domain QA | 24 | 6,758 | 5,951 |
| GMMLU | Multiple-choice QA | 42 | 14,042 | 4,136 |

## Models

We evaluate the following multilingual instruction-tuned LLMs:

- `CohereForAI/aya-expanse-8b`
- `meta-llama/Llama-3.2-3B-Instruct`
- `google/gemma-2-9b-it`
- `Qwen/Qwen2.5-7B-Instruct`

## Method Summary

The repository is organized as a stage-based pipeline for controlled multilingual RAG experiments.

- **Embedding and retrieval** for MKQA and GMMLU
- **Subset filtering** to keep questions with at least one gold passage in any studied language
- **Single-passage experiments** on XQUAD, MKQA, and GMMLU
- **Performance checking** and result aggregation
- **MIRAGE attribution** for analyzing context utilization
- **Multi-passage experiments** with gold passages and distractors
- **Additional evaluation** scripts

For MKQA and GMMLU, passages are retrieved from multilingual Wikipedia using a multilingual embedding retriever. The analysis also uses MIRAGE-based attribution to inspect whether generated answers are grounded in the provided context.

## Repository Structure

The repository follows the numbered execution order below.

### 0. Embedding

- `0_embed_MKQA.sh`
- `0_embed_GMMLU.sh`

### 1. Retrieval

- `1_retrieval_MKQA.sh`
- `1_retrieval_GMMLU.sh`

### 2. Filter subsets

- `2_filter_subset_MKQA.sh`
- `2_filter_subset_GMMLU.sh`

### 3. Single-passage experiments

- `3_XQUAD_open.sh`
- `3_MKQA_open.sh`
- `3_GMMLU_open.sh`
- `3_GMMLU_choice.sh`

Main entry points:

- `XQUAD_open.py`
- `MKQA_open.py`
- `GMMLU_open.py`
- `GMMLU_choice.py`

### 4. Performance checking

- `4_check_performance.sh`
- `check_XQUAD.py`
- `check_MKQA.py`
- `check_GMMLU_open.py`
- `check_GMMLU_choice.py`

### 5-6. MIRAGE attribution and counting

- `5_attribute_XQUAD.sh`
- `5_attribute_MKQA.sh`
- `6_count_mirage.sh`
- `attribute_XQUAD_open.py`
- `attribute_MKQA_open.py`
- `count_mirage.py`

### 7-9, 12. Multi-passage experiments

- `7_MKQA_open_multi.sh`
- `7_GMMLU_choice_multi.sh`
- `8_check_performance_multi.sh`
- `9_attribute_MKQA_multi.sh`
- `9_attribute_GMMLU_multi.sh`
- `12_count_mirage_multi.sh`

Main entry points:

- `MKQA_open_multi.py`
- `GMMLU_choice_multi.py`
- `attribute_MKQA_open_multi.py`
- `attribute_GMMLU_choice_multi.py`
- `count_mirage_multi.py`

### 10-11. Additional evaluation

- `10_LLM_eval.sh`
- `11_LLM_eval_check.sh`
- `check_LLM_eval.py`

## Installation

A Conda environment file is provided:

```bash
conda env create -f RAGConsis.yaml
conda activate RAGConsis
```

## Running the Pipeline

The current repository is organized as a numbered pipeline. A simple way to reproduce the experiments is to run the scripts in order:

```bash
bash 0_embed_MKQA.sh
bash 0_embed_GMMLU.sh

bash 1_retrieval_MKQA.sh
bash 1_retrieval_GMMLU.sh

bash 2_filter_subset_MKQA.sh
bash 2_filter_subset_GMMLU.sh

bash 3_XQUAD_open.sh
bash 3_MKQA_open.sh
bash 3_GMMLU_open.sh
bash 3_GMMLU_choice.sh

bash 4_check_performance.sh

bash 5_attribute_XQUAD.sh
bash 5_attribute_MKQA.sh
bash 6_count_mirage.sh

bash 7_MKQA_open_multi.sh
bash 7_GMMLU_choice_multi.sh
bash 8_check_performance_multi.sh
bash 9_attribute_MKQA_multi.sh
bash 9_attribute_GMMLU_multi.sh
bash 12_count_mirage_multi.sh

bash 10_LLM_eval.sh
bash 11_LLM_eval_check.sh
```

## Practical Notes

- The shell scripts in the public repository are written for a **Slurm-based GPU setup** and use `sbatch`.
- The scripts assume a Conda environment named **`RAGConsis`**.
- You may need to adapt paths, logging directories, GPU resources, and job-scheduler settings to your own environment.

For example, the provided experiment scripts submit jobs with `sbatch` and run Python commands via `conda run -n RAGConsis ...`.

## Experimental Settings

### Single-passage setting

- **XQUAD** uses one gold passage that is available in parallel across 12 languages.
- **MKQA** and **GMMLU** use retrieved gold passages from multilingual Wikipedia.
- The single-passage setup compares:
  - no context
  - one in-language gold passage
  - one out-language gold passage

### Multi-passage setting

For more realistic multilingual RAG scenarios, the repository also includes multi-passage experiments with mixtures of:

- one or more gold passages
- one or more distracting passages
- in-language and out-language context combinations

### GMMLU variants

The repository includes two GMMLU settings:

- **GMMLU-Open**: free-form answer generation
- **GMMLU-Choice**: the model selects among `A / B / C / D`

This helps separate context understanding from target-language generation.

## Outputs

Depending on the stage, the scripts may generate:

- retrieval outputs
- filtered subsets
- model predictions
- evaluation summaries
- MIRAGE attribution files
- aggregated multi-passage analysis results
- scheduler log files

Please inspect the individual Python or shell scripts if you want to change output paths or filenames.



## License

MIT
