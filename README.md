# On the Consistency of Multilingual Context Utilization in Retrieval-Augmented Generation

<div align="center">

**Authors:** Jirui Qi, Raquel Fernández, Arianna Bisazza
[Jirui Qi](https://betswish.github.io/) • [Raquel Fernández](https://staff.fnwi.uva.nl/r.fernandezrovira/) • [[Arianna Bisazza](https://www.cs.rug.nl/~bisazza/)  


</div>

> [!Note]
> Our paper won the Best Paper Award at the [MRL'25 workshop (co-located with EMNLP 2025)](https://sigtyp.github.io/ws2025-mrl.html)! 🎉


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

The repository now separates launchers, pipeline code, configs, and reports:

```text
configs/prompts/                  Prompt templates used by generation and attribution scripts
pipeline/
  embedding/                      Dataset embedding jobs
  retrieval/                      In-language and cross-language retrieval scripts
  filtering/                      Subset construction and dataset statistics
  experiments/
    single_passage/               XQUAD, MKQA, GMMLU single-passage runs
    multi_passage/                MKQA and GMMLU multi-passage runs
  evaluation/                     Performance checking scripts
  attribution/                    MIRAGE attribution and aggregation
analysis/                         Error analysis and correlation studies
scripts/pipeline/                 Numbered shell launchers for the full workflow
metadata/                         Intermediate overlap indices such as has-answer lists
reports/                          Checked-in summary outputs and analysis artifacts
```

The numbered execution order is still preserved through the launchers in `scripts/pipeline/`.

### 0. Embedding

- `scripts/pipeline/0_embed_MKQA.sh`
- `scripts/pipeline/0_embed_GMMLU.sh`

### 1. Retrieval

- `scripts/pipeline/1_retrieval_MKQA.sh`
- `scripts/pipeline/1_retrieval_GMMLU.sh`

### 2. Filter subsets

- `scripts/pipeline/2_filter_subset_MKQA.sh`
- `scripts/pipeline/2_filter_subset_GMMLU.sh`

### 3. Single-passage experiments

- `scripts/pipeline/3_XQUAD_open.sh`
- `scripts/pipeline/3_MKQA_open.sh`
- `scripts/pipeline/3_GMMLU_open.sh`
- `scripts/pipeline/3_GMMLU_choice.sh`

Main entry points:

- `pipeline/experiments/single_passage/XQUAD_open.py`
- `pipeline/experiments/single_passage/MKQA_open.py`
- `pipeline/experiments/single_passage/GMMLU_open.py`
- `pipeline/experiments/single_passage/GMMLU_choice.py`

### 4. Performance checking

- `scripts/pipeline/4_check_performance.sh`
- `pipeline/evaluation/check_XQUAD.py`
- `pipeline/evaluation/check_MKQA.py`
- `pipeline/evaluation/check_GMMLU_open.py`
- `pipeline/evaluation/check_GMMLU_choice.py`

### 5-6. MIRAGE attribution and counting

- `scripts/pipeline/5_attribute_XQUAD.sh`
- `scripts/pipeline/5_attribute_MKQA.sh`
- `scripts/pipeline/6_count_mirage.sh`
- `pipeline/attribution/attribute_XQUAD_open.py`
- `pipeline/attribution/attribute_MKQA_open.py`
- `pipeline/attribution/count_mirage.py`

### 7-9, 12. Multi-passage experiments

- `scripts/pipeline/7_MKQA_open_multi.sh`
- `scripts/pipeline/7_GMMLU_choice_multi.sh`
- `scripts/pipeline/8_check_performance_multi.sh`
- `scripts/pipeline/9_attribute_MKQA_multi.sh`
- `scripts/pipeline/9_attribute_GMMLU_multi.sh`
- `scripts/pipeline/12_count_mirage_multi.sh`

Main entry points:

- `pipeline/experiments/multi_passage/MKQA_open_multi.py`
- `pipeline/experiments/multi_passage/GMMLU_choice_multi.py`
- `pipeline/attribution/attribute_MKQA_open_multi.py`
- `pipeline/attribution/attribute_GMMLU_choice_multi.py`
- `pipeline/attribution/count_mirage_multi.py`

### 10-11. Additional evaluation

- `scripts/pipeline/10_LLM_eval.sh`
- `scripts/pipeline/11_LLM_eval_check.sh`
- `pipeline/evaluation/check_LLM_eval.py`

## Installation

A Conda environment file is provided:

```bash
conda env create -f RAGConsis.yaml
conda activate RAGConsis
```

## Running the Pipeline

The launchers in `scripts/pipeline/` still follow the numbered workflow. They resolve the repository root automatically, so you can call them from anywhere inside the repo.

```bash
bash scripts/pipeline/0_embed_MKQA.sh
bash scripts/pipeline/0_embed_GMMLU.sh

bash scripts/pipeline/1_retrieval_MKQA.sh
bash scripts/pipeline/1_retrieval_GMMLU.sh

bash scripts/pipeline/2_filter_subset_MKQA.sh
bash scripts/pipeline/2_filter_subset_GMMLU.sh

bash scripts/pipeline/3_XQUAD_open.sh
bash scripts/pipeline/3_MKQA_open.sh
bash scripts/pipeline/3_GMMLU_open.sh
bash scripts/pipeline/3_GMMLU_choice.sh

bash scripts/pipeline/4_check_performance.sh

bash scripts/pipeline/5_attribute_XQUAD.sh
bash scripts/pipeline/5_attribute_MKQA.sh
bash scripts/pipeline/6_count_mirage.sh

bash scripts/pipeline/7_MKQA_open_multi.sh
bash scripts/pipeline/7_GMMLU_choice_multi.sh
bash scripts/pipeline/8_check_performance_multi.sh
bash scripts/pipeline/9_attribute_MKQA_multi.sh
bash scripts/pipeline/9_attribute_GMMLU_multi.sh
bash scripts/pipeline/12_count_mirage_multi.sh

bash scripts/pipeline/10_LLM_eval.sh
bash scripts/pipeline/11_LLM_eval_check.sh
```

## Practical Notes

- The shell scripts in the public repository are written for a **Slurm-based GPU setup** and use `sbatch`.
- The scripts assume a Conda environment named **`RAGConsis`**.
- You may need to adapt paths, logging directories, GPU resources, and job-scheduler settings to your own environment.
- `scripts/pipeline/10_LLM_eval.sh` expects a local `sample_instances.py` helper in the repository root.

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

Checked-in summaries now live under `reports/`, while large generated directories such as `results/`, `mirage/`, `log/`, and `log_attri/` are ignored via `.gitignore`.



## License

MIT
