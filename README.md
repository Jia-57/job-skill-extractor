# Job Skill Extractor

A Named Entity Recognition (NER) framework for extracting skills and domain knowledge entities from unstructured job postings. The system supports both token classification via Transformer encoders and generative entity extraction using Large Language Models (LLMs) with Parameter-Efficient Fine-Tuning (LoRA).

## Introduction

Extracting structured skill representations from job descriptions is essential for talent matching, labor market intelligence, and resume screening. This repository provides a modular, reproducible workflow covering:
* Data preprocessing, tokenization, and BIO-to-span schema conversion
* Transformer-based token classification benchmarks
* Generative LLM fine-tuning using Low-Rank Adaptation (LoRA)
* Zero-shot / few-shot inference pipelines
* Standardized token-level evaluation benchmarks


## Data

This project is built and benchmarked on the **SkillSpan** dataset:
* **Source**: [kris927b/SkillSpan](https://github.com/kris927b/SkillSpan)
* **Annotation**: Job postings annotated at sentence level for hard skills, soft skills, and domain knowledge spans, covering the *tech* and *house* sectors.

**Statistics**:

| Split | Job Postings | Sentences | Skill Spans | Knowledge Spans |
| :--- | :---: | :---: | :---: | :---: |
| Train + Dev | 198 | 7,974 | 3,291 | 4,062 |
| Test | 65 | 3,569 | 1,090 | 1,174 |


## Results

Token-level micro-averaged Precision / Recall / F1 (%) on the SkillSpan test set.

### Discriminative Pre-trained Models

| Model | Task | Precision | Recall | F1-Score |
| :--- | :--- | :---: | :---: | :---: |
| jobbert-base-cased | Skill | 74.38 | 77.28 | 75.80 |
| jobbert-base-cased | Knowledge | 75.27 | 82.66 | 78.79 |
| **jobbert-base-cased** | **Overall** | **74.69** | **79.06** | **76.81** |
| BERT-based| Skill | 74.72 | 73.36 | 74.03 |
| BERT-based  | Knowledge | 72.56 | 81.56 | 76.80 |
| **BERT-based** | **Overall** | **73.94** | **76.07** | **74.99** |

### Generative LLMs (Zero-shot / Few-shot / Full-shot LoRA)

| Model | Setting | Precision | Recall | F1-Score |
| :--- | :--- | :---: | :---: | :---: |
| Qwen3-8B | Zero-shot | 40.31 | 56.48 | 47.04 |
| Qwen3-8B | Few-shot | 37.15 | 70.94 | 48.76 |
| Qwen3-8B | Full-shot (LoRA) | 70.44 | 78.86 | 74.41 |
| Llama-3.1-8B-Instruct | Zero-shot | 36.16 | 49.29 | 41.72 |
| Llama-3.1-8B-Instruct | Few-shot | 32.32 | 61.66 | 42.41 |
| Llama-3.1-8B-Instruct | Full-shot (LoRA) | 73.22 | 77.23 | 75.17 |

*Overall rows are micro-averaged across the Skill and Knowledge subtasks. Full per-task and per-domain (tech vs. house) breakdowns are reported in the paper.*

## Getting Started

### 1. Installation

```bash
uv pip install -r requirements.txt
```

### 2. Prepare the Data

Place the dataset files under `Data/raw/`, then run the preprocessing scripts to generate BIO-tagged and span-level formats:

```bash
python data_processing/convert_bio_to_ner.py
python data_processing/convert_to_test_format.py
```

### 3. Train a Model

**Transformer baselines (BERT / JobBERT):**
```bash
python Training/transformer_training.py --model_name bert-base-cased
```

**LLM full-shot fine-tuning (LoRA):**
```bash
python Training/train_fullshot_lora.py --model_name Qwen/Qwen3-8B
```

### 4. Run Inference

```bash
# Transformer checkpoints
python predict/predict_transformers.py

# Zero-shot / few-shot LLM prompting
python predict/run_llm_inference.py --shot_mode few-shot

# Full-shot LoRA fine-tuned model
python predict/run_fullshot_inference.py
```

### 5. Evaluate

```bash
python Evaluation/evaluate_token_level.py --pred_file <path_to_predictions> --gold_file Data/raw/test.json
```


## File Structure

```text
NER_Job/
├── Data/
│   └── raw/
│       ├── dev.json                      # Validation set
│       ├── test.json                     # Test set
│       ├── test_empty.json               # Test set containing negative samples
│       ├── train.json                    # Base training set
│       ├── train_entities.jsonl          # Entity annotations in JSON Lines format
│       └── train_merged.json             # Merged full training set
├── data_processing/
│   ├── Dataset_analysis.py               # Dataset distributions and token statistics
│   ├── convert_bio_to_ner.py             # Converts BIO sequence tags into entity spans
│   └── convert_to_test_format.py         # Formats raw texts for evaluation inference
├── Training/
│   ├── prompts.py                        # Prompt definitions and few-shot templates for LLMs
│   ├── train_fullshot_lora.py            # LoRA fine-tuning routine for generative models
│   └── transformer_training.py           # Training pipeline for encoder-based token classifiers
├── predict/
│   ├── predict_transformers.py           # Inference script for Transformer checkpoints
│   ├── run_fullshot_inference.py         # Batch prediction script for LoRA fine-tuned models
│   └── run_llm_inference.py              # Inference pipeline for prompting LLMs
├── Evaluation/
│   └── evaluate_token_level.py           # Precision, Recall, and F1 calculation scripts
├── requirements.txt                      # Project package dependencies
└── README.md                             # Project overview and documentation
```