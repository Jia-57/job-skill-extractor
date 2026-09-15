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
* **Annotation**: Job postings annotated at sentence level for hard skills, soft skills, and domain knowledge spans.

The dataset files are placed in the `Data/raw/` directory:
* `dev.json`: Validation set for model selection and checkpointing.
* `test.json`: Standard benchmark test split.
* `test_empty.json`: Negative/empty context samples for evaluating precision.
* `train.json`: Standard training split.
* `train_entities.jsonl`: Formatted entity span annotations for extraction tasks.
* `train_merged.json`: Aggregated training set combining multi-source annotations.


## Results

<!-- Placeholder for experimental evaluation metrics -->

| Architecture | Backbone Model | Strategy | Precision | Recall | F1-Score | Notes |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| Transformer | BERT / DeBERTa | Token Classification | - | - | - | Baseline token tagging |
| LLM | Qwen / Llama | LoRA Fine-Tuning | - | - | - | Full-shot PEFT |
| LLM | Qwen / Llama | In-Context Inference | - | - | - | Few-shot prompting |

*Detailed token-level and span-level metrics will be updated upon final evaluation runs.*



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