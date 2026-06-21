# WIBA Argument Mining Pipeline

A pipeline for mining arguments from congressional hearing transcripts using the [WIBA](https://github.com/Armaniii/WIBA) model suite (Detect / Extract / Stance), built for execution on OSC (Ohio Supercomputer Center) SLURM clusters.

## Pipeline Overview

For each transcript dataset, the pipeline:

1. **Load** the `.Rdata` transcript dataset.
2. **Segment** speeches into 3-sentence sliding-window text segments (window=3, step=1).
3. **Detect**: classify each segment as `Argument` / `No Argument` and record a confidence score (WIBA-Detect-V1, LoRA adapter on Meta-Llama-3-8B).
4. **Threshold**: keep only segments with confidence ≥ 0.7.
5. **De-overlap**: within each source document, drop lower-confidence segments that overlap a higher-confidence one already kept (non-maximum suppression).
6. **Extract topic**: identify the topic being argued (WIBA-Extract-V1, fine-tuned Llama causal LM).
7. **Classify stance**: label each segment's stance toward its topic — `Argument in Favor` / `Argument Against` (WIBA-Stance-V1, LoRA adapter on Llama-2-7b-hf).
8. **Final filter**: drop rows where `topic == "No Topic"` or `stance == "No Argument"`.
9. **Save** delivery CSV with columns: `rownum_hs, text_segment, argument_prediction, confidence, topic, stance`, plus a parallel debug CSV with extra metadata for QA.

## Repository Layout

```
wiba/
├── WIBA-Detect-V1/      # LoRA adapter + tokenizer config (argument detection)
├── WIBA-Extract-V1/     # Fine-tuned model config + tokenizer (topic extraction)
├── WIBA-Stance-V1/      # LoRA adapter + tokenizer config (stance classification)
├── scripts/
│   ├── setup_env.sh     # One-time conda environment setup
│   ├── run_wiba.sh      # SLURM batch script — full pipeline run
│   ├── run_test.sh      # SLURM batch script — small subset smoke test
│   ├── wiba_pipeline.py # Core pipeline (segmentation, model loading, inference, I/O)
│   └── test_pipeline.py # Subset test harness with output schema validation
└── .gitignore
```

> **Note:** Base model weights (`base_models/`, e.g. Meta-Llama-3-8B, Llama-2-7b-hf), fine-tuned/adapter weight files (`*.safetensors`, `*.bin`, `*.pth`), and the `data/`, `logs/`, `outputs/` directories are excluded from this repository (see `.gitignore`) — they are too large for GitHub and are kept only on the OSC scratch filesystem.

## Setup

```bash
bash scripts/setup_env.sh
```

Creates a conda environment named `wiba` (Python 3.11) with PyTorch (CUDA 12.1), the Hugging Face stack (`transformers`, `peft`, `accelerate`, `bitsandbytes`), and data utilities (`pyreadr`, `pandas`, etc.).

## Running

**Smoke test** (5 speeches per dataset, validates output schema):
```bash
sbatch scripts/run_test.sh
```

**Full pipeline run** (defense / econ / tech / all):
```bash
sbatch scripts/run_wiba.sh defense
sbatch scripts/run_wiba.sh all
```

Outputs are written to `outputs/{dataset}_output.csv` (delivery) and `outputs/{dataset}_debug.csv` (debug, with metadata for traceability). SLURM logs go to `logs/`.

## Output Schema

| Column                | Description                                              |
|------------------------|------------------------------------------------------------|
| `rownum_hs`            | Source row identifier                                    |
| `text_segment`         | 3-sentence window text                                    |
| `argument_prediction`  | `Argument` (only argumentative segments survive filtering) |
| `confidence`           | Detect model softmax probability for the `Argument` class |
| `topic`                | Topic being argued, extracted by WIBA-Extract-V1           |
| `stance`               | `Argument in Favor` / `Argument Against`                   |
