---
library_name: transformers
base_model: meta-llama/Meta-Llama-3-8B
license: llama3
language:
- en
pipeline_tag: text-generation
tags:
- argument-mining
- topic-extraction
- claim-extraction
- computational-social-science
- llama
- 4-bit
- bitsandbytes
- wiba
---

# WIBA Claim Topic Extraction (Llama-3-8B, pre-quantized 4-bit)

**Topic extraction** model: given an argumentative sentence or passage, it generates the **topic being argued** (a short phrase naming the person, place, thing, entity, or idea at issue), or **`No Topic`** if the text is not an argument. The topic may be explicit in the text or implicit and inferred from context.

This is **Stage 2** of the [WIBA (What Is Being Argued?)](https://arxiv.org/abs/2405.00828) argument mining pipeline:

| Stage | Task | Model | Type |
|---|---|---|---|
| 1. Detect | Is this text an argument? | [armaniii/llama-3-8b-argument-detection](https://huggingface.co/armaniii/llama-3-8b-argument-detection) | LoRA adapter (sequence classification, 2 labels) |
| **2. Extract** | What topic is being argued? | **this repo** | Fine-tuned causal LM (pre-quantized 4-bit) |
| 3. Stance | What position does it take on the topic? | [armaniii/llama-stance-classification](https://huggingface.co/armaniii/llama-stance-classification) | LoRA adapter (sequence classification, 3 labels) |

- 📄 Paper: [WIBA: What Is Being Argued? A Comprehensive Approach to Argument Mining](https://arxiv.org/abs/2405.00828)
- 💻 Code: [github.com/Armaniii/WIBA](https://github.com/Armaniii/WIBA)
- 🌐 Platform: [wiba.dev](https://wiba.dev)

## What this repo contains (full model, stored 4-bit quantized)

This repo is a **complete, self-contained fine-tuned model** — no base download, no adapter. But unlike a normal fp16 checkpoint, the weights are **stored pre-quantized with bitsandbytes NF4** (the format the WIBA platform serves in production):

| File | Purpose |
|---|---|
| `model-0000*-of-00002.safetensors` + index | ~6 GB total. Linear-layer weights as packed 4-bit (uint8) with `absmax`/`quant_map` quantization metadata; embeddings and `lm_head` in float16 |
| `config.json` | Model config including the `quantization_config` (bnb NF4, blocksize 64, compute dtype fp16) that tells transformers how to load the 4-bit weights |
| `generation_config.json` | Default generation settings |
| `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json` | Llama-3 tokenizer |

Practical consequences:

- **`bitsandbytes` is a hard requirement** — the checkpoint cannot be loaded without it.
- Do **not** try to remove/override `quantization_config` to get fp16: the stored weights themselves are 4-bit packed, so there is no full-precision copy in this repo. To obtain higher-precision weights, load 4-bit first and call `model.dequantize()` (see below).
- VRAM needed is only **~6 GB** — the model fits on small GPUs.

## Before you start

**No gated access needed** — unlike the detect and stance stages, this repo is fully self-contained (no Meta base model to download), so there is no license gate, no account, and no token required. The first run downloads ~6 GB with progress bars, cached afterward in `~/.cache/huggingface`.

## Hardware requirements — pick your setup

| Setup | What you need | Speed |
|---|---|---|
| **GPU (recommended)** | NVIDIA GPU with ≥8 GB free VRAM, `pip install bitsandbytes` | fast — this is the wiba.dev production configuration |
| **CPU only** | ~25 GB free RAM, no GPU; loads 4-bit then dequantizes (see below) | ~1–2 min per text on 16 cores |

⚠️ Do **not** run `generate()` directly on the 4-bit model on a CPU: bitsandbytes' CPU 4-bit kernels are single-threaded and a single sentence takes over an hour (measured). Use the dequantize recipe below instead.

## Quickstart — GPU

```bash
pip install torch transformers accelerate bitsandbytes
```

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

REPO = "armaniii/llama-3-8b-claim-topic-extraction"

tokenizer = AutoTokenizer.from_pretrained(REPO)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"

# quantization_config ships in config.json — transformers loads the 4-bit
# weights automatically (~6 GB VRAM)
model = AutoModelForCausalLM.from_pretrained(REPO, device_map="auto", low_cpu_mem_usage=True)
model.eval()
```

## Quickstart — CPU (no GPU)

`bitsandbytes` is still required (the checkpoint is stored 4-bit), but after loading, dequantize to bfloat16 so generation runs on all CPU cores (verified: ~25 GB RAM peak, then ~1–2 min per text on 16 cores):

```python
model = AutoModelForCausalLM.from_pretrained(REPO, device_map="cpu", low_cpu_mem_usage=True)
model = model.dequantize().to(torch.bfloat16)
model.eval()
torch.set_num_threads(16)   # match your core count
```

### Prompt format (must match training)

The model expects the Llama-3 chat header format with the WIBA topic-extraction system prompt, and the generation cut off after a few tokens (topics are short):

```python
SYSTEM_PROMPT = """You are a helpful assistant that is specialized in a single task. If the sentence provided is an argument, decide what the topic being argued is using the rules and steps below.
Rules: 
1. An argument is a sentence that must contain a claim AND AT LEAST ONE premise(i.e evidence) supporting that assertion or claim. 
2. A claim is the position being taken in the argument.
3. A premise is a statement that provides evidence to support the claim.
4. In order for a sentence to be an argument it must contain a claim AND at least one premise.
5. If the sentence does not contain a claim AND does not provide any premises to support the claim, then it is a non-argument.
6. If the sentence provided is an argument, then there must be a single topic being argued that is regarding a person, place, thing, entity, or abstract idea. The topic being argued may be explicitly stated OR it may be implicit and must be inferred from the context of the argument.  
7. If the sentence provided is a non-argument, then there is no topic being argued. 

Steps:
1. Decide if the sentence provided is an argument or non-argument using the Rules provided.
2. If the sentence is an argument, output only the topic being argued and your task is finished.
3. If the sentence is a non-argument, only output: No Topic and your task is finished.
4. If the sentence provided is a non-argument, then there is no topic being argued and you should only output: No Topic
5. Let us think through the problem step by step carefully following all the rules outlined."""

def extract_topic(text: str) -> str:
    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        + SYSTEM_PROMPT
        + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        + text
        + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=8, pad_token_id=128009)
    return tokenizer.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()

print(extract_topic("We must act on climate change because temperatures are rising."))
# -> climate change
print(extract_topic("The weather is nice today."))
# -> No Topic
print(extract_topic("Abortion should remain legal because bodily autonomy is a fundamental right."))
# -> abortion
```

(Outputs above are actual verified predictions, not illustrations.)

The original implementation uses the equivalent `pipeline("text-generation", ..., max_new_tokens=8, pad_token_id=128009)` and takes the text after the final `assistant<|end_header_id|>\n\n` marker — the function above does the same thing with `generate`.

### Output

- An argumentative input → a short topic phrase (e.g. `Climate change`, `Gun control`)
- A non-argument input → the literal string `No Topic`

## Batch processing many texts (with a progress bar)

Model downloads show progress bars automatically; generation doesn't, so wrap your loop in `tqdm` (installed with transformers) exactly as the original WIBA serving code does:

```python
from tqdm import tqdm

texts = ["...", "..."]  # your data
topics = [extract_topic(t) for t in tqdm(texts)]
```

## Getting full-precision weights

The repo stores no fp16 copy, but you can dequantize after loading (needs enough memory for the fp16 model, ~16 GB — this is the same call the CPU quickstart uses):

```python
model = AutoModelForCausalLM.from_pretrained(REPO, device_map="auto")
model = model.dequantize()          # bnb 4-bit -> floating point
```

## Tested configurations

| Stack | Versions | Status |
|---|---|---|
| Modern (2026) | torch 2.5.1, transformers 5.12.0, accelerate 1.14.0, bitsandbytes 0.49.2 | ✅ verified (4-bit load, generation, and `dequantize()` path) |

Notes:
- Without `bitsandbytes` installed, `from_pretrained` raises immediately (the checkpoint is pre-quantized).
- Attempting to load with the `quantization_config` removed fails with shape errors (`ckpt torch.Size([8388608, 1]) vs model torch.Size([4096, 4096])`) — the stored weights really are 4-bit packed.
- CPU-only machines: the 4-bit load works (~4 GB RAM, bitsandbytes ships a CPU backend) but 4-bit *inference* on CPU is single-threaded and impractically slow. For CPU inference, load 4-bit, then `model.dequantize()` and cast to `torch.bfloat16`. For real use, a CUDA GPU (~6 GB VRAM) is the practical choice.
- `use_fast=False` (which the original 2024 serving code passed) is silently ignored on transformers 5.x — slow tokenizers were removed; the default fast tokenizer is correct.

## How it's used in the WIBA implementation

In the WIBA serving code, this model backs the `/api/extract` endpoint at [wiba.dev](https://wiba.dev). Texts that Stage 1 classified as `Argument` are passed here to name the topic; the (text, topic) pair is then passed to Stage 3 ([stance classification](https://huggingface.co/armaniii/llama-stance-classification)) to determine whether the argument is in favor of or against that topic. For batch processing the implementation streams prompts through the pipeline with `batch_size=2` and left-padding.

## Citation

```bibtex
@article{irani2024wiba,
  title={WIBA: What Is Being Argued? A Comprehensive Approach to Argument Mining},
  author={Irani, Arman and Park, Ju Yeon and Esterling, Kevin and Faloutsos, Michalis},
  journal={arXiv preprint arXiv:2405.00828},
  year={2024}
}
```

## Notes

- Fine-tuned from `meta-llama/Meta-Llama-3-8B` (Llama 3 license applies). The weights here are already fine-tuned; the base model is not required.
- Internal fine-tune lineage: `llama_cte_v3`.
