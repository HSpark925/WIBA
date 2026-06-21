---
library_name: peft
base_model: meta-llama/Meta-Llama-3-8B
license: llama3
language:
- en
pipeline_tag: text-classification
tags:
- argument-mining
- argument-detection
- computational-social-science
- llama
- lora
- peft
- wiba
---

# WIBA Argument Detection (Llama-3-8B LoRA)

Binary **argument detection** model: given a sentence or passage, it classifies the text as **`Argument`** or **`NoArgument`**. An argument is defined as text containing a *claim* supported by at least one *premise* (evidence or reasoning).

This is **Stage 1** of the [WIBA (What Is Being Argued?)](https://arxiv.org/abs/2405.00828) argument mining pipeline:

| Stage | Task | Model | Type |
|---|---|---|---|
| **1. Detect** | Is this text an argument? | **this repo** | LoRA adapter (sequence classification, 2 labels) |
| 2. Extract | What topic is being argued? | [armaniii/llama-3-8b-claim-topic-extraction](https://huggingface.co/armaniii/llama-3-8b-claim-topic-extraction) | Fine-tuned causal LM (pre-quantized 4-bit) |
| 3. Stance | What position does it take on the topic? | [armaniii/llama-stance-classification](https://huggingface.co/armaniii/llama-stance-classification) | LoRA adapter (sequence classification, 3 labels) |

- 📄 Paper: [WIBA: What Is Being Argued? A Comprehensive Approach to Argument Mining](https://arxiv.org/abs/2405.00828)
- 💻 Code: [github.com/Armaniii/WIBA](https://github.com/Armaniii/WIBA)
- 🌐 Platform: [wiba.dev](https://wiba.dev)

## What this repo contains (adapter, not a full model)

This repo is a **PEFT LoRA adapter** (~190 MB, float32), **not** standalone model weights. It must be loaded on top of the gated base model [`meta-llama/Meta-Llama-3-8B`](https://huggingface.co/meta-llama/Meta-Llama-3-8B) — request access to the base model and `huggingface-cli login` before use.

| File | Purpose |
|---|---|
| `adapter_config.json` | LoRA config: r=8, alpha=32, dropout=0.05, task type `SEQ_CLS`, target modules = all attention/MLP projections; `modules_to_save=["score"]` |
| `adapter_model.safetensors` | LoRA weights **plus the trained 2-label classification head** (`base_model.model.score.weight`, shape `[2, 4096]`) |
| `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json` | Fine-tuned tokenizer |

Because the trained `score` head ships inside the adapter file, loading this adapter restores the *complete* classifier — the base model's randomly-initialized head is replaced at load time.

> **Checkpoint format note:** the adapter was originally trained and saved with PEFT 0.7.1, whose `score`-head layout cannot be loaded by modern PEFT (≥0.10 raises `KeyError: 'base_model.model.score.weight'`). The files on `main` were converted to the modern format (trained head merged as `base_layer + (alpha/r)·B·A`) and verified **logit-equivalent to the original within 1e-4**. If you are on a 2024-era stack (peft 0.7.1 / transformers 4.38), load the original layout instead with `revision="69bff7d70a27f9255f5c373ff53cff8ad0a517cb"`.

## Before you start: get access to the gated Meta base model (one-time, ~10 minutes)

This adapter repo is freely downloadable, but the Meta base model it sits on is **gated** — Meta requires you to accept their license before you can download it. Step by step:

1. **Create a Hugging Face account** (free): go to [huggingface.co/join](https://huggingface.co/join), sign up, and verify your email.
2. **Request access to the base model**: while logged in, open [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B). At the top of the page is a box saying you need to share your contact information to access the model. Fill in the short form, accept the license, and submit.
3. **Wait for the approval email** — usually minutes to a few hours. When the box on the model page changes to "You have been granted access", you're in.
4. **Create an access token**: click your avatar (top right) → **Settings** → **Access Tokens** → **Create new token** → type **Read** → create, and **copy the token** (it looks like `hf_...`). Treat it like a password.
5. **Log in on your computer**: in a terminal run

   ```bash
   pip install -U "huggingface_hub[cli]"
   huggingface-cli login
   ```

   and paste the token when prompted (nothing is shown as you paste — that's normal). Verify with `huggingface-cli whoami`, which should print your username.

This is once per computer. From then on, the code below downloads everything it needs automatically — you'll see progress bars for each file on the first run (~16.3 GB total), after which everything is cached in `~/.cache/huggingface` and loads from disk.

## Hardware requirements — pick your setup

| Setup | What you need | Speed |
|---|---|---|
| **GPU, fp16** | NVIDIA GPU with ≥18 GB free VRAM (e.g. RTX 3090/4090, A100) | sub-second per text |
| **GPU, 4-bit** | NVIDIA GPU with ≥8 GB free VRAM, plus `pip install bitsandbytes` | fast — this is the wiba.dev production configuration |
| **CPU only** | ~35 GB free RAM, no GPU | ~20 s per text on 16 cores — fine for trying it out, slow for bulk work |

One-time download for any setup: ~16.3 GB (base model + adapter).

## Quickstart — GPU

```bash
pip install torch transformers peft accelerate
```

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

ADAPTER = "armaniii/llama-3-8b-argument-detection"
BASE = "meta-llama/Meta-Llama-3-8B"

tokenizer = AutoTokenizer.from_pretrained(ADAPTER)  # use the repo's tokenizer, not the base's
# The repo tokenizer's [UNK] pad token has id 128256, which is OUTSIDE the base
# model's 128256-token embedding table — padding with it crashes batched
# inference. Use eos as the pad token instead:
tokenizer.pad_token = tokenizer.eos_token

base = AutoModelForSequenceClassification.from_pretrained(
    BASE, num_labels=2, dtype=torch.float16, device_map="auto"
)   # transformers 4.x: use torch_dtype=torch.float16
base.config.pad_token_id = tokenizer.pad_token_id
model = PeftModel.from_pretrained(base, ADAPTER)
model.eval()
```

**Low VRAM? Load the base 4-bit instead** (≈6 GB VRAM, the production setting — needs `pip install bitsandbytes`):

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False,
    bnb_4bit_compute_dtype=torch.float16,
)
base = AutoModelForSequenceClassification.from_pretrained(
    BASE, num_labels=2, device_map="auto", quantization_config=bnb_config
)
```

## Quickstart — CPU (no GPU)

Identical to the GPU code, except load the base in float32 on the CPU:

```python
base = AutoModelForSequenceClassification.from_pretrained(
    BASE, num_labels=2, dtype=torch.float32, device_map="cpu"
)
```

Expect ~90 s to load and ~20 s per prediction on a 16-core machine (verified). Make sure you have ~35 GB of free RAM before starting — on machines without swap, overshooting RAM can freeze the system.

### Prompt format (must match training)

The model was trained with the Llama-2-style instruction wrapper below (kept verbatim in the WIBA implementation, including the chain-of-thought "transition network" system prompt):

```python
SYSTEM_PROMPT = """Premise: A statement that provides evidence, reasons, or support.
Conclusion: A statement that is being argued for or claimed based on the premises.

Argument/NoArgument Transition Network:
Start State --Token matches Premise Definition--> Premise State Augmentation (Premise sub-network) --Token matches Conclusion definition--> Conclusion State Augmentation (Conclusion sub-network) ----> Argument State ----> End State
Start State --Token matches Conclusion definition--> Conclusion State Augmentation (Conclusion sub-network) ----> Premise State Augmentation (Premise sub-network) ----> Argument State ----> End State
Start State --Token matches Premise Definition--> Premise State Augmentation (Premise sub-network) --Token does not match Conclusion Definition--> NoArgument State -> End State
Start State --Token matches Conclusion definition--> Conclusion State Augmentation (Conclusion sub-network) --Token does not match Premise Definition--> NoArgument State ----> End State
Start State ----> NoArgument State ----> End State
Start State --Token does not match Premise Definition--> NoArgument State ----> End State
Start State --Token does not match Conclusion Definition--> NoArgument State ----> End State

Premise State Augmentation (Premise sub-network) ----> Premise Content State ----> Premise Conjunction State ----> Premise State ----> Premise End State
Conclusion State Augmentation (Premise sub-network) ----> Conclusion Content State ----> Conclusion Conjunction State ----> Conclusion State ----> Conclusion End State

Argument State ----> Action: Classify as Argument ----> Argument State
NoArgument State ----> Action: Classify as NoArgument ----> NoArgument State

Follow this chain of thought reasoning and apply the transition network rules and systematically determine whether a given sentence is an argument or not, based on the presence or absence of premises and claims.
If the sentence is an argument, output only 'Argument' and your task is finished.
If the sentence is not an argument, output only 'NoArgument' and your task is finished."""

import string

def detect_argument(text: str) -> str:
    if text and text[-1] not in string.punctuation:  # original implementation adds a final period
        text = text + "."
    prompt = f"[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\nText: '{text}' [/INST] "
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        logits = model(**enc).logits
    return ["NoArgument", "Argument"][int(logits.argmax(-1))]

print(detect_argument("We should ban assault weapons because they enable mass shootings."))
# -> Argument
print(detect_argument("The weather is nice today."))
# -> NoArgument
```

(Outputs above are actual verified predictions, not illustrations.)

### Label mapping

| Logit index | Label |
|---|---|
| 0 (`LABEL_0`) | `NoArgument` |
| 1 (`LABEL_1`) | `Argument` |

## Batch processing many texts (with a progress bar)

Model downloads show progress bars automatically; inference doesn't, so wrap batches in `tqdm` (installed with transformers) exactly as the original WIBA serving code does. The eos pad-token override from the Quickstart must be in place:

```python
from tqdm import tqdm
from transformers import pipeline

clf = pipeline("text-classification", model=model, tokenizer=tokenizer,
               padding=True, truncation=True, max_length=2048)

texts = ["...", "..."]  # your data
prompts = [f"[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\nText: '{t}' [/INST] " for t in texts]
labels = ["Argument" if out["label"] == "LABEL_1" else "NoArgument"
          for out in tqdm(clf(prompts, batch_size=4), total=len(prompts))]
```

## Tested configurations

| Stack | Versions | Status |
|---|---|---|
| Modern (2026) | torch 2.5.1, transformers 5.12.0, peft 0.19.1, accelerate 1.14.0 | ✅ verified (CPU fp32 and the code above) |
| Original (2024) | transformers 4.38.2, peft 0.7.1, accelerate 0.27.2, numpy<2 | ✅ verified against `revision="69bff7d7..."` (original checkpoint layout) |

Logits agree across the two stacks/layouts to ~1e-4.

## How it's used in the WIBA implementation

In the WIBA serving code, this model backs the `/api/detect` endpoint at [wiba.dev](https://wiba.dev): each input text is wrapped in the prompt above, run through the classifier, and `LABEL_1` is mapped to `Argument`. Texts classified as `Argument` are then passed downstream to topic extraction and stance classification.

## Citation

```bibtex
@article{irani2024wiba,
  title={WIBA: What Is Being Argued? A Comprehensive Approach to Argument Mining},
  author={Irani, Arman and Park, Ju Yeon and Esterling, Kevin and Faloutsos, Michalis},
  journal={arXiv preprint arXiv:2405.00828},
  year={2024}
}
```

## Framework versions

- Trained with PEFT 0.7.1; checkpoint on `main` re-saved in modern PEFT format (verified with PEFT 0.19.1)
- Built on `meta-llama/Meta-Llama-3-8B` (Llama 3 license applies)
