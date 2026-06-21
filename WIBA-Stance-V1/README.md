---
library_name: peft
base_model: meta-llama/Llama-2-7b-hf
license: llama2
language:
- en
pipeline_tag: text-classification
tags:
- stance-detection
- stance-classification
- argument-mining
- computational-social-science
- llama
- lora
- peft
- wiba
---

# WIBA Stance Classification (Llama-2-7B LoRA)

**Topic-conditioned stance classification** model: given a text and a target topic, it classifies the text as **`Argument in Favor`**, **`Argument Against`**, or **`No Argument`** with respect to that topic.

This is **Stage 3** of the [WIBA (What Is Being Argued?)](https://arxiv.org/abs/2405.00828) argument mining pipeline:

| Stage | Task | Model | Type |
|---|---|---|---|
| 1. Detect | Is this text an argument? | [armaniii/llama-3-8b-argument-detection](https://huggingface.co/armaniii/llama-3-8b-argument-detection) | LoRA adapter (sequence classification, 2 labels) |
| 2. Extract | What topic is being argued? | [armaniii/llama-3-8b-claim-topic-extraction](https://huggingface.co/armaniii/llama-3-8b-claim-topic-extraction) | Fine-tuned causal LM (pre-quantized 4-bit) |
| **3. Stance** | What position does it take on the topic? | **this repo** | LoRA adapter (sequence classification, 3 labels) |

- 📄 Paper: [WIBA: What Is Being Argued? A Comprehensive Approach to Argument Mining](https://arxiv.org/abs/2405.00828)
- 💻 Code: [github.com/Armaniii/WIBA](https://github.com/Armaniii/WIBA)
- 🌐 Platform: [wiba.dev](https://wiba.dev)

## What this repo contains (adapter, not a full model)

This repo is a **PEFT LoRA adapter** (~80 MB, float32), **not** standalone model weights. It must be loaded on top of the gated base model [`meta-llama/Llama-2-7b-hf`](https://huggingface.co/meta-llama/Llama-2-7b-hf) — request access to the base model and `huggingface-cli login` before use.

| File | Purpose |
|---|---|
| `adapter_config.json` | LoRA config: r=8, alpha=32, dropout=0.05, task type `SEQ_CLS`, target modules = all attention/MLP projections; `modules_to_save=["score"]` |
| `adapter_model.safetensors` | LoRA weights **plus the trained 3-label classification head** (`base_model.model.score.weight`, shape `[3, 4096]`) |
| `tokenizer.json` | Prebuilt fast tokenizer (required by transformers 5.x, which can no longer convert sentencepiece-only Llama-2 repos) |
| `tokenizer.model`, `tokenizer_config.json`, `special_tokens_map.json` | Llama-2 sentencepiece tokenizer (pad token `<unk>`) |

Because the trained `score` head ships inside the adapter file, loading this adapter restores the *complete* classifier — without it, the 3-label head would be randomly initialized and predictions would be meaningless.

> **Checkpoint format note:** the adapter was originally trained and saved with PEFT 0.7.1, whose `score`-head layout cannot be loaded by modern PEFT (≥0.10 raises `KeyError: 'base_model.model.score.weight'`). The files on `main` were converted to the modern format (trained head merged as `base_layer + (alpha/r)·B·A`) and verified **logit-equivalent to the original, on both the modern stack (peft 0.19.1) and the original stack (peft 0.7.1)** — `main` works everywhere. The original-format files are preserved at `revision="937b9babeb146587b5a9463b239ae4ca6ad26e18"`.

## Before you start: get access to the gated Meta base model (one-time, ~10 minutes)

This adapter repo is freely downloadable, but the Meta base model it sits on is **gated** — Meta requires you to accept their license before you can download it. Step by step:

1. **Create a Hugging Face account** (free): go to [huggingface.co/join](https://huggingface.co/join), sign up, and verify your email.
2. **Request access to the base model**: while logged in, open [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf). At the top of the page is a box saying you need to share your contact information to access the model. Fill in the short form, accept the license, and submit.
3. **Wait for the approval email** — usually minutes to a few hours. When the box on the model page changes to "You have been granted access", you're in.
4. **Create an access token**: click your avatar (top right) → **Settings** → **Access Tokens** → **Create new token** → type **Read** → create, and **copy the token** (it looks like `hf_...`). Treat it like a password.
5. **Log in on your computer**: in a terminal run

   ```bash
   pip install -U "huggingface_hub[cli]"
   huggingface-cli login
   ```

   and paste the token when prompted (nothing is shown as you paste — that's normal). Verify with `huggingface-cli whoami`, which should print your username.

This is once per computer. From then on, the code below downloads everything it needs automatically — you'll see progress bars for each file on the first run (~13.6 GB total), after which everything is cached in `~/.cache/huggingface` and loads from disk.

## Hardware requirements — pick your setup

| Setup | What you need | Speed |
|---|---|---|
| **GPU, fp16** | NVIDIA GPU with ≥15 GB free VRAM (e.g. RTX 4090, A100; 16 GB cards work) | sub-second per text |
| **GPU, 4-bit** | NVIDIA GPU with ≥6 GB free VRAM, plus `pip install bitsandbytes` | fast — this is the wiba.dev production configuration |
| **CPU only** | ~30 GB free RAM, no GPU | ~15–25 s per text on 16 cores — fine for trying it out, slow for bulk work |

One-time download for any setup: ~13.6 GB (base model + adapter).

## Quickstart — GPU

```bash
pip install torch transformers peft accelerate sentencepiece
```

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

ADAPTER = "armaniii/llama-stance-classification"
BASE = "meta-llama/Llama-2-7b-hf"

tokenizer = AutoTokenizer.from_pretrained(ADAPTER)  # use the repo's tokenizer
base = AutoModelForSequenceClassification.from_pretrained(
    BASE, num_labels=3, dtype=torch.float16, device_map="auto"
)   # transformers 4.x: use torch_dtype=torch.float16
base.config.pad_token_id = tokenizer.pad_token_id
model = PeftModel.from_pretrained(base, ADAPTER)
model.eval()
```

**Low VRAM? Load the base 4-bit instead** (≈5 GB VRAM, the production setting — needs `pip install bitsandbytes`):

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False,
    bnb_4bit_compute_dtype=torch.float16,
)
base = AutoModelForSequenceClassification.from_pretrained(
    BASE, num_labels=3, device_map="auto", quantization_config=bnb_config
)
```

## Quickstart — CPU (no GPU)

Identical to the GPU code, except load the base in float32 on the CPU:

```python
base = AutoModelForSequenceClassification.from_pretrained(
    BASE, num_labels=3, dtype=torch.float32, device_map="cpu"
)
```

Expect ~15–25 s per prediction on a 16-core machine (verified). Make sure you have ~30 GB of free RAM before starting — on machines without swap, overshooting RAM can freeze the system.

### Prompt format (must match training)

The model uses the Llama-2 instruction wrapper with the WIBA argument-definition system prompt (the same system prompt as the detect stage), and takes **both the target topic and the text**:

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

LABELS = ["No Argument", "Argument in Favor", "Argument Against"]

def classify_stance(topic: str, text: str) -> str:
    prompt = f"[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\nTarget: '{topic}' Text: '{text}' [/INST] "
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        logits = model(**enc).logits
    return LABELS[int(logits.argmax(-1))]

print(classify_stance("gun control", "I support stricter gun control because it reduces gun deaths."))
# -> Argument in Favor
print(classify_stance("gun control", "Gun control laws should be opposed because they violate constitutional rights."))
# -> Argument Against
print(classify_stance("climate change", "The weather is nice today."))
# -> No Argument
```

(Outputs above are actual verified predictions, not illustrations.)

### Label mapping

| Logit index | Label |
|---|---|
| 0 (`LABEL_0`) | `No Argument` |
| 1 (`LABEL_1`) | `Argument in Favor` |
| 2 (`LABEL_2`) | `Argument Against` |

The repo tokenizer's `<unk>` pad token (id 0) is in-vocabulary, so batched inference with `padding=True` works as-is.

## Batch processing many texts (with a progress bar)

Model downloads show progress bars automatically; inference doesn't, so wrap batches in `tqdm` (installed with transformers) exactly as the original WIBA serving code does. The repo's `<unk>` pad token works for batching as-is:

```python
from tqdm import tqdm
from transformers import pipeline

clf = pipeline("text-classification", model=model, tokenizer=tokenizer,
               padding=True, truncation=True, max_length=2048)

pairs = [("climate change", "..."), ("gun control", "...")]  # (topic, text) pairs
prompts = [f"[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\nTarget: '{topic}' Text: '{text}' [/INST] "
           for topic, text in pairs]
idx = {"LABEL_0": "No Argument", "LABEL_1": "Argument in Favor", "LABEL_2": "Argument Against"}
labels = [idx[out["label"]] for out in tqdm(clf(prompts, batch_size=4), total=len(prompts))]
```

## Tested configurations

| Stack | Versions | Status |
|---|---|---|
| Modern (2026) | torch 2.5.1, transformers 5.12.0, peft 0.19.1, accelerate 1.14.0 | ✅ verified (CPU fp32 and the code above) |
| Original (2024) | transformers 4.38.2, peft 0.7.1, accelerate 0.27.2, numpy<2, sentencepiece, protobuf | ✅ verified (`protobuf` is required to read the sentencepiece tokenizer on this stack) |

Logits agree across the two stacks/layouts to ~1e-4.

## How it's used in the WIBA implementation

In the WIBA serving code, this model backs the `/api/stance` endpoint at [wiba.dev](https://wiba.dev): each (text, topic) pair — where the topic typically comes from Stage 2 ([claim topic extraction](https://huggingface.co/armaniii/llama-3-8b-claim-topic-extraction)) or is supplied by the user — is wrapped in the prompt above and classified into the three stance labels.

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
- Built on `meta-llama/Llama-2-7b-hf` (Llama 2 license applies)
