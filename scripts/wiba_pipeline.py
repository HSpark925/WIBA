"""WIBA argument mining pipeline — congressional hearing transcripts.

Pipeline:
  1. Load .Rdata transcript dataset
  2. Create 3-sentence sliding-window segments (window=3, step=1)
  3. Detect arguments + record confidence score
  4. Keep segments with confidence >= 0.7
  5. Remove overlapping neighbouring segments (keep highest confidence per doc)
  6. Extract topic
  7. Classify stance
  8. Drop topic == "No Topic" and stance == "No Argument"
  9. Save CSV with required columns:
       rownum_hs, text_segment, argument_prediction, confidence, topic, stance
"""

import argparse
import gc
import logging
import re
import string
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

BASE_DIR = Path("/fs/scratch/PAS2882/integin1212/wiba")
DATA_DIR = BASE_DIR / "data" / "RA_files"
OUTPUT_DIR = BASE_DIR / "outputs"

DETECT_ADAPTER = str(BASE_DIR / "WIBA-Detect-V1")
EXTRACT_MODEL  = str(BASE_DIR / "WIBA-Extract-V1")
STANCE_ADAPTER = str(BASE_DIR / "WIBA-Stance-V1")

DETECT_BASE = str(BASE_DIR / "base_models" / "Meta-Llama-3-8B")
STANCE_BASE = str(BASE_DIR / "base_models" / "Llama-2-7b-hf")

DETECT_THRESHOLD = 0.7
WINDOW_SIZE = 3
STEP_SIZE   = 1

# Required output columns (in order); extras follow after
REQUIRED_COLS = ["rownum_hs", "text_segment", "argument_prediction", "confidence", "topic", "stance"]

DETECT_SYSTEM_PROMPT = """Premise: A statement that provides evidence, reasons, or support.
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

EXTRACT_SYSTEM_PROMPT = """You are a helpful assistant that is specialized in a single task. If the sentence provided is an argument, decide what the topic being argued is using the rules and steps below.
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

STANCE_LABELS = {
    "LABEL_0": "No Argument",
    "LABEL_1": "Argument in Favor",
    "LABEL_2": "Argument Against",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_rdata(path: str) -> pd.DataFrame:
    """Load an .Rdata / .RData file and return the first data frame found."""
    import pyreadr
    result = pyreadr.read_r(path)
    dfs = [v for v in result.values() if isinstance(v, pd.DataFrame)]
    if not dfs:
        raise ValueError(f"No data frame found in {path}")
    df = dfs[0]
    log.info("Loaded %s  |  %d rows  |  columns: %s", path, len(df), list(df.columns))
    return df


def find_text_column(df: pd.DataFrame) -> str:
    """Return the name of the column that holds the speech / statement text."""
    candidates = [
        "speech", "text", "speech_text", "statement", "content", "body", "utterance",
    ]
    for col in candidates:
        if col in df.columns:
            log.info("Using text column: '%s'", col)
            return col
    # fall back to the string column with the longest average length
    str_cols = df.select_dtypes(include="object").columns.tolist()
    if not str_cols:
        raise ValueError("No string columns found in the data frame")
    col = max(str_cols, key=lambda c: df[c].dropna().str.len().mean())
    log.warning("Text column not recognised by name; falling back to '%s'", col)
    return col


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def split_sentences(text: str) -> list[str]:
    """Lightweight sentence splitter — mirrors the R tokenisation approach."""
    text = str(text).strip()
    raw = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in raw if s.strip()]


def create_segments(
    df: pd.DataFrame,
    text_col: str,
    window: int = 3,
    step: int = 1,
) -> pd.DataFrame:
    """Produce a sliding-window segment table from the speech data.

    Every segment row carries:
      - _doc_id       : original DataFrame index (internal; exposed as original_id in debug output)
      - start_index   : 0-based index of the first sentence in the window
      - end_index     : 0-based index of the last sentence in the window
      - word_count    : number of words in the segment
      - text_segment  : concatenated window text
      - rownum_hs     : preserved from source if present; otherwise = _doc_id
      - all other metadata columns from the source row
    """
    has_rownum = "rownum_hs" in df.columns
    meta_cols  = [c for c in df.columns if c != text_col]
    records    = []

    for row_idx, row in df.iterrows():
        raw_text = row[text_col]
        if not isinstance(raw_text, str) or not raw_text.strip():
            continue
        sentences = split_sentences(raw_text)

        windows = (
            [(0, sentences)]                                             # whole speech as one window
            if len(sentences) < window
            else [
                (start, sentences[start: start + window])
                for start in range(0, len(sentences) - window + 1, step)
            ]
        )
        for start, sent_window in windows:
            record = {c: row[c] for c in meta_cols}
            record["_doc_id"]      = row_idx
            record["start_index"]  = start
            record["end_index"]    = start + len(sent_window) - 1
            record["text_segment"] = " ".join(sent_window)
            record["word_count"]   = len(sent_window[0].split()) if len(sent_window) == 1 \
                                     else len(" ".join(sent_window).split())
            if not has_rownum:
                record["rownum_hs"] = row_idx
            records.append(record)

    result = pd.DataFrame(records)
    log.info("Segmentation: %d segments from %d source rows", len(result), len(df))
    return result


# ---------------------------------------------------------------------------
# Overlap removal (non-maximum suppression within each document)
# ---------------------------------------------------------------------------

def remove_overlapping(
    segments: pd.DataFrame,
    window: int = 3,
) -> pd.DataFrame:
    """Among confidence-ranked segments within each document, drop any segment
    whose window overlaps a higher-confidence segment already selected.

    Two windows overlap when |start_i - start_j| < window_size.
    The `confidence` column must already exist on `segments`.
    """
    kept = []
    for _doc_id, group in segments.groupby("_doc_id", sort=False):
        accepted_starts: list[int] = []
        for idx, row in group.sort_values("confidence", ascending=False).iterrows():
            pos = row["start_index"]
            if all(abs(pos - p) >= window for p in accepted_starts):
                kept.append(idx)
                accepted_starts.append(pos)

    result = segments.loc[kept].sort_index()
    log.info(
        "Overlap removal: %d → %d segments",
        len(segments), len(result),
    )
    return result


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def _bnb_config():
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
        bnb_4bit_compute_dtype=torch.float16,
    )


def load_detect_model(use_4bit: bool = True):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from peft import PeftModel

    log.info("Loading Detect model  (base: %s)", DETECT_BASE)
    tokenizer = AutoTokenizer.from_pretrained(DETECT_ADAPTER)
    tokenizer.pad_token = tokenizer.eos_token

    kwargs = {"num_labels": 2}
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
        kwargs["quantization_config" if use_4bit else "torch_dtype"] = (
            _bnb_config() if use_4bit else torch.float16
        )
    else:
        kwargs["device_map"] = "cpu"

    base = AutoModelForSequenceClassification.from_pretrained(DETECT_BASE, **kwargs)
    base.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(base, DETECT_ADAPTER)
    model.eval()
    log.info("Detect model ready.")
    return tokenizer, model


def load_extract_model():
    from transformers import AutoTokenizer, AutoModelForCausalLM

    log.info("Loading Extract model  (%s)", EXTRACT_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(EXTRACT_MODEL)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            EXTRACT_MODEL, device_map="auto", low_cpu_mem_usage=True,
        )
    else:
        # 4-bit checkpoint on CPU: dequantize to bfloat16 so generation uses all cores
        model = AutoModelForCausalLM.from_pretrained(
            EXTRACT_MODEL, device_map="cpu", low_cpu_mem_usage=True,
        )
        model = model.dequantize().to(torch.bfloat16)
        torch.set_num_threads(16)

    model.eval()
    log.info("Extract model ready.")
    return tokenizer, model


def load_stance_model(use_4bit: bool = True):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from peft import PeftModel

    log.info("Loading Stance model  (base: %s)", STANCE_BASE)
    tokenizer = AutoTokenizer.from_pretrained(STANCE_ADAPTER)

    kwargs = {"num_labels": 3}
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
        kwargs["quantization_config" if use_4bit else "torch_dtype"] = (
            _bnb_config() if use_4bit else torch.float16
        )
    else:
        kwargs["device_map"] = "cpu"

    base = AutoModelForSequenceClassification.from_pretrained(STANCE_BASE, **kwargs)
    base.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(base, STANCE_ADAPTER)
    model.eval()
    log.info("Stance model ready.")
    return tokenizer, model


def _free():
    """Call after `del`-ing all references to a model/tokenizer in the caller's
    scope, to actually release the freed GPU memory back to the allocator."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _add_period(text: str) -> str:
    if text and text[-1] not in string.punctuation:
        return text + "."
    return text


def run_detect(texts: list[str], tokenizer, model, batch_size: int = 4) -> list[float]:
    """Return softmax probability of class 'Argument' (index 1) for each text."""
    device = next(model.parameters()).device

    def make_prompt(t):
        return f"[INST] <<SYS>>\n{DETECT_SYSTEM_PROMPT}\n<</SYS>>\n\nText: '{_add_period(t)}' [/INST] "

    prompts = [make_prompt(t) for t in texts]
    scores  = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="Detect"):
        enc = tokenizer(
            prompts[i: i + batch_size],
            return_tensors="pt", padding=True, truncation=True, max_length=2048,
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits.float(), dim=-1)
        scores.extend(probs[:, 1].cpu().tolist())   # index 1 = Argument

    return scores


def run_extract(texts: list[str], tokenizer, model) -> list[str]:
    """Return the extracted topic string for each argumentative text."""
    device = next(model.parameters()).device

    def make_prompt(t):
        return (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            + EXTRACT_SYSTEM_PROMPT
            + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            + t
            + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    topics = []
    for t in tqdm(texts, desc="Extract"):
        enc = tokenizer(make_prompt(t), return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=8, pad_token_id=128009)
        topics.append(
            tokenizer.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
        )
    return topics


def run_stance(
    texts: list[str], topics: list[str], tokenizer, model, batch_size: int = 4,
) -> list[str]:
    """Return stance label for each (text, topic) pair."""
    device = next(model.parameters()).device

    def make_prompt(topic, text):
        return (
            f"[INST] <<SYS>>\n{DETECT_SYSTEM_PROMPT}\n<</SYS>>\n\n"
            f"Target: '{topic}' Text: '{text}' [/INST] "
        )

    prompts = [make_prompt(tp, tx) for tp, tx in zip(topics, texts)]
    labels  = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="Stance"):
        enc = tokenizer(
            prompts[i: i + batch_size],
            return_tensors="pt", padding=True, truncation=True, max_length=2048,
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        preds = logits.argmax(-1).cpu().tolist()
        labels.extend(STANCE_LABELS[f"LABEL_{p}"] for p in preds)

    return labels


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

# Metadata columns preserved for traceability — not part of the delivery spec
DEBUG_META_COLS = [
    "hearing_topic", "hearing_topic_name",
    "original_id", "start_index", "end_index", "word_count",
]


def build_delivery_output(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the six columns required by the project lead."""
    df = df.copy()
    if "_doc_id" in df.columns:
        df = df.drop(columns=["_doc_id"])
    present = [c for c in REQUIRED_COLS if c in df.columns]
    return df[present]


def build_debug_output(df: pd.DataFrame) -> pd.DataFrame:
    """Return required columns + all metadata columns for internal validation."""
    df = df.copy()
    # Expose _doc_id as original_id
    if "_doc_id" in df.columns:
        df = df.rename(columns={"_doc_id": "original_id"})
    present_required = [c for c in REQUIRED_COLS if c in df.columns]
    present_meta     = [c for c in DEBUG_META_COLS if c in df.columns]
    return df[present_required + present_meta]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(dataset_name: str, use_4bit: bool = True) -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1 · Load data
    for suffix in ("Rdata", "RData", "rdata"):
        rdata_path = DATA_DIR / f"df_{dataset_name}.{suffix}"
        if rdata_path.exists():
            break
    else:
        raise FileNotFoundError(
            f"Dataset not found in {DATA_DIR}. Expected df_{dataset_name}.Rdata"
        )

    df       = load_rdata(str(rdata_path))

    # Rename pre-existing topic/topic_name columns to avoid collision with
    # WIBA-extracted 'topic' produced later in the pipeline.
    rename_map = {}
    if "topic"      in df.columns: rename_map["topic"]      = "hearing_topic"
    if "topic_name" in df.columns: rename_map["topic_name"] = "hearing_topic_name"
    if rename_map:
        df = df.rename(columns=rename_map)
        log.info("Renamed source columns: %s", rename_map)

    text_col = find_text_column(df)

    # 2 · Segment
    segments = create_segments(df, text_col, window=WINDOW_SIZE, step=STEP_SIZE)

    # 3 · Detect
    detect_tok, detect_model = load_detect_model(use_4bit=use_4bit)
    scores = run_detect(segments["text_segment"].tolist(), detect_tok, detect_model)
    del detect_model, detect_tok
    _free()

    segments["confidence"]          = scores
    segments["argument_prediction"] = [
        "Argument" if s >= DETECT_THRESHOLD else "No Argument" for s in scores
    ]

    # 4 · Keep only argumentative segments
    above = segments[segments["argument_prediction"] == "Argument"].copy()
    log.info(
        "After detect threshold %.1f: %d / %d segments kept",
        DETECT_THRESHOLD, len(above), len(segments),
    )

    # 5 · Remove overlapping neighbours (within each source row)
    deduped = remove_overlapping(above, window=WINDOW_SIZE)

    # 6 · Extract topic
    extract_tok, extract_model = load_extract_model()
    deduped = deduped.copy()
    deduped["topic"] = run_extract(
        deduped["text_segment"].tolist(), extract_tok, extract_model,
    )
    del extract_model, extract_tok
    _free()

    # 7 · Classify stance
    stance_tok, stance_model = load_stance_model(use_4bit=use_4bit)
    deduped["stance"] = run_stance(
        deduped["text_segment"].tolist(),
        deduped["topic"].tolist(),
        stance_tok, stance_model,
    )
    del stance_model, stance_tok
    _free()

    # 8 · Final filters
    before  = len(deduped)
    deduped = deduped[deduped["topic"] != "No Topic"]
    deduped = deduped[deduped["stance"] != "No Argument"]
    log.info("Final filter: %d → %d rows", before, len(deduped))

    # 9 · Format and save
    delivery = build_delivery_output(deduped)
    debug    = build_debug_output(deduped)

    delivery_path = OUTPUT_DIR / f"{dataset_name}_output.csv"
    debug_path    = OUTPUT_DIR / f"{dataset_name}_debug.csv"

    delivery.to_csv(delivery_path, index=False)
    debug.to_csv(debug_path, index=False)

    log.info("Delivery output → %s  (%d rows, %d cols)", delivery_path, len(delivery), len(delivery.columns))
    log.info("Debug output    → %s  (%d rows, %d cols)", debug_path,    len(debug),    len(debug.columns))
    return delivery


def main():
    parser = argparse.ArgumentParser(description="WIBA argument mining pipeline")
    parser.add_argument(
        "dataset",
        choices=["defense", "econ", "tech", "all"],
        help="Dataset to process. Use 'all' to run all three sequentially.",
    )
    parser.add_argument(
        "--no-4bit", action="store_true",
        help="Load LoRA base models in fp16 (requires more VRAM)",
    )
    args    = parser.parse_args()
    use_4bit = not args.no_4bit
    datasets = ["defense", "econ", "tech"] if args.dataset == "all" else [args.dataset]

    for name in datasets:
        log.info("===== Dataset: %s =====", name)
        run_pipeline(name, use_4bit=use_4bit)

    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()
