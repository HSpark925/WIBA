"""Small subset test — verifies model loading, end-to-end flow, and output schema.

Runs on the first N speeches of each dataset before the full OSC job.
Usage:
    python test_pipeline.py            # tests all three datasets, N=5
    python test_pipeline.py --n 10     # larger subset
    python test_pipeline.py --dataset defense  # single dataset
"""

import argparse
import sys
from pathlib import Path

# Import shared pipeline components
sys.path.insert(0, str(Path(__file__).parent))
from wiba_pipeline import (
    REQUIRED_COLS,
    DATA_DIR, OUTPUT_DIR,
    DETECT_THRESHOLD, WINDOW_SIZE, STEP_SIZE,
    load_rdata, find_text_column, create_segments,
    remove_overlapping,
    load_detect_model, load_extract_model, load_stance_model,
    run_detect, run_extract, run_stance,
    build_delivery_output, build_debug_output, _free,
)

import logging
import pandas as pd

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

REQUIRED_COLS_SET = set(REQUIRED_COLS)


def validate_schema(df: pd.DataFrame, dataset_name: str):
    """Check that all required columns are present and types are sensible."""
    missing = REQUIRED_COLS_SET - set(df.columns)
    if missing:
        log.error("[%s] MISSING required columns: %s", dataset_name, missing)
        return False

    checks = [
        ("rownum_hs",           lambda c: df[c].notna().all(),       "has nulls"),
        ("text_segment",        lambda c: df[c].str.len().gt(0).all(),"empty strings"),
        ("argument_prediction", lambda c: df[c].eq("Argument").all(), "unexpected values"),
        ("confidence",          lambda c: df[c].between(0, 1).all(),  "outside [0,1]"),
        ("topic",               lambda c: df[c].notna().all(),        "has nulls"),
        ("stance",              lambda c: df[c].isin(
                                    ["Argument in Favor", "Argument Against"]).all(),
                                                                       "unexpected values"),
    ]
    passed = True
    for col, check_fn, msg in checks:
        if not check_fn(col):
            log.warning("[%s] Column '%s': %s", dataset_name, col, msg)
            passed = False

    return passed


def run_test(dataset_name: str, n: int, models: dict) -> pd.DataFrame | None:
    log.info("=" * 55)
    log.info("TEST  dataset=%-10s  subset=%d speeches", dataset_name, n)
    log.info("=" * 55)

    # Load and truncate
    for suffix in ("Rdata", "RData", "rdata"):
        rdata_path = DATA_DIR / f"df_{dataset_name}.{suffix}"
        if rdata_path.exists():
            break
    else:
        log.error("Dataset not found: df_%s.*", dataset_name)
        return None

    df       = load_rdata(str(rdata_path)).head(n)
    text_col = find_text_column(df)
    log.info("Text column detected: '%s'", text_col)

    rename_map = {}
    if "topic"      in df.columns: rename_map["topic"]      = "hearing_topic"
    if "topic_name" in df.columns: rename_map["topic_name"] = "hearing_topic_name"
    if rename_map:
        df = df.rename(columns=rename_map)

    # Step 1 · Segment
    segments = create_segments(df, text_col, window=WINDOW_SIZE, step=STEP_SIZE)
    log.info("Segments created: %d", len(segments))

    # Step 2 · Detect
    detect_tok, detect_model = models["detect"]
    scores = run_detect(segments["text_segment"].tolist(), detect_tok, detect_model, batch_size=2)
    segments["confidence"]          = scores
    segments["argument_prediction"] = [
        "Argument" if s >= DETECT_THRESHOLD else "No Argument" for s in scores
    ]
    above = segments[segments["argument_prediction"] == "Argument"].copy()
    log.info("After detect (>=%.1f): %d / %d", DETECT_THRESHOLD, len(above), len(segments))

    if above.empty:
        log.warning("No argumentative segments found in this subset — try larger --n")
        return None

    # Step 3 · Overlap removal
    deduped = remove_overlapping(above, window=WINDOW_SIZE)

    # Step 4 · Extract
    extract_tok, extract_model = models["extract"]
    deduped = deduped.copy()
    deduped["topic"] = run_extract(deduped["text_segment"].tolist(), extract_tok, extract_model)

    # Step 5 · Stance
    stance_tok, stance_model = models["stance"]
    deduped["stance"] = run_stance(
        deduped["text_segment"].tolist(),
        deduped["topic"].tolist(),
        stance_tok, stance_model,
        batch_size=2,
    )

    # Step 6 · Final filters
    before  = len(deduped)
    deduped = deduped[deduped["topic"] != "No Topic"]
    deduped = deduped[deduped["stance"] != "No Argument"]
    log.info("After final filters: %d / %d rows", len(deduped), before)

    # Build outputs
    delivery = build_delivery_output(deduped)
    debug    = build_debug_output(deduped)

    # Schema validation (on delivery output)
    log.info("----- Schema validation -----")
    ok = validate_schema(delivery, dataset_name)
    log.info("Schema: %s", "PASS" if ok else "FAIL — see warnings above")

    # Sample delivery output
    log.info("----- Delivery output sample (required columns) -----")
    print(delivery.to_string(max_colwidth=60))

    log.info("----- Debug output columns -----")
    log.info("%s", list(debug.columns))

    # Save both test outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    delivery.to_csv(OUTPUT_DIR / f"{dataset_name}_test_output.csv",       index=False)
    debug.to_csv(   OUTPUT_DIR / f"{dataset_name}_test_debug.csv",  index=False)
    log.info("Saved → %s_test_output.csv  +  %s_test_debug.csv", dataset_name, dataset_name)

    return delivery


def main():
    parser = argparse.ArgumentParser(description="WIBA subset test")
    parser.add_argument("--dataset", choices=["defense", "econ", "tech", "all"], default="all")
    parser.add_argument("--n",       type=int, default=5, help="Number of speeches to test")
    parser.add_argument("--no-4bit", action="store_true")
    args     = parser.parse_args()
    use_4bit = not args.no_4bit
    datasets = ["defense", "econ", "tech"] if args.dataset == "all" else [args.dataset]

    # Load models once and share across datasets
    log.info("Loading models (shared across all test datasets)...")
    models = {
        "detect":  load_detect_model(use_4bit=use_4bit),
        "extract": load_extract_model(),
        "stance":  load_stance_model(use_4bit=use_4bit),
    }

    results = {}
    for name in datasets:
        out = run_test(name, args.n, models)
        results[name] = "PASS" if out is not None and len(out) > 0 else "FAIL"

    log.info("=" * 55)
    log.info("TEST SUMMARY")
    for name, status in results.items():
        log.info("  %-10s  %s", name, status)
    log.info("=" * 55)


if __name__ == "__main__":
    main()
