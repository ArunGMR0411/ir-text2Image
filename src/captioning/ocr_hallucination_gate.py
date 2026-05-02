"""
Step 23 — SigLIP2 OCR Hallucination Gate
=========================================
For each record in index/augmented_captions.jsonl that has a non-empty
`ocr_text`, we:
  1. Encode the OCR string via the SigLIP2 text encoder → 1152-dim vector
  2. Compute cosine similarity against the frame's pre-existing visual
     embedding from embeddings/indexing/siglip2_embeddings.npy
  3. If similarity < THRESHOLD → strip the OCR suffix from caption and
     zero-out ocr_text (mark as hallucination)

Outputs
-------
  index/indexing/augmented_captions_clean.jsonl  — filtered captions
  index/captioning/ocr_filter_report.json          — summary stats
  logs/ocr_filter_YYYYMMDD_HHMM.log    — run log
"""

import os
import sys
import json
import math
import time
import logging
import argparse
import datetime
import numpy as np
import torch
from transformers import GemmaTokenizer, AutoModel

MODEL_ID         = "google/siglip2-so400m-patch14-384"
EMBEDDING_DIM    = 1152
# OCR text with similarity below this is likely not visible in the frame
THRESHOLD        = 0.20
TEXT_MAX_LEN     = 64
TEXT_BATCH_SIZE  = 512

INPUT_CAPTIONS   = "index/augmented_captions.jsonl"
VISUAL_EMB_NPY   = "embeddings/indexing/siglip2_embeddings.npy"
VISUAL_IDX_JSONL = "embeddings/indexing/siglip2_index.jsonl"
OUTPUT_CAPTIONS  = "index/indexing/augmented_captions_clean.jsonl"
OUTPUT_REPORT    = "index/captioning/ocr_filter_report.json"
LOGS_DIR         = "logs"



def setup_logging(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    log_path = os.path.join(log_dir, f"ocr_filter_{ts}.log")

    logger = logging.getLogger("ocr_gate")
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Log file: {log_path}")
    return logger


def load_visual_index(jsonl_path: str, logger: logging.Logger):
    """Return {frame_id → row_index} mapping."""
    logger.info(f"Loading visual index from {jsonl_path}")
    mapping = {}
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            mapping[rec["frame_id"]] = rec["manifest_row_index"]
    logger.info(f"Visual index loaded: {len(mapping):,} entries")
    return mapping


def load_text_model(device: str, logger: logging.Logger):
    logger.info(f"Loading GemmaTokenizer + SigLIP2 text encoder on {device}...")
    tokenizer = GemmaTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32
    )
    model.eval()
    model.to(device)
    logger.info("Text encoder ready.")
    return tokenizer, model


@torch.no_grad()
def encode_texts(texts: list[str], tokenizer, model, device: str) -> np.ndarray:
    """Return L2-normalised text embeddings, shape (N, 1152), dtype float32."""
    inputs = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=TEXT_MAX_LEN,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    feats = model.get_text_features(**inputs)
    feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
    return feats.cpu().to(torch.float32).numpy()


def strip_ocr_from_caption(caption: str, ocr_text: str) -> str:
    """
    Remove the OCR suffix from the caption.

    Florence-2 embeds OCR tokens directly into the caption text, e.g.:
      caption = "A room with a couch. Coca-Cola"
      ocr_text = "Coca-Cola"
    Strategy: strip from the right — find last occurrence of ocr_text in
    caption (case-insensitive) and remove everything from that point onward,
    then right-strip punctuation/whitespace.
    """
    if not ocr_text:
        return caption

    idx = caption.lower().rfind(ocr_text.lower())
    if idx == -1:
        return caption

    stripped = caption[:idx].rstrip(" .,;:-|")
    return stripped if stripped else caption



def main(args):
    logger = setup_logging(LOGS_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}  |  Threshold: {args.threshold}")

    # memory-map the embeddings so we don't load 1.8 GB into RAM at once
    logger.info(f"Memory-mapping {VISUAL_EMB_NPY} ...")
    visual_emb = np.load(VISUAL_EMB_NPY, mmap_mode="r")
    logger.info(f"Visual embeddings shape: {visual_emb.shape}")

    frame_id_to_row = load_visual_index(VISUAL_IDX_JSONL, logger)

    logger.info(f"Loading {INPUT_CAPTIONS} ...")
    records = []
    with open(INPUT_CAPTIONS) as f:
        for line in f:
            records.append(json.loads(line))
    logger.info(f"Total records: {len(records):,}")

    # split into two groups so we only run the model on records that need it
    ocr_records   = [(i, r) for i, r in enumerate(records) if r.get("ocr_text", "").strip()]
    no_ocr_records = [(i, r) for i, r in enumerate(records) if not r.get("ocr_text", "").strip()]
    logger.info(f"Records with OCR text: {len(ocr_records):,}")
    logger.info(f"Records without OCR text: {len(no_ocr_records):,}")

    if args.test_limit > 0:
        logger.info(f"TEST MODE — limiting OCR records to {args.test_limit}")
        ocr_records = ocr_records[:args.test_limit]

    tokenizer, model = load_text_model(device, logger)

    n_ocr = len(ocr_records)
    n_batches = math.ceil(n_ocr / TEXT_BATCH_SIZE)

    stripped_count  = 0
    kept_count      = 0
    skipped_missing = 0
    all_sims        = []

    t0 = time.time()
    logger.info(f"Processing {n_ocr:,} OCR records in {n_batches} batch(es) ...")

    for b_idx in range(n_batches):
        batch_slice = ocr_records[b_idx * TEXT_BATCH_SIZE : (b_idx + 1) * TEXT_BATCH_SIZE]

        texts       = [r["ocr_text"] for _, r in batch_slice]
        text_embs   = encode_texts(texts, tokenizer, model, device)

        for local_i, (global_i, rec) in enumerate(batch_slice):
            fid = rec["frame_id"]
            row = frame_id_to_row.get(fid)

            if row is None:
                # frame has no visual embedding — can't run the gate
                logger.debug(f"frame_id not in visual index: {fid} — skipping gate")
                skipped_missing += 1
                kept_count += 1
                continue

            vis_vec = visual_emb[row].astype(np.float32)
            vis_norm = np.linalg.norm(vis_vec)
            if vis_norm > 0:
                vis_vec = vis_vec / vis_norm

            txt_vec = text_embs[local_i]
            sim = float(np.dot(vis_vec, txt_vec))
            all_sims.append(sim)

            if sim < args.threshold:
                # OCR text doesn't match what's visually in the frame
                records[global_i]["caption"] = strip_ocr_from_caption(
                    rec["caption"], rec["ocr_text"]
                )
                records[global_i]["ocr_text"] = ""
                records[global_i]["ocr_sim"]   = round(sim, 6)
                records[global_i]["ocr_gate"]  = "stripped"
                stripped_count += 1
                logger.debug(
                    f"STRIP  sim={sim:.4f} | {fid} | ocr='{rec['ocr_text']}'"
                )
            else:
                records[global_i]["ocr_sim"]  = round(sim, 6)
                records[global_i]["ocr_gate"] = "kept"
                kept_count += 1
                logger.debug(
                    f"KEEP   sim={sim:.4f} | {fid} | ocr='{rec['ocr_text']}'"
                )

        if (b_idx + 1) % 10 == 0 or (b_idx + 1) == n_batches:
            elapsed = time.time() - t0
            done = (b_idx + 1) * TEXT_BATCH_SIZE
            pct = min(100, 100 * done / n_ocr)
            logger.info(
                f"Batch {b_idx+1}/{n_batches} | {pct:.1f}% done | "
                f"stripped={stripped_count:,} kept={kept_count:,} | {elapsed:.1f}s elapsed"
            )

    logger.info(f"Writing {OUTPUT_CAPTIONS} ...")
    with open(OUTPUT_CAPTIONS, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    logger.info(f"Wrote {len(records):,} records to {OUTPUT_CAPTIONS}")

    sims_arr = np.array(all_sims, dtype=np.float32) if all_sims else np.array([])

    report = {
        "run_timestamp"    : datetime.datetime.now().isoformat(),
        "model"            : MODEL_ID,
        "threshold"        : args.threshold,
        "total_records"    : len(records),
        "ocr_records"      : n_ocr,
        "no_ocr_records"   : len(no_ocr_records),
        "stripped"         : stripped_count,
        "kept"             : kept_count,
        "skipped_missing"  : skipped_missing,
        "strip_rate_pct"   : round(100 * stripped_count / n_ocr, 2) if n_ocr else 0,
        "similarity_stats" : {
            "min"  : round(float(sims_arr.min()),  6) if sims_arr.size else None,
            "max"  : round(float(sims_arr.max()),  6) if sims_arr.size else None,
            "mean" : round(float(sims_arr.mean()), 6) if sims_arr.size else None,
            "std"  : round(float(sims_arr.std()),  6) if sims_arr.size else None,
            "p10"  : round(float(np.percentile(sims_arr, 10)), 6) if sims_arr.size else None,
            "p25"  : round(float(np.percentile(sims_arr, 25)), 6) if sims_arr.size else None,
            "p50"  : round(float(np.percentile(sims_arr, 50)), 6) if sims_arr.size else None,
            "p75"  : round(float(np.percentile(sims_arr, 75)), 6) if sims_arr.size else None,
            "p90"  : round(float(np.percentile(sims_arr, 90)), 6) if sims_arr.size else None,
        },
        "elapsed_seconds"  : round(time.time() - t0, 1),
    }

    with open(OUTPUT_REPORT, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("=" * 60)
    logger.info("OCR HALLUCINATION GATE — COMPLETE")
    logger.info(f"  Total records    : {len(records):,}")
    logger.info(f"  OCR records      : {n_ocr:,}")
    logger.info(f"  Stripped         : {stripped_count:,}  ({report['strip_rate_pct']}%)")
    logger.info(f"  Kept             : {kept_count:,}")
    logger.info(f"  Skipped (no vis) : {skipped_missing:,}")
    if sims_arr.size:
        logger.info(f"  Similarity  min={report['similarity_stats']['min']:.4f}  "
                    f"mean={report['similarity_stats']['mean']:.4f}  "
                    f"max={report['similarity_stats']['max']:.4f}")
    logger.info(f"  Report → {OUTPUT_REPORT}")
    logger.info(f"  Clean captions → {OUTPUT_CAPTIONS}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 23: SigLIP2 OCR Hallucination Gate"
    )
    parser.add_argument(
        "--threshold", type=float, default=THRESHOLD,
        help=f"Cosine similarity threshold (default: {THRESHOLD})"
    )
    parser.add_argument(
        "--test_limit", type=int, default=0,
        help="Process only first N OCR records (0 = full run)"
    )
    args = parser.parse_args()
    main(args)
