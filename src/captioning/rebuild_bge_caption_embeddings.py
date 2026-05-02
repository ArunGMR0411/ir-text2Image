"""
Step 24 — Rebuild BGE Caption Embeddings from augmented_captions_clean.jsonl
=============================================================================
Identical model / pooling / normalisation settings as the original
generate_bge_embeddings.py (BAAI/bge-large-en-v1.5, normalize_embeddings=False,
float32, SHARD_SIZE=10_000).

Key differences from the original:
  - Input: index/indexing/augmented_captions_clean.jsonl  (not augmented_captions.jsonl)
  - Outputs only the caption corpus (transcript is untouched)
  - Shard dir: embeddings/bge_caption_shards_rebuild/  (separate to avoid
    colliding with any residual shards from original run)
  - Overwrites embeddings/indexing/bge_caption_embeddings.npy and
    embeddings/indexing/bge_caption_index.jsonl on successful completion

Log: logs/rebuild_bge_captions_YYYYMMDD_HHMM.log
"""

from __future__ import annotations

import json
import logging
import math
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

MODEL_NAME    = "BAAI/bge-large-en-v1.5"
EMBEDDING_DIM = 1024
# write to disk every 10k rows so a crash doesn't lose everything
SHARD_SIZE    = 10_000

REPO_ROOT     = Path(__file__).resolve().parent.parent
INPUT_JSONL   = REPO_ROOT / "index" / "indexing" / "augmented_captions_clean.jsonl"
OUTPUT_NPY    = REPO_ROOT / "embeddings" / "bge_caption_embeddings.npy"
OUTPUT_IDX    = REPO_ROOT / "embeddings" / "bge_caption_index.jsonl"
SHARD_DIR     = REPO_ROOT / "embeddings" / "bge_caption_shards_rebuild"
LOGS_DIR      = REPO_ROOT / "logs"


def build_logger() -> tuple[logging.Logger, Path]:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M")
    log_path = LOGS_DIR / f"rebuild_bge_captions_{ts}.log"

    logger = logging.getLogger("rebuild_bge_captions")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s", "%Y-%m-%dT%H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger, log_path


def iter_caption_rows(path: Path):
    """Yield (row_index, frame_id, text_or_None) from JSONL."""
    with path.open("r", encoding="utf-8") as fh:
        for row_index, line in enumerate(fh):
            line = line.strip()
            if not line:
                raise ValueError(f"{path} line {row_index + 1} is blank")
            record = json.loads(line)
            frame_id = record["frame_id"]
            text = record.get("caption", "")
            text = text.strip() if isinstance(text, str) else ""
            # yield None for empty captions so we write a zero vector
            yield row_index, frame_id, text if text else None


def shard_paths(shard_index: int) -> tuple[Path, Path]:
    name = f"shard_{shard_index:06d}"
    return SHARD_DIR / f"{name}.npy", SHARD_DIR / f"{name}.jsonl"


def shard_complete(shard_index: int) -> bool:
    npy, jsonl = shard_paths(shard_index)
    return npy.is_file() and jsonl.is_file()


def save_shard(
    shard_index: int,
    rows: list[tuple[int, str, str | None]],
    model: SentenceTransformer,
    batch_size: int,
    logger: logging.Logger,
) -> tuple[int, int]:
    npy_path, jsonl_path = shard_paths(shard_index)

    # rows with no text get a zero vector — keeps the index row-aligned
    text_offsets = [(i, row) for i, row in enumerate(rows) if row[2] is not None]
    zero_count   = len(rows) - len(text_offsets)
    embeddings   = np.zeros((len(rows), EMBEDDING_DIM), dtype=np.float32)

    if text_offsets:
        texts   = [r[1][2] for r in text_offsets]
        encoded = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=False,
            convert_to_numpy=True,
        )
        encoded = np.asarray(encoded, dtype=np.float32)
        assert encoded.shape == (len(text_offsets), EMBEDDING_DIM), \
            f"shard {shard_index}: bad shape {encoded.shape}"
        for enc_idx, (row_off, _) in enumerate(text_offsets):
            embeddings[row_off] = encoded[enc_idx]

    np.save(npy_path, embeddings)
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for row_index, frame_id, _ in rows:
            fh.write(json.dumps({"frame_id": frame_id, "row_index": row_index}) + "\n")

    logger.info(
        "shard %d written: rows=%d embedded=%d zero_vectors=%d",
        shard_index, len(rows), len(text_offsets), zero_count,
    )
    return len(rows), zero_count


def merge_shards(total_rows: int, logger: logging.Logger) -> None:
    n_shards = math.ceil(total_rows / SHARD_SIZE)
    # open as memmap so we can write large arrays without loading them all into RAM
    merged = np.lib.format.open_memmap(
        OUTPUT_NPY, mode="w+", dtype=np.float32, shape=(total_rows, EMBEDDING_DIM)
    )
    with OUTPUT_IDX.open("w", encoding="utf-8") as idx_fh:
        row_cursor = 0
        for shard_index in range(n_shards):
            npy_path, jsonl_path = shard_paths(shard_index)
            rows_in_shard = min(SHARD_SIZE, total_rows - row_cursor)
            arr = np.load(npy_path, mmap_mode="r")
            assert arr.shape == (rows_in_shard, EMBEDDING_DIM), \
                f"shard {shard_index} shape mismatch: {arr.shape}"
            merged[row_cursor : row_cursor + rows_in_shard] = arr
            with jsonl_path.open("r", encoding="utf-8") as s_fh:
                shutil.copyfileobj(s_fh, idx_fh)
            row_cursor += rows_in_shard
            logger.info("merged shard %d rows=%d", shard_index, rows_in_shard)
    merged.flush()
    del merged
    logger.info("merge complete — OUTPUT_NPY written")


def main(test_limit: int = 0) -> int:
    logger, log_path = build_logger()
    logger.info("=== rebuild_bge_caption_embeddings START ===")
    logger.info("input  : %s", INPUT_JSONL)
    logger.info("output : %s", OUTPUT_NPY)
    logger.info("index  : %s", OUTPUT_IDX)
    logger.info("shards : %s", SHARD_DIR)
    logger.info("model  : %s  dim=%d", MODEL_NAME, EMBEDDING_DIM)
    if test_limit:
        logger.info("TEST MODE: processing first %d rows only", test_limit)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        total_mem_gib = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        # leave 8 GiB headroom for the model weights
        headroom = max(total_mem_gib - 8.0, 8.0)
        batch_size = max(32, min(int(headroom / 4.0 * 64), 512))
    else:
        batch_size = 32
    logger.info("device=%s  batch_size=%d", device, batch_size)

    logger.info("Loading model …")
    model = SentenceTransformer(MODEL_NAME, device=device)
    logger.info("Model loaded")

    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    total_rows = 0
    total_embedded = 0
    total_zeros = 0
    shards_written = 0

    buffer: list[tuple[int, str, str | None]] = []
    current_shard = 0

    for row_index, frame_id, text in iter_caption_rows(INPUT_JSONL):
        if test_limit and row_index >= test_limit:
            break
        shard_index = row_index // SHARD_SIZE
        if shard_index != current_shard:
            if buffer and not shard_complete(current_shard):
                rw, zc = save_shard(current_shard, buffer, model, batch_size, logger)
                total_rows += rw; total_zeros += zc; total_embedded += rw - zc
                shards_written += 1
            buffer = []
            current_shard = shard_index
        if not shard_complete(shard_index):
            buffer.append((row_index, frame_id, text))
        else:
            total_rows += 1

    if buffer:
        rw, zc = save_shard(current_shard, buffer, model, batch_size, logger)
        total_rows += rw; total_zeros += zc; total_embedded += rw - zc
        shards_written += 1

    elapsed = time.time() - t0
    logger.info(
        "Shard pass complete: rows=%d embedded=%d zero_vectors=%d shards_written=%d elapsed=%.1fs",
        total_rows, total_embedded, total_zeros, shards_written, elapsed,
    )

    if test_limit:
        logger.info("TEST MODE complete — not merging, not overwriting outputs")
        npy0, _ = shard_paths(0)
        arr = np.load(npy0)
        logger.info("Smoke-test shard 0 shape: %s  dtype=%s", arr.shape, arr.dtype)
        nan_count = int(np.isnan(arr).sum())
        inf_count = int(np.isinf(arr).sum())
        logger.info("NaN=%d  Inf=%d", nan_count, inf_count)
        if nan_count > 0 or inf_count > 0:
            logger.error("SMOKE TEST FAILED — NaN or Inf in test output")
            return 1
        logger.info("SMOKE TEST PASSED — shape=(%d, %d) NaN=0 Inf=0", arr.shape[0], arr.shape[1])
        shutil.rmtree(SHARD_DIR)
        logger.info("Smoke-test shard dir deleted")
        return 0

    real_total = sum(1 for _ in iter_caption_rows(INPUT_JSONL))
    logger.info("Total rows in input: %d", real_total)
    merge_shards(real_total, logger)

    arr = np.load(OUTPUT_NPY, mmap_mode="r")
    nan_count = int(np.isnan(arr).sum())
    inf_count = int(np.isinf(arr).sum())
    idx_lines = sum(1 for _ in OUTPUT_IDX.open())

    logger.info("=== FINAL VALIDATION ===")
    logger.info("OUTPUT_NPY shape : %s  (expected (%d, %d))", arr.shape, real_total, EMBEDDING_DIM)
    logger.info("OUTPUT_NPY dtype : %s", arr.dtype)
    logger.info("NaN count        : %d  (expected 0)", nan_count)
    logger.info("Inf count        : %d  (expected 0)", inf_count)
    logger.info("IDX line count   : %d  (expected %d)", idx_lines, real_total)

    checks_ok = (
        arr.shape == (real_total, EMBEDDING_DIM)
        and arr.dtype == np.float32
        and nan_count == 0
        and inf_count == 0
        and idx_lines == real_total
    )
    if not checks_ok:
        logger.error("VALIDATION FAILED — see above for details")
        return 1

    logger.info("ALL VALIDATION CHECKS PASSED")
    logger.info("Total elapsed: %.1fs", time.time() - t0)
    logger.info("log written to %s", log_path)

    shutil.rmtree(SHARD_DIR)
    logger.info("Shard dir deleted: %s", SHARD_DIR)
    logger.info("=== rebuild_bge_caption_embeddings COMPLETE ===")
    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Rebuild BGE caption embeddings from clean JSONL")
    parser.add_argument("--test_limit", type=int, default=0,
                        help="Process only first N rows (0 = full run)")
    args = parser.parse_args()
    sys.exit(main(test_limit=args.test_limit))
