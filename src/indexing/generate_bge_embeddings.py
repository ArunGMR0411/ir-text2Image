"""
Step 18: Generate BGE embeddings for transcripts and augmented captions.

This script processes two corpora in one invocation:
1. index/indexing/transcript_cleaned.csv -> embeddings/indexing/bge_transcript_embeddings.npy
2. index/augmented_captions.jsonl -> embeddings/indexing/bge_caption_embeddings.npy

Key guarantees:
- Streaming readers; no full-corpus text loading
- Row-aligned outputs with zero vectors for empty/null text
- Shard checkpointing every 10,000 rows/records
- Resume-safe shard detection
- Final merge into contiguous float32 .npy arrays
- Final shape validation by reading arrays back
- Shard cleanup only after successful merge and validation
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-large-en-v1.5"
EMBEDDING_DIM = 1024
# write to disk every 10k rows so a crash doesn't lose everything
SHARD_SIZE = 10_000
EXPECTED_TRANSCRIPT_ROWS = 145_140
EXPECTED_CAPTION_ROWS = 244_966
FRAME_ID_PATTERN = "__day"


class ValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class CorpusConfig:
    name: str
    expected_rows: int
    source_path: Path
    source_kind: str
    text_field: str
    output_embeddings_path: Path
    output_index_path: Path
    shard_dir: Path


@dataclass
class ShardStats:
    rows_processed: int = 0
    zero_vectors_written: int = 0
    text_rows_embedded: int = 0
    shards_written: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate BGE embeddings for transcripts and captions.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Repository root containing index/, embeddings/, and logs/.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional override for SentenceTransformer device, e.g. cuda or cpu.",
    )
    return parser.parse_args()


def build_logger(script_path: Path, repo_root: Path) -> tuple[logging.Logger, Path]:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M")
    logs_dir = repo_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{script_path.stem}_{timestamp}.log"

    logger = logging.getLogger(script_path.stem)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)sZ %(levelname)s %(message)s", "%Y-%m-%dT%H:%M:%S")
    )
    logger.addHandler(handler)
    return logger, log_path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def choose_device(device_override: str | None) -> str:
    if device_override:
        return device_override
    return "cuda" if torch.cuda.is_available() else "cpu"


def derive_batch_size(device: str, embedding_dim: int) -> int:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return max(8, embedding_dim // 32)

    device_index = torch.device(device).index
    if device_index is None:
        device_index = torch.cuda.current_device()
    total_memory_bytes = torch.cuda.get_device_properties(device_index).total_memory
    total_memory_gib = total_memory_bytes / float(1024**3)

    # leave 8 GiB headroom for the model weights
    headroom_gib = max(total_memory_gib - 8.0, 8.0)
    scale_units = max(1.0, headroom_gib / 4.0)
    dim_factor = max(1.0, 1024.0 / float(embedding_dim))
    raw_batch_size = int(scale_units * 64 * dim_factor)

    return max(32, min(raw_batch_size, 512))


def normalize_text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None
    raise ValidationError(f"text field must be a string or null, got {type(value).__name__}")


def validate_frame_id(frame_id: object, context: str) -> str:
    require(isinstance(frame_id, str) and frame_id, f"{context}: frame_id must be a non-empty string")
    require(frame_id == frame_id.lower(), f"{context}: frame_id must be lowercase")
    parts = frame_id.split("__")
    require(len(parts) == 3, f"{context}: frame_id must have three double-underscore segments")
    stream_name, day, hh_nnnn = parts
    require(stream_name, f"{context}: missing stream_name segment")
    require(day.startswith("day") and day[3:].isdigit(), f"{context}: invalid day segment '{day}'")
    hh_part, sep, frame_part = hh_nnnn.partition("_")
    require(sep == "_", f"{context}: missing hour/frame separator")
    require(hh_part.isdigit() and len(hh_part) == 2, f"{context}: invalid hour segment '{hh_part}'")
    require(frame_part.isdigit() and len(frame_part) == 4, f"{context}: invalid frame segment '{frame_part}'")
    return frame_id


def iter_transcript_rows(path: Path, text_field: str) -> Iterator[tuple[int, str, str | None]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None, f"{path} is missing a CSV header")
        require("frame_id" in reader.fieldnames, f"{path} is missing frame_id column")
        require(text_field in reader.fieldnames, f"{path} is missing {text_field} column")
        for row_index, row in enumerate(reader):
            frame_id = validate_frame_id(row.get("frame_id"), f"{path} row {row_index}")
            text_value = normalize_text(row.get(text_field))
            yield row_index, frame_id, text_value


def iter_caption_rows(path: Path, text_field: str) -> Iterator[tuple[int, str, str | None]]:
    with path.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            text = line.strip()
            require(text, f"{path} line {row_index + 1} is blank")
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValidationError(f"{path} line {row_index + 1} has invalid JSON: {exc}") from exc
            require(isinstance(record, dict), f"{path} line {row_index + 1} is not a JSON object")
            require("frame_id" in record, f"{path} line {row_index + 1} missing frame_id")
            require(text_field in record, f"{path} line {row_index + 1} missing {text_field}")
            frame_id = validate_frame_id(record["frame_id"], f"{path} line {row_index + 1}")
            text_value = normalize_text(record[text_field])
            yield row_index, frame_id, text_value


def iter_corpus_rows(config: CorpusConfig) -> Iterator[tuple[int, str, str | None]]:
    if config.source_kind == "csv":
        yield from iter_transcript_rows(config.source_path, config.text_field)
        return
    if config.source_kind == "jsonl":
        yield from iter_caption_rows(config.source_path, config.text_field)
        return
    raise ValidationError(f"unsupported source_kind: {config.source_kind}")


def shard_paths(shard_dir: Path, shard_index: int) -> tuple[Path, Path]:
    shard_name = f"shard_{shard_index:06d}"
    return shard_dir / f"{shard_name}.npy", shard_dir / f"{shard_name}.jsonl"


def expected_shard_count(expected_rows: int) -> int:
    return math.ceil(expected_rows / SHARD_SIZE)


def existing_complete_shards(config: CorpusConfig) -> list[int]:
    complete = []
    for shard_index in range(expected_shard_count(config.expected_rows)):
        npy_path, jsonl_path = shard_paths(config.shard_dir, shard_index)
        if npy_path.exists() and jsonl_path.exists():
            complete.append(shard_index)
        elif npy_path.exists() != jsonl_path.exists():
            # one file without the other means the shard write was interrupted
            raise ValidationError(
                f"{config.name}: incomplete shard pair for shard {shard_index}: "
                f"{npy_path.name}, {jsonl_path.name}"
            )
    return complete


def validate_final_outputs(config: CorpusConfig) -> bool:
    if not config.output_embeddings_path.is_file() or not config.output_index_path.is_file():
        return False

    try:
        array = np.load(config.output_embeddings_path, mmap_mode="r")
    except Exception as exc:
        raise ValidationError(f"{config.name}: failed to read final embeddings: {exc}") from exc

    if tuple(array.shape) != (config.expected_rows, EMBEDDING_DIM):
        return False
    if array.dtype != np.float32:
        return False

    index_rows = 0
    with config.output_index_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                return False
            record = json.loads(text)
            if not isinstance(record, dict):
                return False
            if set(record.keys()) != {"frame_id", "row_index"}:
                return False
            validate_frame_id(record["frame_id"], f"{config.output_index_path} line {line_number}")
            require(record["row_index"] == index_rows, f"{config.name}: index row mismatch at line {line_number}")
            index_rows += 1
    return index_rows == config.expected_rows


def save_shard(
    config: CorpusConfig,
    shard_index: int,
    rows: Sequence[tuple[int, str, str | None]],
    model: SentenceTransformer,
    batch_size: int,
    logger: logging.Logger,
) -> tuple[int, int]:
    require(rows, f"{config.name}: attempted to save an empty shard")
    shard_npy_path, shard_index_path = shard_paths(config.shard_dir, shard_index)
    if shard_npy_path.exists() and shard_index_path.exists():
        logger.info("%s shard %d already exists; skipping rewrite", config.name, shard_index)
        zero_count = sum(1 for _, _, text in rows if text is None)
        return len(rows), zero_count

    text_rows = [(offset, row) for offset, row in enumerate(rows) if row[2] is not None]
    # rows with no text get a zero vector to keep the index row-aligned
    zero_count = len(rows) - len(text_rows)
    embeddings = np.zeros((len(rows), EMBEDDING_DIM), dtype=np.float32)

    if text_rows:
        texts = [row[1][2] for row in text_rows]
        encoded = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=False,
            convert_to_numpy=True,
        )
        encoded = np.asarray(encoded, dtype=np.float32)
        require(
            encoded.shape == (len(text_rows), EMBEDDING_DIM),
            f"{config.name} shard {shard_index}: unexpected encoded shape {encoded.shape}",
        )
        for encoded_offset, (row_offset, _) in enumerate(text_rows):
            embeddings[row_offset] = encoded[encoded_offset]

    np.save(shard_npy_path, embeddings.astype(np.float32, copy=False))
    with shard_index_path.open("w", encoding="utf-8") as handle:
        for row_index, frame_id, _ in rows:
            handle.write(json.dumps({"frame_id": frame_id, "row_index": row_index}, ensure_ascii=True) + "\n")

    logger.info(
        "%s shard %d written: rows=%d embedded=%d zero_vectors=%d",
        config.name,
        shard_index,
        len(rows),
        len(text_rows),
        zero_count,
    )
    return len(rows), zero_count


def build_missing_shards(
    config: CorpusConfig,
    model: SentenceTransformer,
    batch_size: int,
    logger: logging.Logger,
) -> ShardStats:
    config.shard_dir.mkdir(parents=True, exist_ok=True)
    stats = ShardStats()
    completed = set(existing_complete_shards(config))
    if completed:
        logger.info("%s detected %d existing shard(s)", config.name, len(completed))

    buffer: list[tuple[int, str, str | None]] = []
    current_shard_index = 0

    for row_index, frame_id, text_value in iter_corpus_rows(config):
        shard_index = row_index // SHARD_SIZE
        if shard_index != current_shard_index:
            if buffer and current_shard_index not in completed:
                rows_written, zero_count = save_shard(
                    config=config,
                    shard_index=current_shard_index,
                    rows=buffer,
                    model=model,
                    batch_size=batch_size,
                    logger=logger,
                )
                stats.shards_written += 1
                stats.rows_processed += rows_written
                stats.zero_vectors_written += zero_count
                stats.text_rows_embedded += rows_written - zero_count
            buffer = []
            current_shard_index = shard_index

        if shard_index in completed:
            continue

        buffer.append((row_index, frame_id, text_value))

    if buffer:
        rows_written, zero_count = save_shard(
            config=config,
            shard_index=current_shard_index,
            rows=buffer,
            model=model,
            batch_size=batch_size,
            logger=logger,
        )
        stats.shards_written += 1
        stats.rows_processed += rows_written
        stats.zero_vectors_written += zero_count
        stats.text_rows_embedded += rows_written - zero_count

    return stats


def validate_shard_pair(
    config: CorpusConfig,
    shard_index: int,
    start_row: int,
    expected_rows_in_shard: int,
) -> None:
    shard_npy_path, shard_index_path = shard_paths(config.shard_dir, shard_index)
    require(shard_npy_path.is_file(), f"{config.name}: missing shard array {shard_npy_path}")
    require(shard_index_path.is_file(), f"{config.name}: missing shard index {shard_index_path}")

    array = np.load(shard_npy_path, mmap_mode="r")
    require(
        tuple(array.shape) == (expected_rows_in_shard, EMBEDDING_DIM),
        f"{config.name}: shard {shard_index} has shape {array.shape}, expected {(expected_rows_in_shard, EMBEDDING_DIM)}",
    )
    require(array.dtype == np.float32, f"{config.name}: shard {shard_index} dtype {array.dtype}, expected float32")

    with shard_index_path.open("r", encoding="utf-8") as handle:
        index_rows = 0
        for index_rows, line in enumerate(handle, start=1):
            record = json.loads(line)
            require(set(record.keys()) == {"frame_id", "row_index"}, f"{config.name}: invalid shard index record")
            validate_frame_id(record["frame_id"], f"{shard_index_path} line {index_rows}")
            expected_row_index = start_row + index_rows - 1
            require(
                record["row_index"] == expected_row_index,
                f"{config.name}: shard {shard_index} row_index mismatch at line {index_rows}",
            )
    require(index_rows == expected_rows_in_shard, f"{config.name}: shard {shard_index} index row count mismatch")


def merge_shards(config: CorpusConfig, logger: logging.Logger) -> None:
    require(config.shard_dir.is_dir(), f"{config.name}: missing shard directory {config.shard_dir}")

    with config.output_index_path.open("w", encoding="utf-8") as index_handle:
        # open as memmap so we can write large arrays without loading them all into RAM
        merged = np.lib.format.open_memmap(
            config.output_embeddings_path,
            mode="w+",
            dtype=np.float32,
            shape=(config.expected_rows, EMBEDDING_DIM),
        )
        row_cursor = 0
        for shard_index in range(expected_shard_count(config.expected_rows)):
            rows_in_shard = min(SHARD_SIZE, config.expected_rows - row_cursor)
            validate_shard_pair(config, shard_index, row_cursor, rows_in_shard)
            shard_npy_path, shard_index_path = shard_paths(config.shard_dir, shard_index)
            shard_array = np.load(shard_npy_path, mmap_mode="r")
            merged[row_cursor : row_cursor + rows_in_shard] = shard_array
            with shard_index_path.open("r", encoding="utf-8") as shard_index_handle:
                shutil.copyfileobj(shard_index_handle, index_handle)
            row_cursor += rows_in_shard
            logger.info("%s merged shard %d rows=%d", config.name, shard_index, rows_in_shard)

        merged.flush()
        del merged

    logger.info("%s final outputs written", config.name)


def validate_final_shape(config: CorpusConfig) -> None:
    array = np.load(config.output_embeddings_path, mmap_mode="r")
    require(
        tuple(array.shape) == (config.expected_rows, EMBEDDING_DIM),
        f"{config.name}: final array shape {array.shape}, expected {(config.expected_rows, EMBEDDING_DIM)}",
    )
    require(array.dtype == np.float32, f"{config.name}: final array dtype {array.dtype}, expected float32")


def cleanup_shards(config: CorpusConfig, logger: logging.Logger) -> None:
    if config.shard_dir.exists():
        shutil.rmtree(config.shard_dir)
        logger.info("%s cleaned shard directory %s", config.name, config.shard_dir)


def process_corpus(
    config: CorpusConfig,
    model: SentenceTransformer,
    batch_size: int,
    logger: logging.Logger,
) -> None:
    logger.info("Starting corpus %s", config.name)
    logger.info("source=%s", config.source_path)
    logger.info("output_embeddings=%s", config.output_embeddings_path)
    logger.info("output_index=%s", config.output_index_path)
    logger.info("shard_dir=%s", config.shard_dir)

    require(config.source_path.is_file(), f"{config.name}: missing source file {config.source_path}")
    config.output_embeddings_path.parent.mkdir(parents=True, exist_ok=True)

    if validate_final_outputs(config):
        logger.info("%s final outputs already valid; skipping corpus generation", config.name)
        cleanup_shards(config, logger)
        return

    stats = build_missing_shards(config, model, batch_size, logger)
    logger.info(
        "%s shard generation summary: rows_processed=%d text_rows_embedded=%d zero_vectors=%d shards_written=%d",
        config.name,
        stats.rows_processed,
        stats.text_rows_embedded,
        stats.zero_vectors_written,
        stats.shards_written,
    )

    merge_shards(config, logger)
    validate_final_shape(config)
    require(validate_final_outputs(config), f"{config.name}: final outputs failed validation")
    cleanup_shards(config, logger)
    logger.info("%s complete", config.name)


def build_configs(repo_root: Path) -> list[CorpusConfig]:
    embeddings_dir = repo_root / "embeddings"
    index_dir = repo_root / "index"
    return [
        CorpusConfig(
            name="transcript",
            expected_rows=EXPECTED_TRANSCRIPT_ROWS,
            source_path=index_dir / "transcript_cleaned.csv",
            source_kind="csv",
            text_field="transcript_text",
            output_embeddings_path=embeddings_dir / "bge_transcript_embeddings.npy",
            output_index_path=embeddings_dir / "bge_transcript_index.jsonl",
            shard_dir=embeddings_dir / "bge_transcript_shards",
        ),
        CorpusConfig(
            name="caption",
            expected_rows=EXPECTED_CAPTION_ROWS,
            source_path=index_dir / "augmented_captions.jsonl",
            source_kind="jsonl",
            text_field="caption",
            output_embeddings_path=embeddings_dir / "bge_caption_embeddings.npy",
            output_index_path=embeddings_dir / "bge_caption_index.jsonl",
            shard_dir=embeddings_dir / "bge_caption_shards",
        ),
    ]


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    script_path = Path(__file__).resolve()
    logger, log_path = build_logger(script_path, repo_root)

    try:
        device = choose_device(args.device)
        batch_size = derive_batch_size(device, EMBEDDING_DIM)
        logger.info("Loading model %s on device=%s batch_size=%d", MODEL_NAME, device, batch_size)
        model = SentenceTransformer(MODEL_NAME, device=device)

        for config in build_configs(repo_root):
            process_corpus(config, model, batch_size, logger)

        logger.info("All corpus embedding jobs completed successfully")
        logger.info("log written to %s", log_path)
        return 0
    except ValidationError as exc:
        logger.error("generate_bge_embeddings failed: %s", exc)
        logger.error("log written to %s", log_path)
        return 1
    except Exception as exc:
        logger.exception("Unexpected failure: %s", exc)
        logger.error("log written to %s", log_path)
        return 1


if __name__ == "__main__":
    sys.exit(main())
