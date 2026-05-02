"""
Step 22: Populate FAISS indexes from embedding arrays.

This script:
- Loads and validates the three embedding arrays
- Applies in-place L2 normalization so inner-product search becomes cosine similarity
- Loads the empty FAISS IndexFlatIP indexes from Step 21 and populates them
- Builds the visual row-alignment map from embeddings/indexing/siglip2_index.jsonl
- Saves populated indexes back to disk and reloads them to confirm persistence

Architectural note:
- The visual row map is built only for the SigLIP2 index because it spans the
  full 416,542-frame universe.
- The transcript and caption indexes intentionally keep separate row-to-frame_id
  mappings via embeddings/indexing/bge_transcript_index.jsonl and
  embeddings/indexing/bge_caption_index.jsonl because they cover different frame
  universes and are not row-aligned to the visual index.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np


EXPECTED_VISUAL_SHAPE = (416_542, 1152)
EXPECTED_TRANSCRIPT_SHAPE = (145_140, 1024)
EXPECTED_CAPTION_SHAPE = (244_966, 1024)
# sample this many rows to verify unit norm after normalisation
NORMALIZATION_SAMPLE_SIZE = 10
FLOAT_NORM_ATOL = 1e-4
FLOAT_NORM_RTOL = 1e-4


class ValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class IndexSpec:
    name: str
    embedding_path: Path
    index_path: Path
    expected_shape: tuple[int, int]


INDEX_SPECS = (
    IndexSpec(
        name="visual",
        embedding_path=Path("embeddings/indexing/siglip2_embeddings.npy"),
        index_path=Path("index/indexing/faiss_visual.index"),
        expected_shape=EXPECTED_VISUAL_SHAPE,
    ),
    IndexSpec(
        name="transcript",
        embedding_path=Path("embeddings/indexing/bge_transcript_embeddings.npy"),
        index_path=Path("index/indexing/faiss_transcript.index"),
        expected_shape=EXPECTED_TRANSCRIPT_SHAPE,
    ),
    IndexSpec(
        name="caption",
        embedding_path=Path("embeddings/indexing/bge_caption_embeddings.npy"),
        index_path=Path("index/indexing/faiss_caption.index"),
        expected_shape=EXPECTED_CAPTION_SHAPE,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Populate FAISS indexes from embedding arrays.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Repository root containing embeddings/, index/, and logs/.",
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


def validate_frame_id(frame_id: object, context: str) -> str:
    require(isinstance(frame_id, str) and frame_id, f"{context}: frame_id must be a non-empty string")
    require(frame_id == frame_id.lower(), f"{context}: frame_id must be lowercase")
    parts = frame_id.split("__")
    require(len(parts) == 3, f"{context}: frame_id must have three double-underscore segments")
    stream_name, day, hh_nnnn = parts
    require(stream_name, f"{context}: missing stream segment")
    require(day.startswith("day") and day[3:].isdigit(), f"{context}: invalid day segment '{day}'")
    hh_part, sep, frame_part = hh_nnnn.partition("_")
    require(sep == "_", f"{context}: missing hour/frame separator")
    require(hh_part.isdigit() and len(hh_part) == 2, f"{context}: invalid hour segment '{hh_part}'")
    require(frame_part.isdigit() and len(frame_part) == 4, f"{context}: invalid frame segment '{frame_part}'")
    return frame_id


def parse_time_offset_seconds(timestamp_str: str, context: str) -> float:
    require(isinstance(timestamp_str, str) and timestamp_str, f"{context}: timestamp_str must be non-empty")
    parts = timestamp_str.split(":")
    require(len(parts) == 3, f"{context}: timestamp_str must have HH:MM:SS[.fff] format")
    hours_str, minutes_str, seconds_str = parts
    require(hours_str.isdigit() and minutes_str.isdigit(), f"{context}: invalid timestamp_str '{timestamp_str}'")
    try:
        seconds_value = float(seconds_str)
    except ValueError as exc:
        raise ValidationError(f"{context}: invalid seconds component in '{timestamp_str}'") from exc
    return int(hours_str) * 3600 + int(minutes_str) * 60 + seconds_value


def load_manifest_time_fields(repo_root: Path, logger: logging.Logger) -> dict[str, dict[str, object]]:
    manifest_path = repo_root / "index" / "manifest_resized.jsonl"
    require(manifest_path.is_file(), f"missing manifest source: {manifest_path}")

    manifest_lookup: dict[str, dict[str, object]] = {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            require(text, f"{manifest_path} line {line_number} is blank")
            record = json.loads(text)
            require(isinstance(record, dict), f"{manifest_path} line {line_number} is not a JSON object")
            for field_name in ("stream_name", "day", "filename", "timestamp_str", "time_offset_sec"):
                require(field_name in record, f"{manifest_path} line {line_number} missing {field_name}")

            frame_id = validate_frame_id(
                f"{record['stream_name']}__{record['day']}__{record['filename']}",
                f"{manifest_path} line {line_number}",
            )
            require(frame_id not in manifest_lookup, f"duplicate frame_id in manifest: {frame_id}")

            timestamp_str = record["timestamp_str"]
            time_offset_sec = record["time_offset_sec"]
            require(
                isinstance(timestamp_str, str) and timestamp_str,
                f"{manifest_path} line {line_number}: timestamp_str must be a non-empty string",
            )
            require(
                isinstance(time_offset_sec, (int, float)),
                f"{manifest_path} line {line_number}: time_offset_sec must be numeric",
            )

            manifest_lookup[frame_id] = {
                "timestamp_str": timestamp_str,
                "time_offset_sec": float(time_offset_sec),
            }

    logger.info("Loaded manifest time fields for %d frame_ids", len(manifest_lookup))
    return manifest_lookup


def load_embedding_array(path: Path, expected_shape: tuple[int, int], logger: logging.Logger) -> np.ndarray:
    require(path.is_file(), f"missing embedding array: {path}")
    array = np.load(path)
    require(array.size > 0, f"{path} is empty")
    require(tuple(array.shape) == expected_shape, f"{path} shape {array.shape} != expected {expected_shape}")
    require(array.dtype == np.float32, f"{path} dtype {array.dtype} != float32")
    require(np.isfinite(array).all(), f"{path} contains NaN or Inf values")
    logger.info("Loaded %s shape=%s dtype=%s", path, array.shape, array.dtype)
    return array


def verify_unit_norm_sample(array: np.ndarray, array_name: str, logger: logging.Logger) -> None:
    row_norms = np.linalg.norm(array, axis=1)
    nonzero_rows = np.flatnonzero(row_norms > 0.0).tolist()
    require(nonzero_rows, f"{array_name} has no non-zero rows available for norm verification")

    sample_count = min(NORMALIZATION_SAMPLE_SIZE, len(nonzero_rows))
    rng = random.Random(0)
    sampled_rows = sorted(rng.sample(nonzero_rows, sample_count))
    sample = array[sampled_rows]
    norms = np.linalg.norm(sample, axis=1)
    require(
        np.allclose(norms, np.ones(sample_count, dtype=np.float32), atol=FLOAT_NORM_ATOL, rtol=FLOAT_NORM_RTOL),
        f"{array_name} normalization sample norms out of tolerance: {norms.tolist()}",
    )
    logger.info("%s normalization sample rows=%s norms=%s", array_name, sampled_rows, norms.tolist())


def normalize_array(array: np.ndarray, array_name: str, logger: logging.Logger) -> None:
    # normalise in-place so inner-product search is equivalent to cosine similarity
    faiss.normalize_L2(array)
    verify_unit_norm_sample(array, array_name, logger)


def load_index(path: Path, expected_dim: int) -> faiss.Index:
    require(path.is_file(), f"missing FAISS index: {path}")
    index_obj = faiss.read_index(str(path))
    require(index_obj.is_trained is True, f"{path} is unexpectedly untrained")
    require(index_obj.d == expected_dim, f"{path} dimension {index_obj.d} != expected {expected_dim}")
    return index_obj


def populate_index(
    name: str,
    index_obj: faiss.Index,
    vectors: np.ndarray,
    expected_count: int,
    logger: logging.Logger,
) -> faiss.Index:
    require(index_obj.ntotal == 0, f"{name} index is not empty before population: ntotal={index_obj.ntotal}")
    index_obj.add(vectors)
    require(index_obj.ntotal == expected_count, f"{name} index ntotal {index_obj.ntotal} != expected {expected_count}")
    logger.info("%s index populated ntotal=%d", name, index_obj.ntotal)
    return index_obj


def write_and_reload_index(
    index_obj: faiss.Index,
    index_path: Path,
    expected_dim: int,
    expected_count: int,
    logger: logging.Logger,
) -> None:
    faiss.write_index(index_obj, str(index_path))
    # reload immediately to confirm the file round-trips correctly
    reloaded = faiss.read_index(str(index_path))
    require(reloaded.d == expected_dim, f"{index_path} reloaded dimension {reloaded.d} != {expected_dim}")
    require(
        reloaded.ntotal == expected_count,
        f"{index_path} reloaded ntotal {reloaded.ntotal} != expected {expected_count}",
    )
    logger.info("%s reloaded successfully ntotal=%d", index_path, reloaded.ntotal)


def build_visual_row_map(repo_root: Path, logger: logging.Logger) -> None:
    source_path = repo_root / "embeddings" / "siglip2_index.jsonl"
    output_path = repo_root / "index" / "indexing" / "faiss_row_map.jsonl"
    require(source_path.is_file(), f"missing visual index source: {source_path}")
    manifest_lookup = load_manifest_time_fields(repo_root, logger)

    seen_frame_ids: set[str] = set()
    total_lines = 0
    with source_path.open("r", encoding="utf-8") as in_handle, output_path.open("w", encoding="utf-8") as out_handle:
        for row_index, line in enumerate(in_handle):
            text = line.strip()
            require(text, f"{source_path} line {row_index + 1} is blank")
            record = json.loads(text)
            require(isinstance(record, dict), f"{source_path} line {row_index + 1} is not a JSON object")

            for field_name in ("frame_id", "stream_name", "day", "hour"):
                require(field_name in record, f"{source_path} line {row_index + 1} missing {field_name}")

            frame_id = validate_frame_id(record["frame_id"], f"{source_path} line {row_index + 1}")
            require(frame_id not in seen_frame_ids, f"duplicate frame_id in row map source: {frame_id}")
            seen_frame_ids.add(frame_id)

            require(
                frame_id in manifest_lookup,
                f"{source_path} line {row_index + 1}: frame_id {frame_id} not found in manifest_resized.jsonl",
            )
            manifest_record = manifest_lookup[frame_id]
            timestamp_str = str(manifest_record["timestamp_str"])
            time_offset_sec = float(manifest_record["time_offset_sec"])
            hour_value = record["hour"]
            require(isinstance(hour_value, int), f"{source_path} line {row_index + 1}: hour must be an integer")
            row_map_record = {
                "row_index": row_index,
                "frame_id": frame_id,
                "stream_name": str(record["stream_name"]),
                "day": str(record["day"]),
                "timestamp_str": timestamp_str,
                "time_offset_sec": time_offset_sec,
                "hour": hour_value,
            }
            out_handle.write(json.dumps(row_map_record, ensure_ascii=True) + "\n")
            total_lines += 1

    require(total_lines == EXPECTED_VISUAL_SHAPE[0], f"faiss_row_map.jsonl lines {total_lines} != {EXPECTED_VISUAL_SHAPE[0]}")
    require(len(seen_frame_ids) == total_lines, "faiss_row_map.jsonl contains duplicate frame_ids")
    logger.info("Built faiss_row_map.jsonl lines=%d unique_frame_ids=%d", total_lines, len(seen_frame_ids))


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    script_path = Path(__file__).resolve()
    logger, log_path = build_logger(script_path, repo_root)

    try:
        loaded_arrays: dict[str, np.ndarray] = {}

        logger.info("Stage 1: load embeddings")
        for spec in INDEX_SPECS:
            array = load_embedding_array(repo_root / spec.embedding_path, spec.expected_shape, logger)
            loaded_arrays[spec.name] = array

        logger.info("Stage 2: L2 normalization")
        for spec in INDEX_SPECS:
            normalize_array(loaded_arrays[spec.name], spec.name, logger)

        logger.info("Stage 3: load indexes and populate")
        populated_indexes: dict[str, faiss.Index] = {}
        for spec in INDEX_SPECS:
            index_obj = load_index(repo_root / spec.index_path, spec.expected_shape[1])
            populated_indexes[spec.name] = populate_index(
                name=spec.name,
                index_obj=index_obj,
                vectors=loaded_arrays[spec.name],
                expected_count=spec.expected_shape[0],
                logger=logger,
            )

        logger.info("Stage 4: build row-alignment map")
        build_visual_row_map(repo_root, logger)

        logger.info("Stage 5: save populated indexes to disk")
        for spec in INDEX_SPECS:
            write_and_reload_index(
                index_obj=populated_indexes[spec.name],
                index_path=repo_root / spec.index_path,
                expected_dim=spec.expected_shape[1],
                expected_count=spec.expected_shape[0],
                logger=logger,
            )

        logger.info("populate_faiss_indexes completed successfully")
        logger.info("log written to %s", log_path)
        return 0
    except ValidationError as exc:
        logger.error("populate_faiss_indexes failed: %s", exc)
        logger.error("log written to %s", log_path)
        return 1
    except Exception as exc:
        logger.exception("Unexpected failure: %s", exc)
        logger.error("log written to %s", log_path)
        return 1


if __name__ == "__main__":
    sys.exit(main())
