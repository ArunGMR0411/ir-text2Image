"""
Step 20: Ingest manifest-aligned documents into the Whoosh index.

This script:
- Aggregates duplicate transcript rows by frame_id
- Loads augmented captions by frame_id
- Streams manifest_resized.jsonl as the authority frame universe
- Adds all 416,542 documents to the existing Whoosh index
- Commits every 10,000 documents
- Uses a checkpoint file for restart-safe ingestion
- Verifies final document count and runs three spot-check queries
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
from whoosh import index
from whoosh.qparser import QueryParser


EXPECTED_DOC_COUNT = 416_542
# commit in batches so a crash doesn't lose all progress
COMMIT_BATCH_SIZE = 10_000
CHECKPOINT_FILENAME = ".ingest_whoosh_checkpoint.json"
TEST_TERMS = ["kitchen", "eating", "table"]
REQUIRED_MANIFEST_FIELDS = ["stream_name", "day", "filename", "timestamp_str", "hour"]


class ValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Paths:
    repo_root: Path
    transcript_csv: Path
    augmented_captions: Path
    manifest_jsonl: Path
    whoosh_dir: Path
    checkpoint_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest transcript and caption text into the Whoosh index.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Repository root containing index/ and logs/.",
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


def build_paths(repo_root: Path) -> Paths:
    index_dir = repo_root / "index"
    whoosh_dir = index_dir / "whoosh_index"
    return Paths(
        repo_root=repo_root,
        transcript_csv=index_dir / "transcript_cleaned.csv",
        augmented_captions=index_dir / "augmented_captions_clean.jsonl",
        manifest_jsonl=index_dir / "manifest_resized.jsonl",
        whoosh_dir=whoosh_dir,
        checkpoint_path=whoosh_dir / CHECKPOINT_FILENAME,
    )


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


def load_transcripts(path: Path, logger: logging.Logger) -> dict[str, str]:
    require(path.is_file(), f"missing transcript source: {path}")
    df = pd.read_csv(path, usecols=["frame_id", "transcript_text"])
    require("frame_id" in df.columns and "transcript_text" in df.columns, f"{path} missing required columns")

    df["frame_id"] = df["frame_id"].map(lambda value: validate_frame_id(value, f"{path}"))
    df["transcript_text"] = df["transcript_text"].fillna("").astype(str).str.strip()

    # multiple transcript chunks can map to the same frame — join them into one string
    grouped = (
        df.groupby("frame_id", sort=False)["transcript_text"]
        .agg(lambda values: " ".join(value for value in values if value))
        .to_dict()
    )
    logger.info(
        "Loaded transcripts: input_rows=%d unique_frame_ids=%d",
        len(df),
        len(grouped),
    )
    return grouped


def load_captions(path: Path, logger: logging.Logger) -> dict[str, str]:
    require(path.is_file(), f"missing caption source: {path}")
    captions: dict[str, str] = {}
    record_count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            require(text, f"{path} line {line_number} is blank")
            record = json.loads(text)
            require(isinstance(record, dict), f"{path} line {line_number} is not a JSON object")
            require("frame_id" in record, f"{path} line {line_number} missing frame_id")
            require("caption" in record, f"{path} line {line_number} missing caption")
            frame_id = validate_frame_id(record["frame_id"], f"{path} line {line_number}")
            caption_value = record["caption"]
            if caption_value is None:
                captions[frame_id] = ""
            else:
                require(isinstance(caption_value, str), f"{path} line {line_number}: caption must be string or null")
                captions[frame_id] = caption_value.strip()
            record_count += 1
    logger.info("Loaded captions: input_rows=%d unique_frame_ids=%d", record_count, len(captions))
    return captions


def load_checkpoint(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    require(isinstance(payload, dict), f"{path} checkpoint payload must be an object")
    committed_docs = payload.get("committed_docs")
    require(isinstance(committed_docs, int) and committed_docs >= 0, f"{path} invalid committed_docs")
    return committed_docs


def save_checkpoint(path: Path, committed_docs: int) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump({"committed_docs": committed_docs}, handle, ensure_ascii=True)


def iter_manifest_records(path: Path):
    require(path.is_file(), f"missing manifest source: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            require(text, f"{path} line {line_number} is blank")
            record = json.loads(text)
            require(isinstance(record, dict), f"{path} line {line_number} is not a JSON object")
            missing = [field for field in REQUIRED_MANIFEST_FIELDS if field not in record]
            require(not missing, f"{path} line {line_number} missing fields: {missing}")
            frame_id = validate_frame_id(
                f"{record['stream_name']}__{record['day']}__{record['filename']}",
                f"{path} line {line_number}",
            )
            yield line_number, {
                "frame_id": frame_id,
                "stream_name": str(record["stream_name"]),
                "day": str(record["day"]),
                "timestamp_str": str(record["timestamp_str"]),
                "hour": str(record["hour"]),
            }


def commit_batch(writer, checkpoint_path: Path, committed_docs: int, logger: logging.Logger) -> None:
    writer.commit()
    save_checkpoint(checkpoint_path, committed_docs)
    logger.info("Committed %d documents", committed_docs)


def ingest_documents(paths: Paths, transcripts: dict[str, str], captions: dict[str, str], logger: logging.Logger) -> None:
    require(paths.whoosh_dir.is_dir(), f"missing Whoosh index directory: {paths.whoosh_dir}")
    require(index.exists_in(paths.whoosh_dir), f"Whoosh index does not exist in {paths.whoosh_dir}")

    idx = index.open_dir(paths.whoosh_dir)
    committed_docs = load_checkpoint(paths.checkpoint_path)
    current_doc_count = idx.doc_count()

    require(
        current_doc_count == committed_docs or (current_doc_count == EXPECTED_DOC_COUNT and committed_docs == 0),
        "checkpoint/index state mismatch: current_doc_count="
        f"{current_doc_count}, checkpoint_committed_docs={committed_docs}",
    )

    if current_doc_count == EXPECTED_DOC_COUNT and committed_docs == 0:
        logger.info("Index already contains expected document count; skipping ingestion")
        return

    if committed_docs == EXPECTED_DOC_COUNT:
        logger.info("Checkpoint indicates completed ingestion; skipping document writes")
        return

    writer = idx.writer()
    docs_in_open_batch = 0
    total_seen = 0

    try:
        for _, manifest_record in iter_manifest_records(paths.manifest_jsonl):
            # skip rows already committed in a previous run
            if total_seen < committed_docs:
                total_seen += 1
                continue

            frame_id = manifest_record["frame_id"]
            transcript_text = transcripts.get(frame_id, "")
            caption_text = captions.get(frame_id, "")

            require(isinstance(transcript_text, str), f"{frame_id}: transcript_text must be a string")
            require(isinstance(caption_text, str), f"{frame_id}: caption_text must be a string")

            writer.add_document(
                frame_id=frame_id,
                stream_name=manifest_record["stream_name"],
                day=manifest_record["day"],
                timestamp_str=manifest_record["timestamp_str"],
                hour=manifest_record["hour"],
                transcript_text=transcript_text,
                caption_text=caption_text,
            )

            total_seen += 1
            docs_in_open_batch += 1

            if docs_in_open_batch == COMMIT_BATCH_SIZE:
                commit_batch(writer, paths.checkpoint_path, total_seen, logger)
                writer = idx.writer()
                docs_in_open_batch = 0

        if docs_in_open_batch > 0:
            commit_batch(writer, paths.checkpoint_path, total_seen, logger)
        else:
            writer.cancel()
    except Exception:
        try:
            writer.cancel()
        except Exception:
            pass
        raise

    require(total_seen == EXPECTED_DOC_COUNT, f"processed {total_seen} documents, expected {EXPECTED_DOC_COUNT}")


def verify_index(paths: Paths, logger: logging.Logger) -> None:
    idx = index.open_dir(paths.whoosh_dir)
    with idx.searcher() as searcher:
        doc_count = searcher.doc_count_all()
        require(
            doc_count == EXPECTED_DOC_COUNT,
            f"fatal index count mismatch: got {doc_count}, expected {EXPECTED_DOC_COUNT}",
        )
        logger.info("Verified index document count: %d", doc_count)

        # spot-check a few common terms to confirm the index is searchable
        parser = QueryParser("transcript_text", schema=idx.schema)
        for term in TEST_TERMS:
            query = parser.parse(f"transcript_text:{term} OR caption_text:{term}")
            results = searcher.search(query, limit=3)
            top_frame_ids = [hit["frame_id"] for hit in results]
            logger.info("spot_check term=%s top3_frame_ids=%s", term, top_frame_ids)


def cleanup_checkpoint(path: Path, logger: logging.Logger) -> None:
    if path.exists():
        path.unlink()
        logger.info("Removed checkpoint file %s", path)


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    script_path = Path(__file__).resolve()
    logger, log_path = build_logger(script_path, repo_root)
    paths = build_paths(repo_root)

    try:
        logger.info("Starting Whoosh ingestion")
        logger.info("transcript_csv=%s", paths.transcript_csv)
        logger.info("augmented_captions=%s", paths.augmented_captions)
        logger.info("manifest_jsonl=%s", paths.manifest_jsonl)
        logger.info("whoosh_dir=%s", paths.whoosh_dir)

        transcripts = load_transcripts(paths.transcript_csv, logger)
        captions = load_captions(paths.augmented_captions, logger)
        ingest_documents(paths, transcripts, captions, logger)
        verify_index(paths, logger)
        cleanup_checkpoint(paths.checkpoint_path, logger)
        logger.info("Whoosh ingestion completed successfully")
        logger.info("log written to %s", log_path)
        return 0
    except ValidationError as exc:
        logger.error("ingest_whoosh failed: %s", exc)
        logger.error("log written to %s", log_path)
        return 1
    except Exception as exc:
        logger.exception("Unexpected failure: %s", exc)
        logger.error("log written to %s", log_path)
        return 1


if __name__ == "__main__":
    sys.exit(main())
