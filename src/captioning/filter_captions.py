"""
Step 15: Filter Florence captions and append OCR text when usable.

Input:
  index/florence_captions.jsonl

Output:
  index/augmented_captions.jsonl

Rules:
- Stream input line by line
- If caption has fewer than 5 words, write caption=None
- Otherwise append OCR text when non-empty after stripping and length >= 2
- Write every record to output JSONL
- Assert output contains zero non-null captions with fewer than 5 words
- Log all process output to logs/filter_captions_[YYYYMMDD_HHMM].log
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator


# captions shorter than this are too noisy to be useful
MIN_CAPTION_WORDS = 5
# OCR strings shorter than this are likely noise or single characters
MIN_OCR_CHARS = 2


@dataclass
class ProcessStats:
    total_processed: int = 0
    total_discarded: int = 0
    total_kept: int = 0
    total_with_ocr_appended: int = 0


class ValidationError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter Florence captions and append OCR text.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Repository root containing index/ and logs/ directories.",
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


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            # blank lines indicate a malformed file
            if not text:
                raise ValidationError(f"{path} contains a blank line at {line_number}")
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValidationError(f"{path} has invalid JSON at line {line_number}: {exc}") from exc
            require(isinstance(record, dict), f"{path} line {line_number} is not a JSON object")
            yield line_number, record


def count_words(text: str) -> int:
    return len(text.split())


def normalize_text(value: object, field_name: str, line_number: int) -> str:
    require(isinstance(value, str), f"line {line_number}: {field_name} must be a string")
    return value


def transform_record(record: dict, line_number: int, stats: ProcessStats) -> dict:
    require("frame_id" in record, f"line {line_number}: missing frame_id")
    require("caption" in record, f"line {line_number}: missing caption")
    require("ocr_text" in record, f"line {line_number}: missing ocr_text")

    caption = normalize_text(record["caption"], "caption", line_number)
    ocr_text = normalize_text(record["ocr_text"], "ocr_text", line_number)

    word_count = count_words(caption)
    output_record = dict(record)

    stats.total_processed += 1

    # discard captions that are too short to be meaningful
    if word_count < MIN_CAPTION_WORDS:
        output_record["caption"] = None
        stats.total_discarded += 1
        return output_record

    stripped_ocr = ocr_text.strip()
    # append OCR text to the caption when it looks like real on-screen text
    if len(stripped_ocr) >= MIN_OCR_CHARS:
        output_record["caption"] = f"{caption} {stripped_ocr}"
        stats.total_with_ocr_appended += 1
    else:
        output_record["caption"] = caption

    stats.total_kept += 1
    return output_record


def write_output(input_path: Path, output_path: Path, logger: logging.Logger) -> ProcessStats:
    stats = ProcessStats()
    with output_path.open("w", encoding="utf-8") as out_handle:
        for line_number, record in iter_jsonl(input_path):
            transformed = transform_record(record, line_number, stats)
            out_handle.write(json.dumps(transformed, ensure_ascii=True) + "\n")

    logger.info("Wrote transformed records to %s", output_path)
    return stats


def validate_output(output_path: Path) -> None:
    # confirm no short captions slipped through
    for line_number, record in iter_jsonl(output_path):
        require("caption" in record, f"output line {line_number}: missing caption")
        caption = record["caption"]
        if caption is None:
            continue
        require(isinstance(caption, str), f"output line {line_number}: caption must be string or null")
        require(
            count_words(caption) >= MIN_CAPTION_WORDS,
            f"output line {line_number}: caption has fewer than {MIN_CAPTION_WORDS} words",
        )


def log_summary(logger: logging.Logger, stats: ProcessStats, log_path: Path) -> None:
    logger.info("total records processed: %d", stats.total_processed)
    logger.info("total discarded (caption=None): %d", stats.total_discarded)
    logger.info("total kept: %d", stats.total_kept)
    logger.info("total with OCR appended: %d", stats.total_with_ocr_appended)
    logger.info("log written to %s", log_path)


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    script_path = Path(__file__).resolve()
    logger, log_path = build_logger(script_path, repo_root)

    input_path = repo_root / "index" / "florence_captions.jsonl"
    output_path = repo_root / "index" / "augmented_captions.jsonl"

    logger.info("Starting caption filtering")
    logger.info("input=%s", input_path)
    logger.info("output=%s", output_path)

    try:
        require(input_path.is_file(), f"missing input file: {input_path}")
        require(output_path.parent.is_dir(), f"missing output directory: {output_path.parent}")

        stats = write_output(input_path, output_path, logger)
        validate_output(output_path)
        log_summary(logger, stats, log_path)
        return 0
    except ValidationError as exc:
        logger.error("filter_captions failed: %s", exc)
        logger.error("log written to %s", log_path)
        return 1
    except Exception as exc:
        logger.exception("Unexpected failure: %s", exc)
        logger.error("log written to %s", log_path)
        return 1


if __name__ == "__main__":
    sys.exit(main())
