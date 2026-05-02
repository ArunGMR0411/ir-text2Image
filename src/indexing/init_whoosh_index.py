"""
Step 19: Initialize an empty Whoosh index with the required schema.

This script creates the index directory and index object only. It does
not add any documents.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from whoosh import index
from whoosh.analysis import StemmingAnalyzer
from whoosh.fields import ID, STORED, TEXT, Schema


EXPECTED_FIELD_NAMES = [
    "frame_id",
    "stream_name",
    "day",
    "timestamp_str",
    "hour",
    "transcript_text",
    "caption_text",
]


class ValidationError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize the empty Whoosh index schema.")
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


def build_schema() -> Schema:
    # StemmingAnalyzer reduces words to their root form for better BM25 recall
    return Schema(
        frame_id=ID(stored=True, unique=True),
        stream_name=STORED,
        day=STORED,
        timestamp_str=STORED,
        hour=STORED,
        transcript_text=TEXT(analyzer=StemmingAnalyzer(), stored=False),
        caption_text=TEXT(analyzer=StemmingAnalyzer(), stored=True),
    )


def validate_schema(schema: Schema) -> list[str]:
    field_names = list(schema.names())
    require(
        set(field_names) == set(EXPECTED_FIELD_NAMES),
        f"schema fields {field_names} do not match expected members {EXPECTED_FIELD_NAMES}",
    )
    require(len(field_names) == 7, f"schema field count is {len(field_names)}, expected 7")
    return field_names


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    script_path = Path(__file__).resolve()
    logger, log_path = build_logger(script_path, repo_root)

    index_dir = repo_root / "index" / "indexing" / "whoosh_index"

    try:
        logger.info("Initializing Whoosh index at %s", index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        schema = build_schema()
        # open existing index if present, otherwise create a fresh one
        if index.exists_in(index_dir):
            created_index = index.open_dir(index_dir)
        else:
            created_index = index.create_in(index_dir, schema)

        require(index_dir.is_dir(), f"missing index directory after creation: {index_dir}")

        schema_field_names = validate_schema(created_index.schema)
        logger.info("schema field names: %s", schema_field_names)
        logger.info("index directory exists: %s", index_dir)
        logger.info("log written to %s", log_path)
        return 0
    except ValidationError as exc:
        logger.error("init_whoosh_index failed: %s", exc)
        logger.error("log written to %s", log_path)
        return 1
    except Exception as exc:
        logger.exception("Unexpected failure: %s", exc)
        logger.error("log written to %s", log_path)
        return 1


if __name__ == "__main__":
    sys.exit(main())
