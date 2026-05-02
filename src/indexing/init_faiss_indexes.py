"""
Step 21: Initialize empty FAISS IndexFlatIP indexes for each modality.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import faiss


@dataclass(frozen=True)
class IndexSpec:
    name: str
    dimension: int
    output_path: Path


# one index per modality — visual uses SigLIP2 dims, text uses BGE dims
INDEX_SPECS = (
    IndexSpec(name="visual", dimension=1152, output_path=Path("index/indexing/faiss_visual.index")),
    IndexSpec(name="transcript", dimension=1024, output_path=Path("index/indexing/faiss_transcript.index")),
    IndexSpec(name="caption", dimension=1024, output_path=Path("index/indexing/faiss_caption.index")),
)


class ValidationError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize empty FAISS IndexFlatIP indexes.")
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


def create_index(spec: IndexSpec, repo_root: Path, logger: logging.Logger) -> None:
    output_path = repo_root / spec.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # IndexFlatIP does exact inner-product search — after L2 normalisation this equals cosine similarity
    index_obj = faiss.IndexFlatIP(spec.dimension)
    require(index_obj.is_trained is True, f"{spec.name} index is unexpectedly untrained")
    require(index_obj.d == spec.dimension, f"{spec.name} index dimension {index_obj.d} != {spec.dimension}")

    faiss.write_index(index_obj, str(output_path))
    logger.info("%s index dimension: %d", spec.name, index_obj.d)
    logger.info("%s index saved to %s", spec.name, output_path)


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    script_path = Path(__file__).resolve()
    logger, log_path = build_logger(script_path, repo_root)

    try:
        for spec in INDEX_SPECS:
            create_index(spec, repo_root, logger)

        logger.info("Initialized all FAISS indexes successfully")
        logger.info("log written to %s", log_path)
        return 0
    except ValidationError as exc:
        logger.error("init_faiss_indexes failed: %s", exc)
        logger.error("log written to %s", log_path)
        return 1
    except Exception as exc:
        logger.exception("Unexpected failure: %s", exc)
        logger.error("log written to %s", log_path)
        return 1


if __name__ == "__main__":
    sys.exit(main())
