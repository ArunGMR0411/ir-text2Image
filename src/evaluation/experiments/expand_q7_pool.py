import csv
import json
import logging
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

EVAL_DIR = PROJECT_ROOT / "evaluation"
INDEX_DIR = PROJECT_ROOT / "index"
LOG_DIR = PROJECT_ROOT / "logs"

ALL_RESULTS_PATH = EVAL_DIR / "all_results.json"
ROW_MAP_PATH = INDEX_DIR / "faiss_row_map.jsonl"
EMBEDDINGS_PATH = PROJECT_ROOT / "embeddings" / "siglip2_embeddings.npy"
FAISS_INDEX_PATH = INDEX_DIR / "faiss_visual.index"
SEARCH_SCRIPT_PATH = PROJECT_ROOT / "src" / "search_approach_b.py"
PRECISION_PATH = EVAL_DIR / "precision_at_10.csv"
OUTPUT_PATH = EVAL_DIR / "q7_expanded_pool.csv"

Q7_ID = "Q7"
Q7_QUERY = "Squirrel christmas tree ornament"


def make_logger() -> tuple[logging.Logger, Path, str]:
    timestamp = datetime.now().strftime("%Y_%b_%d_%H_%M").upper()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"STEP_57_expand_q7_pool_{timestamp}.log"

    logger = logging.getLogger("step57_expand_q7_pool")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger, log_path, timestamp


def log_line(logger: logging.Logger, line: str) -> None:
    logger.info(line)


def load_q7_precision_row() -> dict[str, str] | None:
    with PRECISION_PATH.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["query_id"] == Q7_ID:
                return row
    return None


def run_preflight_checks(logger: logging.Logger) -> list[tuple[str, bool, str]]:
    import faiss
    import numpy as np

    checks: list[tuple[str, bool, str]] = []

    # verify all required artifacts exist before doing any expensive work
    exists = ALL_RESULTS_PATH.exists()
    size = ALL_RESULTS_PATH.stat().st_size if exists else 0
    checks.append(("1. evaluation/all_results.json exists", exists, f"size={size} bytes" if exists else "missing"))

    row_map_exists = ROW_MAP_PATH.exists()
    row_count = 0
    if row_map_exists:
        with ROW_MAP_PATH.open("r", encoding="utf-8") as f:
            row_count = sum(1 for _ in f)
    checks.append((
        "2. index/indexing/faiss_row_map.jsonl exists with 416542 lines",
        row_map_exists and row_count == 416542,
        f"lines={row_count}" if row_map_exists else "missing",
    ))

    emb_exists = EMBEDDINGS_PATH.exists()
    emb_shape = None
    emb_dtype = None
    if emb_exists:
        embeddings = np.load(EMBEDDINGS_PATH, mmap_mode="r")
        emb_shape = embeddings.shape
        emb_dtype = embeddings.dtype
    checks.append((
        "3. embeddings/indexing/siglip2_embeddings.npy exists with shape (416542, 1152)",
        emb_exists and emb_shape == (416542, 1152),
        f"shape={emb_shape} dtype={emb_dtype}" if emb_exists else "missing",
    ))

    index_exists = FAISS_INDEX_PATH.exists()
    ntotal = None
    if index_exists:
        index = faiss.read_index(str(FAISS_INDEX_PATH))
        ntotal = index.ntotal
    checks.append((
        "4. index/indexing/faiss_visual.index exists with ntotal 416542",
        index_exists and ntotal == 416542,
        f"ntotal={ntotal}" if index_exists else "missing",
    ))

    search_exists = SEARCH_SCRIPT_PATH.exists()
    search_size = SEARCH_SCRIPT_PATH.stat().st_size if search_exists else 0
    checks.append((
        "5. src/search_approach_b.py exists",
        search_exists,
        f"size={search_size} bytes" if search_exists else "missing",
    ))

    q7_row = load_q7_precision_row()
    q7_ok = q7_row is not None and all(q7_row[k] == "0.0" for k in ["approach_a", "approach_b", "approach_c", "approach_d"])
    checks.append((
        "6. evaluation/precision_at_10.csv Q7 row is all 0.0",
        q7_ok,
        (
            "approach_a={approach_a}, approach_b={approach_b}, approach_c={approach_c}, approach_d={approach_d}".format(**q7_row)
            if q7_row else "missing Q7 row"
        ),
    ))

    return checks


def load_row_map() -> dict[str, dict[str, object]]:
    row_map: dict[str, dict[str, object]] = {}
    with ROW_MAP_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            row_map[rec["frame_id"]] = rec
    return row_map


def write_output(results: list[tuple[str, float]], row_map: dict[str, dict[str, object]]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "frameid", "streamname", "day", "timestampstr", "score"])
        for rank, (frame_id, score) in enumerate(results, start=1):
            rec = row_map[frame_id]
            writer.writerow([
                rank,
                frame_id,
                rec["stream_name"],
                rec["day"],
                rec["timestamp_str"],
                f"{score:.6f}",
            ])


def validate_output() -> dict[str, object]:
    with OUTPUT_PATH.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    ranks = [int(row["rank"]) for row in rows]
    frame_ids = [row["frameid"] for row in rows]
    scores = [float(row["score"]) for row in rows]
    living_count = sum(1 for row in rows if "living" in row["streamname"].lower())

    validations = {
        "exists": OUTPUT_PATH.exists(),
        "row_count": len(rows),
        "rank_sequence": ranks == list(range(1, 51)),
        "unique_frameids": len(frame_ids) == len(set(frame_ids)),
        "score_range": all(0.0 <= score <= 1.0 for score in scores),
        "living_count": living_count,
        "rows": rows,
    }
    return validations


def main() -> int:
    logger, log_path, timestamp = make_logger()
    start = datetime.now()

    log_line(logger, "=" * 80)
    log_line(logger, "[STAGE 57] Expand Q7 Pool to Top-50")
    log_line(logger, f"Started : {timestamp}")
    log_line(logger, "Script  : src/expand_q7_pool.py")
    log_line(logger, "Input   : evaluation/all_results.json, index/indexing/faiss_row_map.jsonl, embeddings/indexing/siglip2_embeddings.npy, index/indexing/faiss_visual.index")
    log_line(logger, "=" * 80)

    log_line(logger, f"[{datetime.now().strftime('%H:%M:%S')}] INIT     Running pre-flight checks")
    checks = run_preflight_checks(logger)
    all_checks_pass = True
    for desc, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        log_line(logger, f"  {status} | {desc} ({detail})")
        if not passed:
            all_checks_pass = False

    if not all_checks_pass:
        log_line(logger, f"[{datetime.now().strftime('%H:%M:%S')}] FAIL     Pre-flight gate failed")
        log_line(logger, "-" * 80)
        log_line(logger, "VALIDATION")
        log_line(logger, "  FAIL | Step 57 pre-flight gate did not pass")
        log_line(logger, "-" * 80)
        log_line(logger, "ARTIFACTS")
        log_line(logger, f"  {log_path}   {log_path.stat().st_size if log_path.exists() else 0} B")
        log_line(logger, "=" * 80)
        log_line(logger, "[STAGE 57] COMPLETE — FAILED")
        log_line(logger, "=" * 80)
        return 1

    from src.retrieval.search_approach_b import search_approach_b

    log_line(logger, f"[{datetime.now().strftime('%H:%M:%S')}] START    Running Approach B for Q7 with topk=50")
    raw_results = search_approach_b(Q7_ID, Q7_QUERY, top_k=50)
    # sort descending by score so rank 1 is the best match
    sorted_results = sorted(raw_results.items(), key=lambda item: item[1], reverse=True)
    row_map = load_row_map()
    write_output(sorted_results, row_map)

    log_line(logger, f"[{datetime.now().strftime('%H:%M:%S')}] PROG     Top-10 Q7 candidates for manual review")
    for rank, (frame_id, score) in enumerate(sorted_results[:10], start=1):
        stream_name = str(row_map[frame_id]["stream_name"])
        log_line(logger, f"  {rank:02d} | {frame_id} | {stream_name} | {score:.6f}")

    validation = validate_output()
    duration = (datetime.now() - start).total_seconds()
    stream_counter = Counter(row["streamname"] for row in validation["rows"])

    log_line(logger, f"[{datetime.now().strftime('%H:%M:%S')}] DONE     50 candidates written | {duration:.2f}s total")
    log_line(logger, "-" * 80)
    log_line(logger, "VALIDATION")
    vchecks = [
        ("1. evaluation/q7_expanded_pool.csv exists", validation["exists"]),
        ("2. Exactly 50 rows excluding header", validation["row_count"] == 50),
        ("3. Rank values are 1..50 with no gaps", validation["rank_sequence"]),
        ("4. No frameid appears more than once", validation["unique_frameids"]),
        ("5. All scores are in [0.0, 1.0]", validation["score_range"]),
        ("6. At least 5 rows have streamname containing 'living'", validation["living_count"] >= 5),
        ("7. Top-10 frameids + streamnames + scores logged", True),
    ]
    all_pass = True
    for desc, passed in vchecks:
        log_line(logger, f"  {'PASS' if passed else 'FAIL'} | {desc}")
        if not passed:
            all_pass = False

    log_line(logger, "-" * 80)
    log_line(logger, "ARTIFACTS")
    log_line(logger, f"  src/expand_q7_pool.py                {Path(__file__).stat().st_size} B")
    log_line(logger, f"  evaluation/q7_expanded_pool.csv      {OUTPUT_PATH.stat().st_size} B")
    log_line(logger, f"  {log_path.relative_to(PROJECT_ROOT)}   {log_path.stat().st_size} B")
    if stream_counter:
        log_line(logger, f"  Top streamnames by frequency         {stream_counter.most_common(3)}")
    log_line(logger, "=" * 80)
    log_line(logger, f"[STAGE 57] COMPLETE — {'PASS' if all_pass else 'FAIL'}")
    log_line(logger, "=" * 80)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
