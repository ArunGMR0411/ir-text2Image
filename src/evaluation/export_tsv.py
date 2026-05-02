"""
CASTLE 2024 — Export Approach C results as TSV files (Steps 51).

Input:  evaluation/all_results.json
Output: evaluation/results_tsv/{1..10}_fusion.tsv

TSV columns (tab-separated, NO header):
  rank  stream_name  start_time  end_time  score

- rank: 1-based integer
- start_time / end_time: HH:MM:SS using hour + time_offset_sec from faiss_row_map.jsonl
- score: float, 4 decimal places
"""

import os
import sys
import json
import logging
from datetime import datetime

PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_PATH  = os.path.join(PROJECT_ROOT, "evaluation", "all_results.json")
FAISS_MAP     = os.path.join(PROJECT_ROOT, "index", "indexing", "faiss_row_map.jsonl")
OUTPUT_DIR    = os.path.join(PROJECT_ROOT, "evaluation", "results_tsv")
LOGS_DIR      = os.path.join(PROJECT_ROOT, "logs")

# each frame represents a 5-second window
WINDOW_SEC    = 5.0


def setup_logging() -> tuple[logging.Logger, str]:
    os.makedirs(LOGS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    log_path = os.path.join(LOGS_DIR, f"export_tsv_{ts}.log")
    logger = logging.getLogger("export_tsv")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s",
                                datefmt="%Y-%m-%d %H:%M:%S")
        for h in [logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)]:
            h.setFormatter(fmt)
            logger.addHandler(h)
    return logger, log_path


def format_time(hour: int, offset_sec: float) -> str:
    """Convert hour + offset_sec to HH:MM:SS, handling overflow into next hour."""
    total = int(offset_sec)
    extra_hours = total // 3600
    remaining   = total % 3600
    h = hour + extra_hours
    m = remaining // 60
    s = remaining % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_timestamps(hour: int, time_offset_sec: float,
                      window: float = WINDOW_SEC) -> tuple[str, str]:
    # clamp start to zero so we don't produce negative timestamps
    start_offset = max(0.0, time_offset_sec - window)
    end_offset   = time_offset_sec + window
    return format_time(hour, start_offset), format_time(hour, end_offset)


def load_frame_meta(path: str) -> dict[str, dict]:
    meta = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            meta[row["frame_id"]] = {
                "stream_name":    row["stream_name"],
                "hour":           int(row["hour"]),
                "time_offset_sec": float(row["time_offset_sec"]),
            }
    return meta


def main() -> int:
    logger, log_path = setup_logging()

    logger.info("=" * 70)
    logger.info(f"[export_tsv] START — {datetime.now().isoformat()}")
    logger.info(f"  Results : {RESULTS_PATH}")
    logger.info(f"  FAISS map: {FAISS_MAP}")
    logger.info(f"  Output  : {OUTPUT_DIR}")
    logger.info("=" * 70)

    logger.info("Loading all_results.json...")
    with open(RESULTS_PATH, encoding="utf-8") as f:
        all_results = json.load(f)

    logger.info("Loading faiss_row_map.jsonl...")
    frame_meta = load_frame_meta(FAISS_MAP)
    logger.info(f"  {len(frame_meta)} frame entries loaded")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    queries = sorted(all_results.keys(), key=lambda x: int(x[1:]))

    validations: list[tuple[str, bool, str]] = []
    all_pass = True

    for query_id in queries:
        query_nr = int(query_id[1:])
        filename = f"{query_nr}_fusion.tsv"
        filepath = os.path.join(OUTPUT_DIR, filename)

        results = all_results[query_id]["approach_c"]

        rows_written = 0
        with open(filepath, "w", encoding="utf-8") as f:
            for entry in results:
                rank        = entry["rank"]
                frame_id    = entry["frame_id"]
                score       = entry["score"]

                meta = frame_meta.get(frame_id)
                if meta is None:
                    logger.warning(f"  [{query_id}] frame_id not in meta: {frame_id}")
                    stream_name = entry.get("stream_name", "unknown")
                    hour        = 0
                    offset      = 0.0
                else:
                    stream_name = meta["stream_name"]
                    hour        = meta["hour"]
                    offset      = meta["time_offset_sec"]

                start_t, end_t = format_timestamps(hour, offset)

                line = f"{rank}\t{stream_name}\t{start_t}\t{end_t}\t{score:.4f}\n"
                f.write(line)
                rows_written += 1

        ok = rows_written == 10
        if not ok:
            all_pass = False
        logger.info(f"  {filename}: {rows_written} rows {'OK' if ok else 'FAIL'}")
        validations.append((f"{filename} has 10 rows", ok, f"rows={rows_written}"))

    logger.info("")
    logger.info("=" * 70)
    logger.info("VALIDATIONS")
    logger.info("=" * 70)

    expected_files = [f"{i}_fusion.tsv" for i in range(1, 11)]
    files_exist = all(os.path.exists(os.path.join(OUTPUT_DIR, fn))
                      for fn in expected_files)
    v1 = files_exist
    validations.insert(0, ("1. All 10 TSV files exist", v1,
                            f"found={sum(os.path.exists(os.path.join(OUTPUT_DIR, fn)) for fn in expected_files)}/10"))

    v2 = all(ok for _, ok, _ in validations[1:])
    validations.insert(1, ("2. Every file has exactly 10 rows", v2, ""))

    q1_top1 = all_results["Q1"]["approach_c"][0]["frame_id"]
    v3 = q1_top1 == "cathal__day4__18_0018"
    validations.append(("3. Q1 top-1 == cathal__day4__18_0018", v3,
                         f"got={q1_top1}"))

    bad_times = []
    for query_id in queries:
        results = all_results[query_id]["approach_c"]
        for entry in results:
            frame_id = entry["frame_id"]
            meta = frame_meta.get(frame_id)
            if meta is None:
                continue
            hour   = meta["hour"]
            offset = meta["time_offset_sec"]
            start_t, _ = format_timestamps(hour, offset)
            parts = start_t.split(":")
            if len(parts) != 3:
                bad_times.append(f"{frame_id}: malformed {start_t}")
                continue
            actual_h = int(parts[0])
            if hour != 0 and actual_h == 0:
                bad_times.append(f"{frame_id}: hour={hour} but start={start_t}")
    v4 = len(bad_times) == 0
    validations.append(("4. All start_time values match expected hour", v4,
                         f"bad={len(bad_times)}" + (f" e.g. {bad_times[0]}" if bad_times else "")))

    rank_ok = True
    for query_id in queries:
        ranks = [r["rank"] for r in all_results[query_id]["approach_c"]]
        if ranks != list(range(1, 11)):
            rank_ok = False
            logger.warning(f"  {query_id} ranks: {ranks}")
    v5 = rank_ok
    validations.append(("5. rank column is 1..10 with no gaps", v5, ""))

    for desc, passed, detail in validations:
        status = "PASS" if passed else "FAIL"
        suffix = f" ({detail})" if detail else ""
        logger.info(f"  {status}  {desc}{suffix}")
        if not passed:
            all_pass = False

    logger.info("=" * 70)
    if all_pass:
        logger.info("[export_tsv] COMPLETE — ALL VALIDATIONS PASSED")
    else:
        logger.error("[export_tsv] INCOMPLETE — SOME VALIDATIONS FAILED")
    logger.info(f"Log: {log_path}")
    logger.info("=" * 70)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
