"""
CASTLE 2024 Multimodal Search — Precision@10 Metrics
Computes P@10 per query per approach against ground_truth.csv.

Inputs:
  evaluation/all_results.json   — 40 combinations, 10 results each
  evaluation/ground_truth.csv   — relevant frames (is_relevant == 1)

Output:
  evaluation/precision_at_10.csv
  logs/calculate_metrics_[YYYYMMDD_HHMM].log
"""

import csv
import json
import logging
import math
import os
import sys
from collections import defaultdict
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_PATH = os.path.join(PROJECT_ROOT, "evaluation", "all_results.json")
GT_PATH = os.path.join(PROJECT_ROOT, "evaluation", "ground_truth.csv")
OUTPUT_CSV = os.path.join(PROJECT_ROOT, "evaluation", "precision_at_10.csv")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

APPROACHES = ["approach_a", "approach_b", "approach_c", "approach_d"]


def setup_logging() -> tuple[logging.Logger, str]:
    os.makedirs(LOGS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_path = os.path.join(LOGS_DIR, f"calculate_metrics_{timestamp}.log")

    logger = logging.getLogger("calculate_metrics")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger, log_path


def load_ground_truth(path: str) -> set[tuple[str, str]]:
    # store as (query_id, frame_id) pairs for O(1) hit checking
    gt = set()
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if str(row.get("is_relevant", "")).strip() == "1":
                gt.add((row["query_id"].strip(), row["frame_id"].strip()))
    return gt


def load_all_results(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def precision_at_10(query_id: str, results: list[dict], ground_truth: set[tuple[str, str]]) -> float:
    hits = sum(1 for r in results if (query_id, r["frame_id"]) in ground_truth)
    return round(hits / 10, 1)


def best_approach(scores: dict[str, float]) -> str:
    best_val = max(scores.values())
    winners = [label for label, value in scores.items() if value == best_val]
    return "/".join(winners)


def generate_note(scores: dict[str, float]) -> str:
    if all(value == 0.0 for value in scores.values()):
        return "all fail — no ground truth"
    winners = best_approach(scores).split("/")
    if winners == ["D"]:
        return "D dominates"
    if "D" in winners and len(winners) > 1:
        return "D ties best"
    if winners == ["C"]:
        return "C dominates"
    if winners == ["B"]:
        return "B dominates"
    if winners == ["A"]:
        return "A dominates"
    return "mixed"


def main() -> int:
    logger, log_path = setup_logging()

    logger.info("=" * 72)
    logger.info(f"[calculate_metrics] START — {datetime.now().isoformat()}")
    logger.info(f"  Results : {RESULTS_PATH}")
    logger.info(f"  GT      : {GT_PATH}")
    logger.info(f"  Output  : {OUTPUT_CSV}")
    logger.info("=" * 72)

    logger.info("Loading ground truth...")
    ground_truth = load_ground_truth(GT_PATH)
    logger.info(f"  {len(ground_truth)} relevant (query_id, frame_id) pairs loaded")

    # count relevant frames per query for diagnostic logging
    gt_by_query: dict[str, int] = defaultdict(int)
    for qid, _ in ground_truth:
        gt_by_query[qid] += 1
    for q in sorted(gt_by_query.keys(), key=lambda x: int(x[1:])):
        logger.info(f"  {q}: {gt_by_query[q]} relevant frames in ground truth")

    logger.info("Loading all_results.json...")
    all_results = load_all_results(RESULTS_PATH)
    queries = sorted(all_results.keys(), key=lambda x: int(x[1:]))
    logger.info(f"  {len(queries)} queries × {len(APPROACHES)} approaches loaded")

    logger.info("Computing P@10...")
    table: list[dict] = []
    for q in queries:
        row = {"query_id": q}
        score_map = {}
        for approach in APPROACHES:
            score = precision_at_10(q, all_results[q][approach], ground_truth)
            row[approach] = score
            # use single-letter keys for the note generator
            score_map[approach.split("_")[-1].upper()] = score
        row["best_approach"] = best_approach(score_map)
        row["notes"] = generate_note(score_map)
        table.append(row)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    fieldnames = ["query_id", "approach_a", "approach_b", "approach_c", "approach_d", "best_approach", "notes"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table)
    logger.info(f"Wrote {len(table)} rows to {OUTPUT_CSV}")

    mean_a = round(sum(r["approach_a"] for r in table) / len(table), 3)
    mean_b = round(sum(r["approach_b"] for r in table) / len(table), 3)
    mean_c = round(sum(r["approach_c"] for r in table) / len(table), 3)
    mean_d = round(sum(r["approach_d"] for r in table) / len(table), 3)

    logger.info("")
    logger.info("=" * 72)
    logger.info("P@10 TABLE (examiner-ready)")
    logger.info("=" * 72)
    logger.info(f"{'Query':<6}  {'A':>6}  {'B':>6}  {'C':>6}  {'D':>6}  {'Best':<8}  Notes")
    logger.info("-" * 72)
    for r in table:
        logger.info(
            f"{r['query_id']:<6}  {r['approach_a']:>6.1f}  {r['approach_b']:>6.1f}  "
            f"{r['approach_c']:>6.1f}  {r['approach_d']:>6.1f}  {r['best_approach']:<8}  {r['notes']}"
        )
    logger.info("-" * 72)
    logger.info(f"{'Mean':<6}  {mean_a:>6.3f}  {mean_b:>6.3f}  {mean_c:>6.3f}  {mean_d:>6.3f}")
    logger.info("=" * 72)

    logger.info("")
    logger.info("SUMMARY STATISTICS")
    logger.info(f"  Mean P@10 — A: {mean_a:.3f}  B: {mean_b:.3f}  C: {mean_c:.3f}  D: {mean_d:.3f}")

    logger.info("")
    logger.info("=" * 72)
    logger.info("VALIDATIONS")
    logger.info("=" * 72)
    validations: list[tuple[str, bool, str]] = []

    csv_ok = os.path.exists(OUTPUT_CSV)
    with open(OUTPUT_CSV, newline="", encoding="utf-8") as f:
        rows_written = sum(1 for _ in csv.DictReader(f))
    validations.append(("1. precision_at_10.csv exists, 10 rows", csv_ok and rows_written == 10, f"rows={rows_written}"))

    all_vals = [r[ap] for r in table for ap in APPROACHES]
    validations.append((
        "2. All P@10 values are multiples of 0.1",
        all(math.isclose(v * 10, round(v * 10), rel_tol=1e-9) for v in all_vals),
        "",
    ))

    q7 = next(r for r in table if r["query_id"] == "Q7")
    q7_gt_count = gt_by_query.get("Q7", 0)
    if q7_gt_count == 0:
        validations.append((
            "3. Q7 all approaches == 0.0 when no ground truth exists",
            all(q7[ap] == 0.0 for ap in APPROACHES),
            f"A={q7['approach_a']} B={q7['approach_b']} C={q7['approach_c']} D={q7['approach_d']}",
        ))
    else:
        validations.append((
            "3. Q7 Approach B > 0.0 when Q7 ground truth exists",
            q7["approach_b"] > 0.0,
            f"q7_gt={q7_gt_count} A={q7['approach_a']} B={q7['approach_b']} C={q7['approach_c']} D={q7['approach_d']}",
        ))

    q8 = next(r for r in table if r["query_id"] == "Q8")
    validations.append((
        "4. Q8 B==1.0 and C==1.0",
        q8["approach_b"] == 1.0 and q8["approach_c"] == 1.0,
        f"B={q8['approach_b']} C={q8['approach_c']}",
    ))

    validations.append(("5. Mean P@10 Approach B > 0.5", mean_b > 0.5, f"mean_B={mean_b:.3f}"))
    validations.append(("6. Mean P@10 Approach D >= Approach C", mean_d >= mean_c, f"mean_D={mean_d:.3f} mean_C={mean_c:.3f}"))

    q1 = next(r for r in table if r["query_id"] == "Q1")
    validations.append(("7. Q1 D >= Q1 C", q1["approach_d"] >= q1["approach_c"], f"D={q1['approach_d']} C={q1['approach_c']}"))
    validations.append(("8. Full P@10 table printed to log", True, "see table above"))

    all_pass = True
    for desc, passed, detail in validations:
        logger.info(f"  {'PASS' if passed else 'FAIL'}  {desc}{f' ({detail})' if detail else ''}")
        if not passed:
            all_pass = False

    logger.info("=" * 72)
    if all_pass:
        logger.info("[calculate_metrics] COMPLETE — ALL VALIDATIONS PASSED")
    else:
        logger.error("[calculate_metrics] INCOMPLETE — SOME VALIDATIONS FAILED")
    logger.info(f"Log: {log_path}")
    logger.info("=" * 72)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
