"""
CASTLE 2024 Multimodal Search — Programmatic Evaluation
Steps 46 + 49 + 61: Run all 10 queries × 4 approaches, save evaluation/all_results.json
"""
import os
import sys
import json
import logging
from datetime import datetime
from typing import List, Dict, Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from src.retrieval.query_expansion import RAW_QUERIES
from src.ranking.postprocess import load_frame_meta, temporal_dedup, flag_cross_stream

logger = logging.getLogger(__name__)


def assert_top10(results: list, query_id: str, approach: str) -> list:
    """
    Step 49: Applies [:10] truncation and asserts len <= 10.
    Raises ValueError with query_id + approach if violated.
    Returns truncated list.
    """
    truncated = results[:10]
    
    if len(truncated) > 10:
        raise ValueError(
            f"Top-10 assertion FAILED for {query_id}/{approach}: "
            f"got {len(truncated)} items (expected <= 10)"
        )
    
    return truncated


def setup_logging():
    """Setup logging to file and console."""
    timestamp = datetime.now().strftime("%Y_%b_%d_%H_%M").upper()
    log_file = os.path.join(PROJECT_ROOT, "logs", f"run_evaluation_{timestamp}.log")
    
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return log_file


def clear_gpu_cache():
    """Clear GPU cache to prevent OOM."""
    try:
        import torch
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def run_single_evaluation(
    query_id: str,
    raw_query: str,
    approach: str,
    frame_meta: dict
) -> List[Dict[str, Any]]:
    """
    Run a single query/approach combination.
    Returns list of dicts with rank, frame_id, score, stream_name, multi_angle.
    
    Note: Import search modules inside function to allow memory cleanup between runs.
    """
    logger.info(f"  [{query_id}/{approach}] Running search...")
    
    if approach == "approach_a":
        from src.retrieval.search_approach_a import search_approach_a
        results_dict = search_approach_a(query_id, raw_query, top_k=50)
        results_list = sorted(results_dict.items(), key=lambda x: (-x[1], x[0]))
    elif approach == "approach_b":
        from src.retrieval.search_approach_b import search_approach_b
        results_dict = search_approach_b(query_id, raw_query, top_k=50)
        results_list = sorted(results_dict.items(), key=lambda x: (-x[1], x[0]))
        clear_gpu_cache()
    elif approach == "approach_c":
        from src.retrieval.search_approach_c import search_approach_c
        results_list = search_approach_c(query_id, raw_query, top_k=50)
        clear_gpu_cache()
    else:
        from src.retrieval.search_approach_d import search_approach_d
        results_list = search_approach_d(query_id, raw_query)
        clear_gpu_cache()
    
    # approach_d handles its own dedup internally
    if approach == "approach_d":
        deduped = results_list
    else:
        deduped = temporal_dedup(results_list, frame_meta, window_sec=10.0)
    
    flagged = flag_cross_stream(deduped, frame_meta)
    
    top10 = assert_top10(flagged, query_id, approach)
    
    formatted = []
    for rank, (frame_id, score, multi_angle) in enumerate(top10, start=1):
        meta = frame_meta.get(frame_id, {})
        stream_name = meta.get("stream_name", "unknown")
        
        formatted.append({
            "rank": rank,
            "frame_id": frame_id,
            "score": round(float(score), 4),
            "stream_name": stream_name,
            "multi_angle": bool(multi_angle)
        })
    
    logger.info(f"  [{query_id}/{approach}] DONE — {len(formatted)} results")
    return formatted


def verify_all_results(filepath: str) -> bool:
    """
    Step 49: Post-run check on all_results.json.
    Assert every result list has exactly 10 items.
    """
    logger.info("=" * 60)
    logger.info("TOP-10 INTEGRITY CHECK")
    logger.info("=" * 60)
    
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    violations = []
    
    for query_id in data:
        for approach in data[query_id]:
            results = data[query_id][approach]
            if len(results) != 10:
                violations.append(f"{query_id}/{approach}: {len(results)} items")
    
    if violations:
        logger.error(f"  FAIL — Violations found:")
        for v in violations:
            logger.error(f"    {v}")
        return False
    else:
        logger.info("  PASS — All 40 result lists have exactly 10 items")
        return True


def main():
    """Main evaluation runner."""
    log_file = setup_logging()
    
    logger.info("=" * 80)
    logger.info(f"[STEPS 46+49] Programmatic Evaluation — {datetime.now().strftime('%Y_%b_%d_%H_%M').upper()}")
    logger.info("Script  : src/run_evaluation.py")
    logger.info("=" * 80)
    
    logger.info("INIT     Loading frame metadata...")
    meta_path = os.path.join(PROJECT_ROOT, "index/indexing/faiss_row_map.jsonl")
    frame_meta = load_frame_meta(meta_path)
    logger.info(f"  Loaded {len(frame_meta)} frame entries")
    
    logger.info("INIT     Search modules will be loaded per-approach for memory management")
    
    all_results: Dict[str, Dict[str, List[Dict]]] = {}
    
    total = 40
    succeeded = 0
    failed = 0
    failures = []
    
    queries = sorted(RAW_QUERIES.keys())
    approaches = ["approach_a", "approach_b", "approach_c", "approach_d"]
    
    logger.info("START    Running 40 query/approach combinations...")
    logger.info("-" * 60)
    
    for query_id in queries:
        all_results[query_id] = {}
        raw_query = RAW_QUERIES[query_id]
        
        for approach in approaches:
            try:
                results = run_single_evaluation(
                    query_id=query_id,
                    raw_query=raw_query,
                    approach=approach,
                    frame_meta=frame_meta
                )
                
                all_results[query_id][approach] = results
                succeeded += 1
                
            except Exception as e:
                logger.error(f"  [{query_id}/{approach}] FAILED: {e}")
                all_results[query_id][approach] = []
                failed += 1
                failures.append(f"{query_id}/{approach}: {str(e)}")
    
    logger.info("-" * 60)
    
    eval_dir = os.path.join(PROJECT_ROOT, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    
    output_path = os.path.join(eval_dir, "all_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"DONE     Saved: {output_path}")
    
    integrity_passed = verify_all_results(output_path)
    
    logger.info("=" * 80)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"  Total combinations:    {total}")
    logger.info(f"  Succeeded:             {succeeded}")
    logger.info(f"  Failed:                {failed}")
    logger.info(f"  Top-10 integrity:      {'PASS' if integrity_passed else 'FAIL'}")
    
    if failures:
        logger.info("  Failures:")
        for f in failures:
            logger.info(f"    - {f}")
    
    logger.info("-" * 80)
    logger.info("ARTIFACTS")
    logger.info(f"  evaluation/all_results.json   {os.path.getsize(output_path)} B")
    logger.info(f"  {log_file}   {os.path.getsize(log_file)} B")
    logger.info("=" * 80)
    
    if failed == 0 and integrity_passed:
        logger.info("[STEPS 46+49] COMPLETE — ALL VALIDATIONS PASSED")
        return 0
    else:
        logger.error("[STEPS 46+49] INCOMPLETE — SOME VALIDATIONS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
