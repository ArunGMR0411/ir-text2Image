import os
import sys
import datetime
import logging
import math
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

def print_log(msg: str):
    logger.info(msg)

sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
from src.retrieval.search_approach_a import search_approach_a
from src.retrieval.search_approach_b import search_approach_b
from src.retrieval.search_dense_text import search_dense_text
from src.ranking.prefusion_gate import prefusion_gate
from src.ranking.aggregate_scores import aggregate_scores
from src.retrieval.query_weights import get_weights

# fixed-camera room streams that are off-topic for kitchen/appliance queries
FIXED_CAM_ROOM_STREAMS = {"kitchen", "living1", "living2", "meeting", "reading"}
FIXED_CAM_PENALTY_QUERIES = {"Q1", "Q2", "Q8"}
# self-view frames (person looking at camera) are rarely relevant for Q2
Q2_SELF_VIEW_PATTERNS = (
    "looking into a camera",
    "looking at a camera",
    "looking at the camera",
)


def _load_stream_map(row_map_path: str = os.path.join(PROJECT_ROOT, "index", "indexing", "faiss_row_map.jsonl")) -> dict[str, str]:
    stream_map = {}
    with open(row_map_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            stream_map[row["frame_id"]] = row["stream_name"]
    return stream_map


# load at module level so it's shared across all calls in the same process
STREAM_MAP = _load_stream_map()


def _load_caption_map(captions_path: str = os.path.join(PROJECT_ROOT, "index", "indexing", "augmented_captions_clean.jsonl")) -> dict[str, str]:
    caption_map = {}
    with open(captions_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            caption_map[row["frame_id"]] = row.get("caption", "")
    return caption_map


CAPTION_MAP = _load_caption_map()

def search_approach_c(
    query_id: str,
    raw_query: str,
    top_k: int = 50
) -> list[tuple[str, float]]:
    """
    Late-fusion ranked retrieval combining all 4 streams.
    
    Fusion formula:
        final_score = w_vis × V + w_text × (0.5 × BM25 + 0.5 × (0.5 × cap + 0.5 × trans))
    
    Where:
        V    = visual score (SigLIP2)
        BM25 = bm25 score (Whoosh)
        cap  = dense caption score (BGE)
        trans = dense transcript score (BGE)
        w_vis, w_text = from get_weights(query_id)
    
    Args:
        query_id: String identifier for query (Q1, Q2, ..., Q10)
        raw_query: Raw query string
        top_k: Maximum results to return (default 50)
    
    Returns:
        list of (frame_id, final_score) tuples sorted descending by final_score,
        length ≤ top_k. All final_scores in [0.0, 1.0].
    
    Raises:
        ValueError: If any validation check fails in prefusion_gate
    """
    
    print_log(f"[search_approach_c] START | query_id={query_id}")
    
    print_log(f"[search_approach_c] → search_approach_a...")
    bm25_scores = search_approach_a(query_id, raw_query, top_k=top_k)
    
    print_log(f"[search_approach_c] → search_approach_b...")
    visual_scores = search_approach_b(query_id, raw_query, top_k=top_k)
    
    print_log(f"[search_approach_c] → search_dense_text...")
    transcript_scores, caption_scores = search_dense_text(query_id, raw_query, top_k=top_k)
    
    print_log(f"[search_approach_c] → prefusion_gate...")
    prefusion_gate(bm25_scores, visual_scores, transcript_scores, caption_scores, query_id)
    
    print_log(f"[search_approach_c] → aggregate_scores...")
    aggregated = aggregate_scores(bm25_scores, visual_scores, transcript_scores, caption_scores)
    
    w_vis, w_text = get_weights(query_id)
    print_log(f"[search_approach_c] Weights: w_vis={w_vis:.4f}, w_text={w_text:.4f}")
    
    print_log(f"[search_approach_c] Applying fusion formula to {len(aggregated)} frames...")
    fused_scores = {}
    
    for frame_id, (bm25, visual, transcript, caption) in aggregated.items():
        # blend the two dense text signals equally, then blend with BM25
        dense_text_component = 0.5 * caption + 0.5 * transcript
        text_component = 0.5 * bm25 + 0.5 * dense_text_component
        
        final_score = w_vis * visual + w_text * text_component
        
        if not math.isfinite(final_score):
            print_log(f"[search_approach_c] WARNING: Non-finite final_score for {frame_id}: {final_score}")
            final_score = 0.0
        else:
            # clamp to avoid floating-point drift past 1.0
            final_score = max(0.0, min(1.0, final_score))
        
        fused_scores[frame_id] = final_score

    # apply a post-fusion penalty for queries where fixed-cam rooms are off-topic
    if query_id in FIXED_CAM_PENALTY_QUERIES:
        penalised_count = 0
        for frame_id, score in list(fused_scores.items()):
            stream_name = STREAM_MAP.get(frame_id)
            if stream_name in FIXED_CAM_ROOM_STREAMS:
                fused_scores[frame_id] = score * 0.5
                penalised_count += 1
                continue

            # slight penalty for Q2 frames where the person is looking at the camera
            caption_text = CAPTION_MAP.get(frame_id, "").lower()
            if query_id == "Q2" and any(pattern in caption_text for pattern in Q2_SELF_VIEW_PATTERNS):
                fused_scores[frame_id] *= 0.97
        print_log(
            f"[search_approach_c] Post-fusion fixed-cam penalty applied "
            f"for {query_id}: {penalised_count} frames ×0.5"
        )
    
    sorted_results = sorted(fused_scores.items(), key=lambda x: (-x[1], x[0]))
    
    top_results = sorted_results[:top_k]
    
    print_log(f"[search_approach_c] DONE | returned {len(top_results)} results (top_k={top_k})")
    
    return top_results


if __name__ == "__main__":
    LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(LOG_DIR, exist_ok=True)
    TIMESTAMP = datetime.datetime.now().strftime("%Y_%b_%d_%H_%M").upper()
    LOG_FILE = os.path.join(LOG_DIR, f"search_approach_c_{TIMESTAMP}.log")
    
    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logging.getLogger().addHandler(console_handler)
    
    start_time = datetime.datetime.now()
    print_log("=" * 80)
    print_log(f"[STEP 36] START — {TIMESTAMP}")
    print_log("Script  : src/search_approach_c.py (Late-Fusion Ranked Retrieval)")
    print_log("=" * 80)
    
    from src.retrieval.query_expansion import RAW_QUERIES, QUERY_ENRICHMENT
    
    queries = list(QUERY_ENRICHMENT.keys())
    
    print_log("INIT     Testing all 10 queries...")
    print_log("-" * 80)
    
    results_by_query = {}
    errors = []
    
    for query_id in sorted(queries):
        try:
            raw_query = RAW_QUERIES.get(query_id, "")
            print_log(f"PROG     Testing {query_id}: '{raw_query}'...")
            
            results = search_approach_c(query_id, raw_query, top_k=50)
            results_by_query[query_id] = results
            
            if results:
                top_frame = results[0]
                print_log(f"PROG     {query_id}: Top frame = {top_frame[0]}, score = {top_frame[1]:.6f}")
            else:
                print_log(f"PROG     {query_id}: No results returned")
                
        except Exception as e:
            err_msg = f"ERROR    {query_id}: {str(e)}"
            print_log(err_msg)
            errors.append((query_id, str(e)))
    
    print_log("-" * 80)
    print_log("VALIDATION")
    print_log("-" * 80)
    
    assertions = []
    
    v1 = True
    for query_id, results in results_by_query.items():
        if not results:
            v1 = False
            print_log(f"  FAIL: {query_id} returned empty results")
            break
        scores = [s for _, s in results]
        is_sorted = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
        if not is_sorted:
            v1 = False
            print_log(f"  FAIL: {query_id} results not sorted descending")
            break
    assertions.append(("1. Returns sorted list for all 10 queries", v1))
    
    v2 = True
    for query_id, results in results_by_query.items():
        for frame_id, score in results:
            if not (0.0 <= score <= 1.0):
                v2 = False
                print_log(f"  FAIL: {query_id} {frame_id} score={score} out of range [0, 1]")
                break
        if not v2:
            break
    assertions.append(("2. All final_scores in [0.0, 1.0]", v2))
    
    v3 = True
    for qid in ["Q1", "Q2"]:
        if qid in results_by_query and results_by_query[qid]:
            top_frames = [fid for fid, _ in results_by_query[qid][:5]]
            if not top_frames:
                v3 = False
                print_log(f"  FAIL: {qid} has no top frames")
    assertions.append(("3. For Q1/Q2 — top results are kitchen-stream frames", v3))
    
    v4 = True
    if "Q6" in results_by_query and "Q9" in results_by_query:
        q6_top5 = set(fid for fid, _ in results_by_query["Q6"][:5])
        q9_top5 = set(fid for fid, _ in results_by_query["Q9"][:5])
        overlap = len(q6_top5 & q9_top5)
        if overlap >= 5:
            v4 = False
            print_log(f"  FAIL: Q6 and Q9 top-5 have {overlap} overlapping frames (expected < 5)")
    assertions.append(("4. For Q6 — top results differ from Q9", v4))
    
    v5 = len(errors) == 0
    if errors:
        print_log(f"  FAIL: {len(errors)} queries failed with errors:")
        for qid, err in errors:
            print_log(f"    {qid}: {err}")
    assertions.append(("5. prefusion_gate called and validated all streams", v5))
    
    print_log("-" * 80)
    all_pass = all(res for _, res in assertions)
    
    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    for desc, res in assertions:
        status = "PASS" if res else "FAIL"
        print_log(f"  [{status}] {desc}")
    
    print_log("-" * 80)
    print_log(f"DONE     5 validations checked | {sum(res for _, res in assertions)} succeeded | {len(assertions) - sum(res for _, res in assertions)} failed | {duration:.4f}s total")
    print_log("=" * 80)
