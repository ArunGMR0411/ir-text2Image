import sys
import os
import math
import logging
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y_%b_%d_%H_%M").upper()
LOG_FILE = os.path.join(LOG_DIR, f"STEP_35_prefusion_gate_{TIMESTAMP}.log")

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

logger = logging.getLogger(__name__)

def print_log(msg: str):
    logger.info(msg)

def prefusion_gate(
    bm25_scores: dict[str, float],
    visual_scores: dict[str, float],
    transcript_scores: dict[str, float],
    caption_scores: dict[str, float],
    query_id: str
) -> None:
    """
    Validates all 4 score dicts before fusion.
    Raises ValueError on any violation.
    Logs all 4 stream maxima regardless of pass/fail.
    """
    streams = {
        "BM25": bm25_scores,
        "Visual": visual_scores,
        "Transcript": transcript_scores,
        "Caption": caption_scores
    }
    
    # log the max score per stream so we can spot normalisation issues at a glance
    maxima = {}
    for name, stream in streams.items():
        if stream:
            maxima[name] = max(stream.values())
        else:
            maxima[name] = 0.0
            
    max_str = f"Q-ID: {query_id:<5} | BM25 max: {maxima['BM25']:.4f} | Vis max: {maxima['Visual']:.4f} | Trans max: {maxima['Transcript']:.4f} | Cap max: {maxima['Caption']:.4f}"
    print_log(max_str)
    
    for name, stream in streams.items():
        if not stream:
            continue
            
        for val in stream.values():
            if val is None:
                raise ValueError(f"[prefusion_gate] {name} failed: Dict contains None | value=None")
            if not math.isfinite(val):
                raise ValueError(f"[prefusion_gate] {name} failed: Value is not finite | value={val}")
            # scores must be normalised to [0, 1] before fusion
            if not (-0.000001 <= val <= 1.000001):
                raise ValueError(f"[prefusion_gate] {name} failed: Value out of range [0, 1] | value={val}")
                
        current_max = max(stream.values())
        if current_max <= 0.0:
            raise ValueError(f"[prefusion_gate] {name} failed: Non-empty dict max <= 0.0 | value={current_max}")

if __name__ == "__main__":
    start_time = datetime.now()
    print_log("=" * 80)
    print_log(f"[STAGE 35] START — {TIMESTAMP}")
    print_log("Script  : src/prefusion_gate.py")
    print_log("=" * 80)

    sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
    from src.retrieval.search_approach_a import search_approach_a
    from src.retrieval.search_approach_b import search_approach_b
    from src.retrieval.search_dense_text import search_dense_text
    from src.retrieval.query_expansion import QUERY_ENRICHMENT, RAW_QUERIES
    
    queries = list(QUERY_ENRICHMENT.keys())
    
    print_log("INIT     Running real queries for validation...")
    print_log("-" * 80)
    print_log("STREAM MAXIMA TABLE")
    print_log("-" * 80)
    
    assertions = []
    
    try:
        for q_id in queries:
            raw_query = RAW_QUERIES[q_id]
            bm25 = search_approach_a(q_id, raw_query, top_k=50)
            vis = search_approach_b(q_id, raw_query, top_k=50)
            trans, cap = search_dense_text(q_id, raw_query, top_k=50)
            
            prefusion_gate(bm25, vis, trans, cap, q_id)
        
        assertions.append(("1. Gate passes all 10 real queries silently", True))
    except Exception as e:
        print_log(f"FAIL: Gate raised exception on real query: {e}")
        import traceback
        print_log(traceback.format_exc())
        assertions.append(("1. Gate passes all 10 real queries silently", False))
        
    print_log("-" * 80)
    
    try:
        # inject a score above 1.0 to verify the gate catches it
        bad_bm25 = {"frame_1": 1.5}
        good_vis = {"frame_1": 0.5}
        good_trans = {"frame_1": 0.5}
        good_cap = {"frame_1": 0.5}
        
        prefusion_gate(bad_bm25, good_vis, good_trans, good_cap, "Q_INJ")
        assertions.append(("2. Injection test raises ValueError for score=1.5", False))
    except ValueError as e:
        if "out of range" in str(e):
            assertions.append(("2. Injection test raises ValueError for score=1.5", True))
            print_log(f"Injection caught successfully: {e}")
        else:
            assertions.append(("2. Injection test raises ValueError for score=1.5", False))
    except Exception:
        assertions.append(("2. Injection test raises ValueError for score=1.5", False))

    print_log("START    Running validation assertions")
    print_log("PROG     100% done | 2 assertions checked | ETA ~0 min")
    
    all_pass = all(res for _, res in assertions)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print_log(f"DONE     2 assertions checked | {sum(res for _, res in assertions)} succeeded | {len(assertions) - sum(res for _, res in assertions)} failed | {duration:.2f}s total")
    
    print_log("-" * 80)
    print_log("VALIDATION")
    for msg, res_val in assertions:
        status = "PASS" if res_val else "FAIL"
        print_log(f"  {status} | {msg}")
    print_log("-" * 80)
    
    script_size = os.path.getsize(os.path.abspath(__file__))
    log_size = os.path.getsize(LOG_FILE) if os.path.exists(LOG_FILE) else 0
    
    print_log("ARTIFACTS")
    print_log(f"  src/prefusion_gate.py        {script_size} B")
    print_log(f"  logs/STEP_35_prefusion_gate_{TIMESTAMP}.log   {log_size} B (pre-close)")
    
    print_log("=" * 80)
    print_log(f"[STAGE 35] COMPLETE — {TIMESTAMP}")
    print_log("=" * 80)
    
    sys.exit(0 if all_pass else 1)
apply_prefusion_gate = prefusion_gate
