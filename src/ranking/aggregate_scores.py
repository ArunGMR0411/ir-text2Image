import sys
import os
import math
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y_%b_%d_%H_%M").upper()
LOG_FILE = os.path.join(LOG_DIR, f"STEP_34_aggregate_scores_{TIMESTAMP}.log")

import logging
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

def aggregate_scores(
    bm25_scores:        dict[str, float],
    visual_scores:      dict[str, float],
    transcript_scores:  dict[str, float],
    caption_scores:     dict[str, float]
) -> dict[str, tuple[float, float, float, float]]:
    """
    Merges 4 independent score dicts into a single per-frame tuple.
    Returns: { frame_id: (bm25, visual, transcript, caption) }
    Missing scores are imputed as 0.0.
    """
    # union all frame_ids so every frame that appeared in any stream is represented
    all_frames = set(bm25_scores.keys()) | set(visual_scores.keys()) | set(transcript_scores.keys()) | set(caption_scores.keys())
    
    aggregated = {}
    for fid in all_frames:
        aggregated[fid] = (
            float(bm25_scores.get(fid, 0.0)),
            float(visual_scores.get(fid, 0.0)),
            float(transcript_scores.get(fid, 0.0)),
            float(caption_scores.get(fid, 0.0))
        )
    return aggregated

if __name__ == "__main__":
    start_time = datetime.now()
    print_log("=" * 80)
    print_log(f"[STAGE 34] START — {TIMESTAMP}")
    print_log("Script  : src/aggregate_scores.py")
    print_log("=" * 80)

    print_log("INIT     Loading synthetic data for validation...")
    
    bm25_mock = {"frame_1": 0.5, "frame_all": 0.9}
    visual_mock = {"frame_2": 0.8, "frame_all": 0.8}
    transcript_mock = {"frame_all": 0.7}
    caption_mock = {"frame_all": 0.6}
    
    res = aggregate_scores(bm25_mock, visual_mock, transcript_mock, caption_mock)
    
    assertions = []
    
    expected_keys = {"frame_1", "frame_2", "frame_all"}
    v1 = set(res.keys()) == expected_keys
    assertions.append(("1. Union of all frame_ids in output == union of all frame_ids across 4 inputs", v1))
    
    v2 = all(isinstance(v, tuple) and len(v) == 4 and all(isinstance(x, float) for x in v) for v in res.values())
    assertions.append(("2. Every tuple has exactly 4 elements, all floats", v2))
    
    v3 = res.get("frame_2") == (0.0, 0.8, 0.0, 0.0)
    assertions.append(("3. A frame present only in visual_scores has (0.0, score, 0.0, 0.0)", v3))
    
    v4 = res.get("frame_all") == (0.9, 0.8, 0.7, 0.6)
    assertions.append(("4. A frame present in all 4 dicts retains all 4 original scores exactly", v4))
    
    v5 = aggregate_scores({}, {}, {}, {}) == {}
    assertions.append(("5. aggregate_scores({}, {}, {}, {}) == {}", v5))
    
    v6 = all(all(math.isfinite(x) for x in v) for v in res.values())
    assertions.append(("6. No tuple contains None or NaN — use math.isfinite() check", v6))

    print_log("START    Running validation assertions")
    print_log("PROG     100% done | 6 assertions checked | ETA ~0 min")
    
    all_pass = all(res for _, res in assertions)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print_log(f"DONE     6 assertions checked | {sum(res for _, res in assertions)} succeeded | {len(assertions) - sum(res for _, res in assertions)} failed | {duration:.4f}s total")
    
    print_log("-" * 80)
    print_log("VALIDATION")
    for msg, res_val in assertions:
        status = "PASS" if res_val else "FAIL"
        print_log(f"  {status} | {msg}")
    print_log("-" * 80)
    
    script_size = os.path.getsize(os.path.abspath(__file__))
    log_size = os.path.getsize(LOG_FILE) if os.path.exists(LOG_FILE) else 0
    
    print_log("ARTIFACTS")
    print_log(f"  src/aggregate_scores.py        {script_size} B")
    print_log(f"  logs/STEP_34_aggregate_scores_{TIMESTAMP}.log   {log_size} B (pre-close)")
    
    print_log("=" * 80)
    print_log(f"[STAGE 34] COMPLETE — {TIMESTAMP}")
    print_log("=" * 80)
    
    sys.exit(0 if all_pass else 1)
