import sys
import os
import math
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y_%b_%d_%H_%M").upper()
LOG_FILE = os.path.join(LOG_DIR, f"STEP_32_query_weights_{TIMESTAMP}.log")

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

def print_log(msg: str):
    logger.info(msg)

# visual-heavy queries get higher w_vis; speech/text queries get higher w_text
QUERY_WEIGHTS = {
    "Q1":  {"w_vis": 0.7, "w_text": 0.3},
    "Q2":  {"w_vis": 0.6, "w_text": 0.4},
    "Q3":  {"w_vis": 0.6, "w_text": 0.4},
    "Q4":  {"w_vis": 0.7, "w_text": 0.3},
    "Q5":  {"w_vis": 0.7, "w_text": 0.3},
    "Q6":  {"w_vis": 0.8, "w_text": 0.2},
    "Q7":  {"w_vis": 0.8, "w_text": 0.2},
    "Q8":  {"w_vis": 0.8, "w_text": 0.2},
    "Q9":  {"w_vis": 0.6, "w_text": 0.4},
    "Q10": {"w_vis": 0.8, "w_text": 0.2},
}

def get_weights(query_id: str) -> tuple[float, float]:
    """Returns (w_vis, w_text). Returns (0.5, 0.5) for unknown query_id."""
    if query_id not in QUERY_WEIGHTS:
        return 0.5, 0.5
    w = QUERY_WEIGHTS[query_id]
    return w["w_vis"], w["w_text"]

if __name__ == "__main__":
    start_time = datetime.now()
    print_log("=" * 80)
    print_log(f"[STAGE 32] START — {TIMESTAMP}")
    print_log("Script  : src/query_weights.py")
    print_log("=" * 80)

    print_log("INIT     Loading query weights...")
    
    assertions = []
    
    expected_keys = {f"Q{i}" for i in range(1, 11)}
    v1 = set(QUERY_WEIGHTS.keys()) == expected_keys and len(QUERY_WEIGHTS) == 10
    assertions.append(("1. Exactly 10 keys: Q1..Q10", v1))
    
    # weights must sum to 1.0 so the fusion formula stays in [0, 1]
    v2 = all(math.isclose(w["w_vis"] + w["w_text"], 1.0, rel_tol=1e-9) for w in QUERY_WEIGHTS.values())
    assertions.append(("2. w_vis + w_text == 1.0 for every query", v2))
    
    v3 = get_weights("Q4") == (0.2, 0.8)
    assertions.append(("3. get_weights('Q4') == (0.2, 0.8)", v3))
    
    v4 = get_weights("Q6") == (0.8, 0.2)
    assertions.append(("4. get_weights('Q6') == (0.8, 0.2)", v4))
    
    try:
        get_weights("Q99")
        v5 = False
    except KeyError:
        v5 = True
    except Exception:
        v5 = False
    assertions.append(("5. get_weights('Q99') raises KeyError", v5))

    print_log("START    Running validation assertions")
    print_log("PROG     100% done | 5 assertions checked | ETA ~0 min")
    
    all_pass = all(res for _, res in assertions)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print_log(f"DONE     5 assertions checked | {sum(res for _, res in assertions)} succeeded | {len(assertions) - sum(res for _, res in assertions)} failed | {duration:.4f}s total")
    
    print_log("-" * 80)
    print_log("VALIDATION")
    for msg, res in assertions:
        status = "PASS" if res else "FAIL"
        print_log(f"  {status} | {msg}")
    print_log("-" * 80)
    
    script_size = os.path.getsize(os.path.abspath(__file__))
    log_size = os.path.getsize(LOG_FILE) if os.path.exists(LOG_FILE) else 0
    
    print_log("ARTIFACTS")
    print_log(f"  src/query_weights.py        {script_size} B")
    print_log(f"  logs/STEP_32_query_weights_{TIMESTAMP}.log   {log_size} B (pre-close)")
    
    print_log("=" * 80)
    print_log(f"[STAGE 32] COMPLETE — {TIMESTAMP}")
    print_log("=" * 80)
    
    sys.exit(0 if all_pass else 1)
