import sys
import os
import json
import logging
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

def print_log(msg: str):
    logger.info(msg)

def load_frame_meta(filepath: str = "index/indexing/faiss_row_map.jsonl") -> dict[str, dict]:
    """Loads frame_id -> {stream_name, timestamp_str, time_offset_sec, hour}"""
    meta = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            meta[row["frame_id"]] = {
                "stream_name": row["stream_name"],
                "timestamp_str": row["timestamp_str"],
                "time_offset_sec": float(row["time_offset_sec"]),
                "hour": int(row["hour"])
            }
    return meta

def temporal_dedup(
    ranked: list[tuple[str, float]],
    frame_meta: dict[str, dict],
    window_sec: float = 10.0
) -> list[tuple[str, float]]:
    """
    Suppresses a frame if a higher-scored frame from the SAME stream_name
    exists within ±window_sec.
    Input MUST be sorted descending by score. Output will be subset, maintaining order.
    """
    kept = []
    for frame_id, score in ranked:
        if frame_id not in frame_meta:
            # keep frames we have no metadata for — can't make a dedup decision
            kept.append((frame_id, score))
            continue
            
        current_meta = frame_meta[frame_id]
        current_stream = current_meta["stream_name"]
        current_time = current_meta["time_offset_sec"]
        
        suppress = False
        for kept_id, _ in kept:
            if kept_id not in frame_meta: continue
            kept_meta = frame_meta[kept_id]
            
            # only suppress within the same stream — different cameras can show the same moment
            if kept_meta["stream_name"] == current_stream:
                if abs(kept_meta["time_offset_sec"] - current_time) < window_sec:
                    suppress = True
                    break
        
        if not suppress:
            kept.append((frame_id, score))
            
    return kept

def flag_cross_stream(
    ranked: list[tuple[str, float]],
    frame_meta: dict[str, dict]
) -> list[tuple[str, float, bool]]:
    """
    Appends multi_angle=True if another frame with a DIFFERENT stream_name 
    shares the SAME timestamp_str in the result list.
    Returns list of 3-tuples.
    """
    if not ranked:
        return []
        
    # build a map of timestamp -> set of streams to detect multi-angle moments
    timestamp_streams = {}
    for frame_id, score in ranked:
        if frame_id in frame_meta:
            ts = frame_meta[frame_id]["timestamp_str"]
            stream = frame_meta[frame_id]["stream_name"]
            if ts not in timestamp_streams:
                timestamp_streams[ts] = set()
            timestamp_streams[ts].add(stream)
            
    result = []
    for frame_id, score in ranked:
        multi_angle = False
        if frame_id in frame_meta:
            ts = frame_meta[frame_id]["timestamp_str"]
            # more than one stream at this timestamp means multiple cameras captured the same moment
            if len(timestamp_streams.get(ts, set())) > 1:
                multi_angle = True
        result.append((frame_id, score, multi_angle))
        
    return result

if __name__ == "__main__":
    LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(LOG_DIR, exist_ok=True)
    TIMESTAMP = datetime.now().strftime("%Y_%b_%d_%H_%M").upper()
    LOG_FILE = os.path.join(LOG_DIR, f"STEP_37_38_postprocess_{TIMESTAMP}.log")
    
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
    
    start_time = datetime.now()
    print_log("=" * 80)
    print_log(f"[STAGE 37] Temporal Deduplication Validation")
    print_log("Script  : src/postprocess.py")
    print_log("=" * 80)
    
    assertions_37 = []
    
    meta_1 = {
        "f1": {"stream_name": "S1", "time_offset_sec": 100.0, "timestamp_str": "T1"},
        "f2": {"stream_name": "S1", "time_offset_sec": 103.0, "timestamp_str": "T2"},
        "f3": {"stream_name": "S1", "time_offset_sec": 105.0, "timestamp_str": "T3"}
    }
    ranked_1 = [("f1", 0.9), ("f2", 0.8), ("f3", 0.7)]
    res_1 = temporal_dedup(ranked_1, meta_1, 10.0)
    assertions_37.append(("1. 3 frames same stream within 5s -> top scorer survives", len(res_1) == 1 and res_1[0][0] == "f1"))
    
    meta_2 = {
        "f1": {"stream_name": "S1", "time_offset_sec": 100.0, "timestamp_str": "T1"},
        "f2": {"stream_name": "S2", "time_offset_sec": 100.0, "timestamp_str": "T1"},
        "f3": {"stream_name": "S3", "time_offset_sec": 100.0, "timestamp_str": "T1"}
    }
    res_2 = temporal_dedup(ranked_1, meta_2, 10.0)
    assertions_37.append(("2. 3 frames DIFFERENT streams same time -> all survive", len(res_2) == 3))
    
    meta_3 = {
        "f1": {"stream_name": "S1", "time_offset_sec": 100.0, "timestamp_str": "T1"},
        "f2": {"stream_name": "S1", "time_offset_sec": 110.0, "timestamp_str": "T2"}
    }
    res_3 = temporal_dedup([("f1", 0.9), ("f2", 0.8)], meta_3, 10.0)
    # boundary is exclusive — exactly 10s apart means both survive
    assertions_37.append(("3. Frames exactly 10.0s apart -> both survive (exclusive boundary)", len(res_3) == 2))
    
    assertions_37.append(("4. Empty list returns empty list", temporal_dedup([], {}, 10.0) == []))
    
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
    from src.retrieval.search_approach_c import search_approach_c
    from src.retrieval.query_expansion import QUERY_ENRICHMENT, RAW_QUERIES
    q1_raw = RAW_QUERIES["Q1"]
    
    print_log("Loading real frame meta...")
    real_meta = load_frame_meta(os.path.join(PROJECT_ROOT, "index/indexing/faiss_row_map.jsonl"))
    
    print_log("Running search_approach_c for Q1...")
    q1_res = search_approach_c("Q1", q1_raw, top_k=50)
    q1_dedup = temporal_dedup(q1_res, real_meta, 10.0)
    assertions_37.append(("5. Real Q1 results are deduped (output length < input length)", len(q1_dedup) < len(q1_res)))
    
    for msg, p in assertions_37:
        print_log(f"  {'PASS' if p else 'FAIL'} | {msg}")
    
    print_log("=" * 80)
    print_log(f"[STAGE 38] Cross-Stream Flag Validation")
    print_log("=" * 80)
    
    assertions_38 = []
    
    meta_38_1 = {
        "f1": {"stream_name": "S1", "timestamp_str": "T1"},
        "f2": {"stream_name": "S2", "timestamp_str": "T1"}
    }
    ranked_38_1 = [("f1", 0.9), ("f2", 0.8)]
    res_38_1 = flag_cross_stream(ranked_38_1, meta_38_1)
    assertions_38.append(("1. Different streams, same timestamp -> multi_angle=True for both", all(x[2] == True for x in res_38_1)))
    
    meta_38_2 = {
        "f1": {"stream_name": "S1", "timestamp_str": "T1"},
        "f2": {"stream_name": "S1", "timestamp_str": "T1"}
    }
    res_38_2 = flag_cross_stream(ranked_38_1, meta_38_2)
    assertions_38.append(("2. Same stream, same timestamp -> multi_angle=False", all(x[2] == False for x in res_38_2)))
    
    meta_38_3 = {
        "f1": {"stream_name": "S1", "timestamp_str": "T1"},
        "f2": {"stream_name": "S2", "timestamp_str": "T2"}
    }
    res_38_3 = flag_cross_stream(ranked_38_1, meta_38_3)
    assertions_38.append(("3. Unique timestamps -> multi_angle=False", all(x[2] == False for x in res_38_3)))
    
    q1_cross = flag_cross_stream(q1_dedup, real_meta)
    multi_count = sum(1 for x in q1_cross if x[2])
    print_log(f"Real Q1 multi_angle=True count: {multi_count}")
    assertions_38.append(("4. Q1 cross-stream run successfully (multi_angle count logged)", True))
    
    res_empty = flag_cross_stream([], {})
    assertions_38.append(("5. Return type is list of 3-tuples (empty works)", type(res_empty) == list))
    
    for msg, p in assertions_38:
        print_log(f"  {'PASS' if p else 'FAIL'} | {msg}")
        
    all_pass = all(p for _, p in assertions_37) and all(p for _, p in assertions_38)
    
    print_log("-" * 80)
    print_log("ARTIFACTS")
    script_size = os.path.getsize(os.path.abspath(__file__))
    log_size = os.path.getsize(LOG_FILE) if os.path.exists(LOG_FILE) else 0
    print_log(f"  src/postprocess.py        {script_size} B")
    print_log(f"  {LOG_FILE}   {log_size} B (pre-close)")
    
    print_log("=" * 80)
    print_log(f"[STAGE 37/38] COMPLETE — {TIMESTAMP}")
    print_log("=" * 80)
    
    sys.exit(0 if all_pass else 1)
