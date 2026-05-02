"""
Pool top-20 results per query across all 3 approaches for manual relevance annotation.

For each query Q1..Q10:
1. Take top-20 from approach_a, approach_b, approach_c (slice [:20])
2. Union all frame_ids with deduplication
3. For each unique frame_id, record:
   - query_id, frame_id, stream_name
   - best_score: highest score across any approach
   - appeared_in: comma-separated list of approaches
   - full_path: image path from manifest.jsonl

Output:
- evaluation/pooled_candidates.csv (with is_relevant column for manual annotation)
- evaluation/ground_truth.csv (header only)
- logs/pool_candidates_[YYYYMMDD_HHMM].log
"""

import json
import csv
import os
from datetime import datetime
from collections import defaultdict


def load_all_results(results_path: str) -> dict:
    """Load all_results.json."""
    with open(results_path, 'r') as f:
        return json.load(f)


def iter_manifest_records(manifest_path: str):
    """Yield manifest rows from either CSV or JSONL format."""
    if manifest_path.endswith(".csv"):
        with open(manifest_path, "r", encoding="utf-8", newline="") as f:
            yield from csv.DictReader(f)
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def resolve_manifest_path() -> str:
    # try several candidate locations in order of preference
    candidate_paths = [
        'index/ingestion/manifest.csv',
        'index/ingestion/manifest.jsonl',
        'index/manifest.csv',
        'index/manifest.jsonl',
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No manifest file found. Tried: {candidate_paths}")


def load_manifest_for_paths(faiss_map_path: str, manifest_path: str) -> dict:
    """Load faiss_row_map.jsonl and manifest to create complete mapping."""
    frame_to_path = {}
    frame_to_stream = {}
    
    # build stream_type lookup from manifest so we know which path template to use
    stream_to_type = {}
    for entry in iter_manifest_records(manifest_path):
        stream = entry['stream_name']
        if stream not in stream_to_type:
            stream_to_type[stream] = entry.get('stream_type', 'unknown')
    
    with open(faiss_map_path, 'r') as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                frame_id = entry['frame_id']
                
                parts = frame_id.split('__')
                stream = parts[0]
                day = parts[1]
                filename = parts[2]
                
                stream_type = stream_to_type.get(stream, 'unknown')
                
                # fixed cameras live under a different directory than member egocentric streams
                if stream_type == 'fixed':
                    full_path = f"dataset/{day}/fixed/{stream}/{filename}.webp"
                else:
                    full_path = f"dataset/{day}/members/{stream}/{filename}.webp"
                
                frame_to_path[frame_id] = full_path
                frame_to_stream[frame_id] = stream
    
    return frame_to_path, frame_to_stream


def pool_candidates_for_query(
    query_id: str,
    query_data: dict,
    frame_to_path: dict,
    frame_to_stream: dict,
    logger
) -> list:
    """
    Pool top-20 results from all approaches for a single query.
    
    Returns list of dicts with: query_id, frame_id, stream_name, best_score, 
    appeared_in, full_path, is_relevant
    """
    all_results = []
    approach_names = ['approach_a', 'approach_b', 'approach_c']
    
    for approach in approach_names:
        if approach in query_data:
            results = query_data[approach][:20]
            for r in results:
                all_results.append({
                    'frame_id': r['frame_id'],
                    'score': r['score'],
                    'stream_name': r.get('stream_name', ''),
                    'approach': approach
                })
    
    # keep the highest score seen for each frame across all approaches
    frame_data = defaultdict(lambda: {'best_score': -1, 'approaches': set(), 'stream_name': ''})
    
    for r in all_results:
        fid = r['frame_id']
        if r['score'] > frame_data[fid]['best_score']:
            frame_data[fid]['best_score'] = r['score']
            frame_data[fid]['stream_name'] = r['stream_name']
        frame_data[fid]['approaches'].add(r['approach'])
    
    rows = []
    for frame_id, data in frame_data.items():
        full_path = frame_to_path.get(frame_id, '')
        
        if not full_path:
            parts = frame_id.split('__')
            if len(parts) >= 3:
                stream = parts[0]
                day = parts[1]
                filename = parts[2]
                full_path = ''
        
        stream_name = data['stream_name'] or frame_to_stream.get(frame_id, '')
        
        appeared_in = ','.join(sorted(data['approaches']))
        
        rows.append({
            'query_id': query_id,
            'frame_id': frame_id,
            'stream_name': stream_name,
            'best_score': data['best_score'],
            'appeared_in': appeared_in,
            'full_path': full_path,
            'is_relevant': ''  # left blank for manual annotation
        })
    
    # sort by best score so annotators see the most likely relevant frames first
    rows.sort(key=lambda x: x['best_score'], reverse=True)
    
    logger.info(f"Query {query_id}: {len(rows)} unique candidates from {len(all_results)} total results")
    return rows


def count_overlaps(query_data: dict, approach_names: list) -> dict:
    """Count overlap statistics between approaches."""
    sets = {}
    for approach in approach_names:
        if approach in query_data:
            frame_ids = set(r['frame_id'] for r in query_data[approach][:20])
            sets[approach] = frame_ids
    
    overlaps = {}
    for i, a1 in enumerate(approach_names):
        for a2 in approach_names[i+1:]:
            if a1 in sets and a2 in sets:
                key = f"{a1}_∩_{a2}"
                overlaps[key] = len(sets[a1] & sets[a2])
    
    if all(a in sets for a in approach_names):
        overlaps['all_three'] = len(sets[approach_names[0]] & sets[approach_names[1]] & sets[approach_names[2]])
    
    return overlaps


def main():
    results_path = 'evaluation/all_results.json'
    manifest_path = resolve_manifest_path()
    output_dir = 'evaluation'
    logs_dir = 'logs'
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    log_path = f'{logs_dir}/pool_candidates_{timestamp}.log'
    
    class SimpleLogger:
        def __init__(self, path):
            self.path = path
            self.lines = []
        
        def info(self, msg):
            self.lines.append(msg)
        
        def warning(self, msg):
            self.lines.append(msg)
        
        def save(self):
            with open(self.path, 'w') as f:
                f.write('\n'.join(self.lines))
    
    logger = SimpleLogger(log_path)
    logger.info(f"Pool Candidates Script - Started at {datetime.now().isoformat()}")
    logger.info(f"Results: {results_path}")
    logger.info(f"Manifest: {manifest_path}")
    
    faiss_map_path = 'index/indexing/faiss_row_map.jsonl'
    
    logger.info("Loading all_results.json...")
    all_results = load_all_results(results_path)
    
    query_ids = sorted([k for k in all_results.keys() if k.startswith('Q')], 
                       key=lambda x: int(x[1:]))
    logger.info(f"Found queries: {query_ids}")
    
    logger.info("Loading manifest for path lookups...")
    frame_to_path, frame_to_stream = load_manifest_for_paths(faiss_map_path, manifest_path)
    logger.info(f"Loaded {len(frame_to_path)} frame mappings")
    
    all_rows = []
    query_counts = {}
    approach_names = ['approach_a', 'approach_b', 'approach_c']
    total_results_before_dedup = 0
    
    for query_id in query_ids:
        query_data = all_results[query_id]
        
        overlaps = count_overlaps(query_data, approach_names)
        total_approaches = sum(1 for a in approach_names if a in query_data)
        
        logger.info(f"--- Query {query_id} ---")
        for approach in approach_names:
            if approach in query_data:
                logger.info(f"  {approach}: {len(query_data[approach][:20])} results (top 20)")
                total_results_before_dedup += len(query_data[approach][:20])
        
        for overlap_key, count in overlaps.items():
            logger.info(f"  {overlap_key}: {count}")
        
        rows = pool_candidates_for_query(query_id, query_data, frame_to_path, frame_to_stream, logger)
        all_rows.extend(rows)
        query_counts[query_id] = len(rows)
    
    output_csv = f'{output_dir}/pooled_candidates.csv'
    fieldnames = ['query_id', 'frame_id', 'stream_name', 'best_score', 'appeared_in', 'full_path', 'is_relevant']
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    
    logger.info(f"\nWrote {len(all_rows)} rows to {output_csv}")
    
    # create the ground truth file with just the header — to be filled in manually
    gt_csv = f'{output_dir}/ground_truth.csv'
    with open(gt_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['query_id', 'frame_id', 'is_relevant'])
    
    logger.info(f"Wrote header to {gt_csv}")
    
    logger.info("\n=== Summary ===")
    logger.info(f"Total unique candidates: {len(all_rows)}")
    logger.info(f"Total results before dedup: {total_results_before_dedup}")
    logger.info("\nPer-query candidate counts:")
    for qid in query_ids:
        logger.info(f"  {qid}: {query_counts[qid]}")
    
    logger.info("\n=== Validation ===")
    
    empty_path_count = sum(1 for r in all_rows if not r['full_path'])
    if empty_path_count > 0:
        logger.warning(f"WARNING: {empty_path_count} rows have empty full_path!")
    else:
        logger.info("PASS: All rows have non-empty full_path")
    
    # check for duplicate (query_id, frame_id) pairs which would indicate a pooling error
    seen = set()
    dup_count = 0
    for r in all_rows:
        key = (r['query_id'], r['frame_id'])
        if key in seen:
            dup_count += 1
        seen.add(key)
    
    if dup_count > 0:
        logger.warning(f"WARNING: Found {dup_count} duplicate (query_id, frame_id) pairs!")
    else:
        logger.info("PASS: No duplicate (query_id, frame_id) pairs")
    
    valid_approaches = set(approach_names)
    invalid_appeared = []
    for r in all_rows:
        for app in r['appeared_in'].split(','):
            if app and app not in valid_approaches:
                invalid_appeared.append((r['query_id'], r['frame_id'], app))
    
    if invalid_appeared:
        logger.warning(f"WARNING: Found {len(invalid_appeared)} rows with invalid approach names")
    else:
        logger.info("PASS: All appeared_in values are valid")
    
    expected_min, expected_max = 150, 250
    if expected_min <= len(all_rows) <= expected_max:
        logger.info(f"PASS: Row count {len(all_rows)} is within expected range ({expected_min}-{expected_max})")
    else:
        logger.warning(f"WARNING: Row count {len(all_rows)} is outside expected range ({expected_min}-{expected_max})")
    
    logger.info(f"\nLog saved to: {log_path}")
    logger.info(f"Completed at {datetime.now().isoformat()}")
    
    logger.save()


if __name__ == '__main__':
    main()
