"""
QAF Decimate: Filter frames based on tier and similarity.

Process each stream independently. For each stream, iterate through frames
in chronological order and apply decimation rules:
- HOT: always keep
- WARM: keep if pHash Hamming distance >= 8 from previous kept frame
- COLD: keep if pHash Hamming distance >= 12 from previous kept frame
"""

import json
import os
from collections import defaultdict
from pathlib import Path

import imagehash
from PIL import Image
from tqdm import tqdm


def extract_stream_info(frame_id: str, full_path_resized: str) -> tuple:
    """Extract stream name and day from frame_id and path."""
    stream_name = frame_id.split("__")[0]
    
    if "/day" in full_path_resized:
        day = full_path_resized.split("/day")[1].split("/")[0]
    else:
        day = "day1"
    
    return stream_name, day


def extract_sort_key(frame_id: str) -> tuple:
    """Extract chronological sort key from frame_id.
    
    Format: stream__HH_XXXX where HH is hour and XXXX is frame index.
    Returns (hour, frame_index) for sorting.
    """
    parts = frame_id.split("__")
    if len(parts) >= 2:
        frame_part = parts[1]
        hour = int(frame_part.split("_")[0])
        frame_index = int(frame_part.split("_")[1])
        return (hour, frame_index)
    return (0, 0)


def compute_phash(image_path: str) -> imagehash.ImageHash:
    """Compute pHash for an image."""
    img = Image.open(image_path)
    return imagehash.phash(img)


def load_classified_frames(jsonl_path: str) -> list:
    """Load all classified frames from JSONL file."""
    frames = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            frames.append(json.loads(line.strip()))
    return frames


def group_by_stream(frames: list) -> dict:
    """Group frames by stream (stream_name + day combination)."""
    streams = defaultdict(list)
    for frame in frames:
        stream_name, day = extract_stream_info(frame['frame_id'], frame['full_path_resized'])
        stream_key = f"{stream_name}_{day}"
        streams[stream_key].append(frame)
    return streams


def process_stream(stream_key: str, frames: list, base_path: str) -> list:
    """Process a single stream and return kept frames."""
    # sort chronologically before applying the dedup window
    sorted_frames = sorted(frames, key=lambda f: extract_sort_key(f['frame_id']))
    
    kept_frames = []
    prev_hash = None
    
    for frame in sorted_frames:
        tier = frame['tier']
        
        # hot frames are always kept regardless of visual similarity
        if tier == 'hot':
            kept_frames.append(frame)
        else:
            image_path = os.path.join(base_path, frame['full_path_resized'])
            
            if not os.path.exists(image_path):
                # keep the frame if we can't load it — better to over-include
                kept_frames.append(frame)
                prev_hash = None
                continue
            
            try:
                current_hash = compute_phash(image_path)
                
                if prev_hash is None:
                    kept_frames.append(frame)
                else:
                    distance = current_hash - prev_hash
                    
                    # warm frames need less visual change to be kept than cold frames
                    if tier == 'warm':
                        threshold = 8
                    else:
                        threshold = 12
                    
                    # keep only frames that are visually distinct enough from the previous kept frame
                    if distance >= threshold:
                        kept_frames.append(frame)
                
                prev_hash = current_hash
                
            except Exception as e:
                # keep the frame if hashing fails — better to over-include
                kept_frames.append(frame)
                prev_hash = None
    
    return kept_frames


def main():
    base_path = "/home/arun-gmr/MSC_Projects/MOS"
    input_jsonl = os.path.join(base_path, "index/indexing/qaff_classified.jsonl")
    output_jsonl = os.path.join(base_path, "index/indexing/upload_manifest.jsonl")
    audit_log_path = os.path.join(base_path, "audit_log.md")
    
    all_frames = load_classified_frames(input_jsonl)
    total_input = len(all_frames)
    
    input_tier_counts = {'hot': 0, 'warm': 0, 'cold': 0}
    for frame in all_frames:
        tier = frame['tier']
        if tier in input_tier_counts:
            input_tier_counts[tier] += 1
    
    # group frames by stream so we process each camera independently
    streams = group_by_stream(all_frames)
    
    all_kept_frames = []
    output_tier_counts = {'hot': 0, 'warm': 0, 'cold': 0}
    
    for stream_key, stream_frames in tqdm(streams.items(), desc="Streams"):
        kept = process_stream(stream_key, stream_frames, base_path)
        all_kept_frames.extend(kept)
        
        for frame in kept:
            tier = frame['tier']
            if tier in output_tier_counts:
                output_tier_counts[tier] += 1
    
    with open(output_jsonl, 'w') as f:
        for frame in all_kept_frames:
            f.write(json.dumps(frame) + '\n')
    
    total_kept = len(all_kept_frames)
    total_dropped = total_input - total_kept
    
    # rough storage estimate assuming 45 KB per frame
    estimated_gb = (total_kept * 45 * 1024) / (1024 * 1024 * 1024)
    
    # hot frames must all survive — they were explicitly flagged as high-relevance
    assert output_tier_counts['hot'] == input_tier_counts['hot'], \
        f"HOT frame count mismatch! Input: {input_tier_counts['hot']}, Output: {output_tier_counts['hot']}"
    
    audit_entry = f"""## QAF Decimate Run - {Path(input_jsonl).stat().st_mtime}

**Input:** {input_jsonl}
**Output:** {output_jsonl}

- Total input frames: {total_input}
- Total kept: {total_kept}
- Total dropped: {total_dropped}

- HOT: {input_tier_counts['hot']}
- WARM: {input_tier_counts['warm']}
- COLD: {input_tier_counts['cold']}

- HOT: {output_tier_counts['hot']}
- WARM: {output_tier_counts['warm']}
- COLD: {output_tier_counts['cold']}

- {estimated_gb:.2f} GB (assuming 45KB per frame)

- HOT frames preserved: {output_tier_counts['hot']} / {input_tier_counts['hot']} (100%)
- Streams processed: {len(streams)}

"""
    
    with open(audit_log_path, 'a') as f:
        f.write(audit_entry)
    


if __name__ == "__main__":
    main()
