"""
Parse all transcript JSON files and align with keyframes.

For each chunk:
- Calculate frame_index = round(start_sec / 5) + 1
- Calculate time_offset_sec = (frame_index - 1) × 5
- Build sliding window transcript_text (±10s = 20s window)
- Output: frame_id, day, stream_name, hour, frame_index, time_offset_sec, transcript_text, raw_text
"""

import json
import os
import glob
from collections import defaultdict
from tqdm import tqdm

# ±10 seconds around each chunk's start time
WINDOW_SIZE = 20


def parse_transcript_path(path: str) -> tuple:
    """Extract day, person_name, hour from transcript file path."""
    parts = path.split('/')
    day = parts[-3]
    person_name = parts[-2]
    hour = parts[-1].replace('.json', '')
    return day, person_name.lower(), hour


def calculate_frame_index(start_sec: float) -> int:
    """Calculate frame index from start_sec: frame_index = round(start_sec / 5) + 1"""
    return round(start_sec / 5) + 1


def calculate_time_offset(frame_index: int) -> int:
    """Calculate time offset: time_offset_sec = (frame_index - 1) × 5"""
    return (frame_index - 1) * 5


def build_sliding_window(chunks: list, current_idx: int, window_size: int) -> str:
    """Build transcript text by collecting chunks within ±window_size/2 of current chunk's start_sec."""
    if current_idx >= len(chunks):
        return ""
    
    current_start = chunks[current_idx]['start_sec']
    half_window = window_size / 2
    
    window_texts = []
    for chunk in chunks:
        chunk_start = chunk['start_sec']
        # include chunks that overlap with the window around the current chunk
        if abs(chunk_start - current_start) <= half_window:
            if chunk['text'].strip():
                window_texts.append(chunk['text'].strip())
    
    return ' '.join(window_texts)


def main():
    base_path = '/home/arun-gmr/MSC_Projects/MOS'
    transcripts_dir = os.path.join(base_path, 'dataset/transcripts')
    output_csv = os.path.join(base_path, 'index/retrieval/transcript_aligned.csv')
    
    transcript_files = glob.glob(f'{transcripts_dir}/**/*.json', recursive=True)
    
    # group chunks by stream so we can build sliding windows per stream
    stream_chunks = defaultdict(list)
    
    for tfile in tqdm(transcript_files):
        try:
            with open(tfile, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            continue
        
        day, stream_name, hour = parse_transcript_path(tfile)
        stream_key = f"{stream_name}_{day}_{hour}"
        
        for chunk in data.get('chunks', []):
            ts = chunk.get('timestamp')
            if not ts or len(ts) < 2:
                continue
            
            start_sec = ts[0]
            end_sec = ts[1]
            
            if start_sec is None or end_sec is None:
                continue
            # skip zero-duration or reversed chunks
            if end_sec <= start_sec:
                continue
            
            text = chunk.get('text', '') or ''
            if not text.strip():
                continue
            
            frame_index = calculate_frame_index(start_sec)
            time_offset_sec = calculate_time_offset(frame_index)
            
            stream_chunks[stream_key].append({
                'day': day,
                'stream_name': stream_name,
                'hour': hour,
                'frame_index': frame_index,
                'time_offset_sec': time_offset_sec,
                'text': text,
                'start_sec': start_sec,
            })
    
    
    output_rows = []
    
    for stream_key, chunks in tqdm(stream_chunks.items()):
        # sort by start time so the sliding window sees chunks in order
        chunks_sorted = sorted(chunks, key=lambda x: x['start_sec'])
        
        for idx, chunk in enumerate(chunks_sorted):
            transcript_text = build_sliding_window(chunks_sorted, idx, WINDOW_SIZE)
            
            frame_id = f"{chunk['stream_name']}__{chunk['day']}__{chunk['hour']}_{chunk['frame_index']:04d}"
            
            output_rows.append({
                'frame_id': frame_id,
                'day': chunk['day'],
                'stream_name': chunk['stream_name'],
                'hour': chunk['hour'],
                'frame_index': chunk['frame_index'],
                'time_offset_sec': chunk['time_offset_sec'],
                'transcript_text': transcript_text,
                'raw_text': chunk['text'],
            })
    
    import csv
    
    fieldnames = ['frame_id', 'day', 'stream_name', 'hour', 'frame_index', 
                  'time_offset_sec', 'transcript_text', 'raw_text']
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    
    for row in output_rows[:5]:
        pass


if __name__ == "__main__":
    main()
