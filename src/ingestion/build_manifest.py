"""
build_manifest.py
-----------------
Recursively scans dataset/day1–day4 and enumerates every .webp file.
For each file records:
  - full_path       : relative path from project root
  - day             : day1 / day2 / day3 / day4
  - stream_type     : 'member' or 'fixed'
  - stream_name     : person name (e.g. 'bjorn') or room name (e.g. 'kitchen')
  - filename        : raw filename without extension (e.g. '08_0066')
  - hour            : integer hour of day (HH segment)
  - frame_index     : integer frame counter within the hour (NNNN segment)
  - time_offset_sec : approximate seconds into the hour  = (frame_index - 1) * 5
  - timestamp_str   : human-readable HH:MM:SS approximation

Output: index/ingestion/manifest.jsonl  (one JSON object per line)
        index/ingestion/manifest.csv    (flat CSV for quick inspection)

Filename convention (confirmed in audit_log.md v0.3):
  HH_NNNN.webp
    HH   = 2-digit hour of day (24h clock)
    NNNN = frame index within that hour (1-based, ~1 frame per 5 seconds)
  time_offset_sec ≈ (NNNN - 1) * 5
"""

import os
import csv
import json
import re
from pathlib import Path

DATASET_ROOT = Path("dataset")
OUTPUT_DIR   = Path("index/ingestion")
MANIFEST_JSONL = OUTPUT_DIR / "manifest.jsonl"
MANIFEST_CSV   = OUTPUT_DIR / "manifest.csv"

DAYS = ["day1", "day2", "day3", "day4"]
FILENAME_RE = re.compile(r"^(\d{2})_(\d{4})$")

# each frame represents approximately 5 seconds of footage
SECONDS_PER_FRAME = 5


def offset_to_hms(hour: int, offset_sec: int) -> str:
    """Return HH:MM:SS string for a given hour + intra-hour offset."""
    total = hour * 3600 + offset_sec
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def scan_directory(day: str, stream_type: str, stream_name: str,
                   dir_path: Path) -> list[dict]:
    """Return a list of record dicts for all .webp files in dir_path."""
    records = []
    for fpath in sorted(dir_path.glob("*.webp")):
        stem = fpath.stem
        m = FILENAME_RE.match(stem)
        if not m:
            continue
        hour        = int(m.group(1))
        frame_index = int(m.group(2))
        # frame_index is 1-based, so subtract 1 before multiplying
        offset_sec  = (frame_index - 1) * SECONDS_PER_FRAME

        records.append({
            "full_path"       : str(fpath),
            "day"             : day,
            "stream_type"     : stream_type,
            "stream_name"     : stream_name,
            "filename"        : stem,
            "hour"            : hour,
            "frame_index"     : frame_index,
            "time_offset_sec" : offset_sec,
            "timestamp_str"   : offset_to_hms(hour, offset_sec),
        })
    return records



def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_records: list[dict] = []

    for day in DAYS:
        day_path = DATASET_ROOT / day
        if not day_path.exists():
            continue

        # member streams are egocentric (wearable cameras)
        members_path = day_path / "members"
        if members_path.exists():
            for member_dir in sorted(members_path.iterdir()):
                if member_dir.is_dir():
                    recs = scan_directory(day, "member", member_dir.name,
                                          member_dir)
                    all_records.extend(recs)

        # fixed cameras are room-level exocentric views
        fixed_path = day_path / "fixed"
        if fixed_path.exists():
            for room_dir in sorted(fixed_path.iterdir()):
                if room_dir.is_dir():
                    recs = scan_directory(day, "fixed", room_dir.name,
                                          room_dir)
                    all_records.extend(recs)


    with open(MANIFEST_JSONL, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")

    if all_records:
        fieldnames = list(all_records[0].keys())
        with open(MANIFEST_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_records)


if __name__ == "__main__":
    main()
