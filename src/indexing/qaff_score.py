"""
qaff_score.py
-------------
Query-Aware Frame Filtering (QAFF) using CLIP ViT-B/32.
Scores all 416,542 resized frames against 10 assignment queries,
classifies into HOT/WARM/COLD tiers, and reports statistics.

Output files:
  index/qaff_scores_shard_*.jsonl   intermediate shards (50K frames each)
  index/indexing/qaff_scores.jsonl           merged full scores
  index/indexing/qaff_classified.jsonl       with tier labels
  index/indexing/qaff_summary.json           statistics report

This script is READ-ONLY with respect to dataset_resized/ — it never
modifies or deletes any image files.
"""

import json
import os
import time
import datetime
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
import clip
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


QUERIES = {
    1: "somebody using a portable electric kitchen gadget like a handheld blender or mixer",

    2: "somebody pressing buttons on or operating a coffee machine",

    3: "somebody unwrapping and eating a sweet snack like a chocolate bar or candy",

    4: "somebody smiling and complimenting the food giving a thumbs up",

    5: "people singing together with mouths open",

    6: "small yellow rubber or plush octopus toy",

    7: "squirrel christmas tree ornament hanging on a decorated tree",

    8: "metal bird shaped cookie cutter on a surface",

    9: "ace of spades black playing card face up",

    10: "partially eaten apple with bite marks on a table",
}

MANIFEST_PATH    = Path("index/manifest_resized.jsonl")
INDEX_DIR        = Path("index")
SHARD_PREFIX     = "qaff_scores_shard_"
# write to disk every 50k frames so a crash doesn't lose everything
SHARD_SIZE       = 50_000
SCORES_PATH      = INDEX_DIR / "qaff_scores.jsonl"
CLASSIFIED_PATH  = INDEX_DIR / "qaff_classified.jsonl"
SUMMARY_PATH     = INDEX_DIR / "qaff_summary.json"

BATCH_SIZE       = 128
NUM_WORKERS      = 12
PREFETCH_FACTOR  = 2
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

# frames above HOT are always uploaded; frames between WARM and HOT are decimated
THRESHOLD_HOT    = 0.20
THRESHOLD_WARM   = 0.12

EXPECTED_TOTAL   = 416_542
REPORT_EVERY     = 10_000



def encode_queries(model):
    query_list = [QUERIES[i] for i in sorted(QUERIES)]
    tokens = clip.tokenize(query_list).to(DEVICE)
    with torch.no_grad():
        vecs = model.encode_text(tokens).float()
        # normalise so cosine similarity = dot product
        vecs = vecs / vecs.norm(dim=-1, keepdim=True)
    assert vecs.shape == (10, 512), f"Unexpected shape: {vecs.shape}"
    return vecs



class FrameDataset(Dataset):
    def __init__(self, records: list[dict], preprocess):
        self.records   = records
        self.preprocess = preprocess

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        path = self.records[i]["full_path"]
        try:
            img = cv2.imread(path)
            if img is None:
                raise ValueError("cv2 returned None")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            from PIL import Image as PILImage
            pil = PILImage.fromarray(img)
            return self.preprocess(pil), i
        except Exception:
            # return a zero tensor so the batch doesn't break
            return torch.zeros(3, 224, 224), i



def score_batch(image_tensors: torch.Tensor,
                model,
                query_vectors: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      max_scores  : float32 array [B]  — max cosine sim across all 10 queries
      best_qids   : int array   [B]  — 1-based query id of best match
    """
    imgs = image_tensors.to(DEVICE, non_blocking=True)
    with torch.no_grad():
        feats = model.encode_image(imgs).float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
    sim = feats @ query_vectors.T
    max_scores, best_idx = sim.max(dim=1)
    return (max_scores.cpu().numpy().astype(np.float32),
            (best_idx.cpu().numpy() + 1).astype(np.int32))



def shard_path(n: int) -> Path:
    return INDEX_DIR / f"{SHARD_PREFIX}{n}.jsonl"


def load_completed_shards() -> tuple[set[int], int]:
    """Return set of already-processed manifest indices and next shard number."""
    done_indices: set[int] = set()
    shard_n = 0
    while shard_path(shard_n).exists():
        with open(shard_path(shard_n)) as f:
            for line in f:
                rec = json.loads(line)
                done_indices.add(rec["manifest_idx"])
        shard_n += 1
    return done_indices, shard_n



def merge_shards(n_shards: int) -> list[dict]:
    all_records = []
    for i in range(n_shards):
        with open(shard_path(i)) as f:
            for line in f:
                all_records.append(json.loads(line))
    return all_records


def classify(records: list[dict]) -> list[dict]:
    # assign tier based on the max relevance score across all 10 queries
    for r in records:
        s = r["relevance_score"]
        if s >= THRESHOLD_HOT:
            r["tier"] = "hot"
        elif s >= THRESHOLD_WARM:
            r["tier"] = "warm"
        else:
            r["tier"] = "cold"
    return records



def compute_stats(records: list[dict]) -> dict:
    total = len(records)
    hot   = [r for r in records if r["tier"] == "hot"]
    warm  = [r for r in records if r["tier"] == "warm"]
    cold  = [r for r in records if r["tier"] == "cold"]

    # count how many hot frames each query contributed
    hot_qid_dist: dict[int, int] = defaultdict(int)
    for r in hot:
        hot_qid_dist[r["best_query_id"]] += 1

    stream_scores: dict[str, list[float]] = defaultdict(list)
    for r in records:
        stream_scores[r["frame_id"].split("__")[0]].append(r["relevance_score"])
    stream_avg = {k: round(sum(v) / len(v), 5) for k, v in stream_scores.items()}
    sorted_streams = sorted(stream_avg.items(), key=lambda x: x[1], reverse=True)

    upload_count = len(hot) + len(warm)
    upload_gb    = round(upload_count * 45 / 1024 / 1024, 2)

    return {
        "total_frames"        : total,
        "hot_count"           : len(hot),
        "hot_pct"             : round(len(hot) / total * 100, 2),
        "warm_count"          : len(warm),
        "warm_pct"            : round(len(warm) / total * 100, 2),
        "cold_count"          : len(cold),
        "cold_pct"            : round(len(cold) / total * 100, 2),
        "hot_query_distribution": {str(k): v for k, v in sorted(hot_qid_dist.items())},
        "top5_streams_highest_avg": sorted_streams[:5],
        "top5_streams_lowest_avg" : sorted_streams[-5:][::-1],
        "estimated_upload_frames" : upload_count,
        "estimated_upload_gb"     : upload_gb,
    }



def print_histogram(records: list[dict]):
    bands = [(0.00, 0.05), (0.05, 0.10), (0.10, 0.15),
             (0.15, 0.20), (0.20, 0.25), (0.25, 0.30), (0.30, 1.01)]
    labels = ["0.00–0.05", "0.05–0.10", "0.10–0.15",
              "0.15–0.20", "0.20–0.25", "0.25–0.30", "0.30+   "]
    counts = [0] * len(bands)
    for r in records:
        s = r["relevance_score"]
        for j, (lo, hi) in enumerate(bands):
            if lo <= s < hi:
                counts[j] += 1
                break

    total = len(records)
    for label, count in zip(labels, counts):
        pct  = count / total * 100
        bar  = "█" * int(pct / 0.5)



def main():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    model.eval()

    query_vectors = encode_queries(model)

    all_records = []
    with open(MANIFEST_PATH) as f:
        for line in f:
            all_records.append(json.loads(line))

    done_indices, next_shard = load_completed_shards()
    if done_indices:
        pass  # resume from where we left off

    pending = [(i, r) for i, r in enumerate(all_records)
               if i not in done_indices]

    if pending:
        pending_records = [r for _, r in pending]
        pending_indices = [i for i, _ in pending]

        ds     = FrameDataset(pending_records, preprocess)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                            pin_memory=True, prefetch_factor=PREFETCH_FACTOR)

        shard_buf: list[dict] = []
        current_shard = next_shard
        total_scored  = len(done_indices)
        t0 = time.time()
        t_last = t0

        for batch_tensors, batch_local_idx in loader:
            max_scores, best_qids = score_batch(batch_tensors, model,
                                                query_vectors)

            for j, local_i in enumerate(batch_local_idx.tolist()):
                manifest_i = pending_indices[local_i]
                rec        = pending_records[local_i]
                stream     = rec.get("stream_name", "unknown")
                fname      = Path(rec["full_path"]).stem

                shard_buf.append({
                    "manifest_idx"      : manifest_i,
                    "frame_id"          : f"{stream}__{fname}",
                    "full_path_resized" : rec["full_path"],
                    "relevance_score"   : round(float(max_scores[j]), 4),
                    "best_query_id"     : int(best_qids[j]),
                })

            total_scored += len(batch_local_idx)

            # flush to disk when the buffer reaches the shard size
            if len(shard_buf) >= SHARD_SIZE:
                sp = shard_path(current_shard)
                with open(sp, "w") as f:
                    for r in shard_buf:
                        f.write(json.dumps(r) + "\n")
                shard_buf = []
                current_shard += 1

            if total_scored % REPORT_EVERY < BATCH_SIZE:
                now  = time.time()
                fps  = REPORT_EVERY / max(now - t_last, 1e-6)
                eta  = (EXPECTED_TOTAL - total_scored) / max(fps, 1)
                ts   = datetime.datetime.now().strftime("%H:%M:%S")
                t_last = now

        if shard_buf:
            sp = shard_path(current_shard)
            with open(sp, "w") as f:
                for r in shard_buf:
                    f.write(json.dumps(r) + "\n")
            current_shard += 1

        elapsed = time.time() - t0
    else:
        current_shard = next_shard

    scored = merge_shards(current_shard)
    assert len(scored) == EXPECTED_TOTAL, (
        f"Count mismatch: expected {EXPECTED_TOTAL:,}, got {len(scored):,}")

    scored = classify(scored)

    with open(SCORES_PATH, "w") as f:
        for r in scored:
            f.write(json.dumps(r) + "\n")

    with open(CLASSIFIED_PATH, "w") as f:
        for r in scored:
            f.write(json.dumps(r) + "\n")

    stats = compute_stats(scored)
    with open(SUMMARY_PATH, "w") as f:
        json.dump(stats, f, indent=2)

    for qid, cnt in sorted(stats["hot_query_distribution"].items(),
                           key=lambda x: int(x[0])):
        pass
    for stream, avg in stats["top5_streams_highest_avg"]:
        pass
    for stream, avg in stats["top5_streams_lowest_avg"]:
        pass

    print_histogram(scored)


if __name__ == "__main__":
    main()
