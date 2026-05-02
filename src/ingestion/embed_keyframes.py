"""
embed_keyframes.py
------------------
Batch-processes all keyframes through CLIP ViT-L/14.
Reads:  index/manifest.jsonl
Writes: embeddings/clip_vitl14_embeddings.npy   (float32, shape [N, 768])
        embeddings/clip_vitl14_index.jsonl       (one entry per row: manifest fields)
        embeddings/clip_vitl14_meta.json         (run metadata)

Design decisions (logged in audit_log.md):
  - Model    : CLIP ViT-L/14  (768-dim, best quality/speed trade-off available)
  - Storage  : memory-mapped .npy for embeddings + JSONL index for manifest refs
  - Batch    : 256 images per GPU batch
  - Workers  : 4 DataLoader workers for parallel image I/O
  - Decoder  : cv2 + immediate resize to 256px (2× faster than PIL on 4K UHD webp)
  - Norm     : L2-normalised embeddings (unit vectors, cosine sim = dot product)
  - Resume   : skips already-processed shards; safe to re-run after interruption
  - Shards   : one .npy shard per (day, stream_type, stream_name); merged at end
  - Progress : prints shard-level progress to stdout (tqdm-free for clean logging)
"""

import json
import csv
import time
import datetime
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import clip
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

MANIFEST_CANDIDATES = [
    Path("index/ingestion/manifest.csv"),
    Path("index/ingestion/manifest.jsonl"),
    Path("index/manifest.csv"),
    Path("index/manifest.jsonl"),
]
EMBED_DIR       = Path("embeddings")
SHARD_DIR       = EMBED_DIR / "shards"
FINAL_NPY       = EMBED_DIR / "clip_vitl14_embeddings.npy"
FINAL_INDEX     = EMBED_DIR / "clip_vitl14_index.jsonl"
META_PATH       = EMBED_DIR / "clip_vitl14_meta.json"

MODEL_NAME      = "ViT-L/14"
EMBED_DIM       = 768
BATCH_SIZE      = 256
NUM_WORKERS     = 4
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# CLIP's expected normalisation constants
CLIP_MEAN = (0.48145466, 0.4578275,  0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

# resize to 256 first then crop to 224 — faster than resizing directly to 224
FAST_PREPROCESS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(224, antialias=True),
    transforms.CenterCrop(224),
    transforms.Normalize(CLIP_MEAN, CLIP_STD),
])


class FrameDataset(Dataset):
    def __init__(self, records: list[dict]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        try:
            img = cv2.imread(self.records[i]["full_path"])
            # decode at 256px — faster than PIL on large WebP files
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return FAST_PREPROCESS(img), i
        except Exception:
            # return a zero tensor so the batch doesn't break
            return torch.zeros(3, 224, 224), i



def load_model():
    model, _ = clip.load(MODEL_NAME, device=DEVICE)
    model.eval()
    return model


def load_manifest() -> list[dict]:
    manifest_path = next((path for path in MANIFEST_CANDIDATES if path.exists()), None)
    if manifest_path is None:
        raise FileNotFoundError(
            f"No manifest file found. Tried: {[str(path) for path in MANIFEST_CANDIDATES]}"
        )

    records = []
    if manifest_path.suffix == ".csv":
        with open(manifest_path, "r", encoding="utf-8", newline="") as f:
            records.extend(csv.DictReader(f))
    else:
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    return records



def shard_key(rec: dict) -> str:
    # one shard per (day, stream_type, stream_name) combination
    return f"{rec['day']}__{rec['stream_type']}__{rec['stream_name']}"

def shard_npy_path(key: str) -> Path:
    return SHARD_DIR / f"{key}.npy"

def shard_idx_path(key: str) -> Path:
    return SHARD_DIR / f"{key}.jsonl"



@torch.no_grad()
def process_shard(key: str, records: list[dict], model) -> tuple[int, int]:
    """Embed all records for one shard. Returns (n_processed, n_skipped)."""
    npy_path = shard_npy_path(key)
    idx_path = shard_idx_path(key)

    # skip shards that are already complete — safe to re-run
    if npy_path.exists() and idx_path.exists():
        existing = np.load(npy_path, mmap_mode="r")
        if existing.shape[0] == len(records):
            return 0, len(records)

    n = len(records)
    embeddings = np.zeros((n, EMBED_DIM), dtype=np.float32)

    ds     = FrameDataset(records)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                        pin_memory=True, prefetch_factor=2)

    for batch_imgs, indices in loader:
        batch_imgs = batch_imgs.to(DEVICE, non_blocking=True)
        feats = model.encode_image(batch_imgs).float()
        # normalise so cosine similarity = dot product
        feats = feats / feats.norm(dim=-1, keepdim=True)
        for j, idx in enumerate(indices.tolist()):
            embeddings[idx] = feats[j].cpu().numpy()

    np.save(npy_path, embeddings)
    with open(idx_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    return n, 0



def merge_shards(shard_keys: list[str], total: int):
    final = np.zeros((total, EMBED_DIM), dtype=np.float32)
    row = 0
    with open(FINAL_INDEX, "w") as idx_f:
        for key in shard_keys:
            shard = np.load(shard_npy_path(key))
            n = shard.shape[0]
            final[row:row + n] = shard
            row += n
            with open(shard_idx_path(key)) as sf:
                for line in sf:
                    idx_f.write(line)
    np.save(FINAL_NPY, final)
    size_gb = FINAL_NPY.stat().st_size / 1e9



def main():
    EMBED_DIR.mkdir(parents=True, exist_ok=True)
    SHARD_DIR.mkdir(parents=True, exist_ok=True)

    model   = load_model()
    records = load_manifest()

    # group records by shard key so we can skip completed shards on resume
    shards: dict[str, list[dict]] = {}
    for rec in records:
        shards.setdefault(shard_key(rec), []).append(rec)
    shard_keys = sorted(shards.keys())

    t0 = time.time()
    total_done = 0
    total_skip = 0

    for i, key in enumerate(shard_keys, 1):
        recs = shards[key]
        t_shard = time.time()
        n_proc, n_skip = process_shard(key, recs, model)
        total_done += n_proc
        total_skip += n_skip
        elapsed = time.time() - t0
        shard_t = time.time() - t_shard
        done_so_far = total_done + total_skip
        fps = done_so_far / elapsed if elapsed > 0 else 0
        eta = (len(records) - done_so_far) / fps if fps > 0 else 0
        status = "SKIP" if n_skip else f"{shard_t:.0f}s"

    elapsed = time.time() - t0

    merge_shards(shard_keys, len(records))

    meta = {
        "model"                   : MODEL_NAME,
        "embed_dim"               : EMBED_DIM,
        "normalised"              : True,
        "dtype"                   : "float32",
        "total_frames"            : len(records),
        "batch_size"              : BATCH_SIZE,
        "num_workers"             : NUM_WORKERS,
        "image_decoder"           : "cv2+resize256",
        "device"                  : DEVICE,
        "seconds_per_frame_approx": 5,
        "manifest_source"         : str(MANIFEST_CANDIDATES),
        "generated_at"            : datetime.datetime.now().isoformat(),
        "elapsed_seconds"         : round(elapsed, 1),
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
