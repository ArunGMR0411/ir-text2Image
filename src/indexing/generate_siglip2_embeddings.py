import os
import json
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoProcessor
from collections import defaultdict
import time
import argparse

class WebPDataset(Dataset):
    def __init__(self, records, processor):
        self.records = records
        self.processor = processor

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        manifest_row_index = record['manifest_row_index']
        stream_name = record.get('stream_name', '').lower()
        day = record.get('day', '')
        filename_stem = record.get('filename', '')
        frame_id = record.get('frame_id', f"{stream_name}__{day}__{filename_stem}")
        full_path = record['full_path']

        try:
            img_bgr = cv2.imread(full_path)
            if img_bgr is None:
                raise ValueError(f"Failed to load image: {full_path}")
            
            # SigLIP2 uses 384px input
            img_resized = cv2.resize(img_bgr, (384, 384), interpolation=cv2.INTER_AREA)
            
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            
            pil_img = Image.fromarray(img_rgb)
        except Exception as e:
            # return a black frame so the batch doesn't break
            pil_img = Image.fromarray(np.zeros((384, 384, 3), dtype=np.uint8))

        inputs = self.processor(images=pil_img, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)

        return pixel_values, frame_id, manifest_row_index


def main(args):
    manifest_path = "index/manifest_resized.jsonl"
    shards_dir = "embeddings/siglip2_shards"
    os.makedirs(shards_dir, exist_ok=True)

    all_records = []
    with open(manifest_path, 'r') as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            rec = json.loads(line)
            rec['manifest_row_index'] = idx
            all_records.append(rec)

    if args.test_limit > 0:
        all_records = all_records[:args.test_limit]

    # group by stream+day so each shard covers one camera's footage for one day
    shards_dict = defaultdict(list)
    for rec in all_records:
        shard_key = f"{rec['stream_name']}__{rec['day']}"
        shards_dict[shard_key].append(rec)

    existing_npy = set([f for f in os.listdir(shards_dir) if f.endswith('.npy')])
    existing_jsonl = set([f for f in os.listdir(shards_dir) if f.endswith('.jsonl')])

    # a shard is complete only when both the .npy and .jsonl files exist
    pending_shards = []
    for shard_key in shards_dict.keys():
        if f"{shard_key}.npy" in existing_npy and f"{shard_key}.jsonl" in existing_jsonl:
            if not args.test_limit:
                pass
        else:
            pending_shards.append(shard_key)

    if not pending_shards:
        return

    model_name = "google/siglip2-so400m-patch14-384"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to("cuda")
    model.eval()

    # disable OpenCV threading — DataLoader manages its own workers
    cv2.setNumThreads(0)

    num_workers = 4
    batch_size = 16

    total_frames_processed = 0
    start_time = time.time()

    for shard_key in pending_shards:
        records = shards_dict[shard_key]
        dataset = WebPDataset(records, processor)
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            num_workers=num_workers, 
            pin_memory=True, 
            prefetch_factor=2,
            shuffle=False
        )

        shard_embeddings = []
        shard_indices = []

        shard_start_time = time.time()
        shard_failed = False

        # build a lookup so we can retrieve metadata by row index during the loop
        record_map = {r['manifest_row_index']: r for r in records}

        try:
            with torch.no_grad():
                with torch.autocast("cuda"):
                    for batch_idx, (pixel_values, frame_ids, manifest_row_indices) in enumerate(dataloader):
                        pixel_values = pixel_values.to("cuda")
                        
                        # use the vision encoder only — we don't need the text side
                        vision_outputs = model.vision_model(pixel_values=pixel_values)
                        pooler_output = vision_outputs.pooler_output
                        
                        # cast to float32 before saving — float16 can cause issues with FAISS
                        emb_np = pooler_output.cpu().to(torch.float32).numpy()
                        shard_embeddings.append(emb_np)
                        
                        for i in range(len(frame_ids)):
                            row_idx = manifest_row_indices[i].item()
                            rec = record_map[row_idx]
                            
                            shard_indices.append({
                                'frame_id': frame_ids[i],
                                'full_path': rec['full_path'],
                                'day': rec['day'],
                                'stream_name': rec['stream_name'],
                                'hour': rec['hour'],
                                'frame_index': rec['frame_index'],
                                'manifest_row_index': rec['manifest_row_index']
                            })
                            
        except Exception as e:
            shard_failed = True
            
        if not shard_failed:
            all_emb_np = np.concatenate(shard_embeddings, axis=0)
            npy_path = os.path.join(shards_dir, f"{shard_key}.npy")
            jsonl_path = os.path.join(shards_dir, f"{shard_key}.jsonl")
            
            np.save(npy_path, all_emb_np)
            with open(jsonl_path, 'w') as f:
                for idx_rec in shard_indices:
                    f.write(json.dumps(idx_rec) + '\n')
            
            shard_time = time.time() - shard_start_time
            fps = len(records) / shard_time if shard_time > 0 else 0
            total_frames_processed += len(records)
            cum_time = time.time() - start_time
            cum_fps = total_frames_processed / cum_time if cum_time > 0 else 0
            
        else:
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_limit", type=int, default=0, help="Process only the first N frames for testing")
    args = parser.parse_args()
    main(args)
