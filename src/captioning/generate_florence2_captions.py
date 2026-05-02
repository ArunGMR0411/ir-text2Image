import os
import json
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.dynamic_module_utils import check_imports
from collections import defaultdict
import time
import argparse
import logging
from datetime import datetime, timedelta

# disable OpenCV's internal thread pool — DataLoader manages its own workers
cv2.setNumThreads(0)

def get_shard_key(record):
    # prefer the frame_id prefix as the shard key for consistency
    frame_id = str(record.get('frame_id', '')).strip().lower()
    if frame_id and '__' in frame_id:
        parts = frame_id.split('__')
        if len(parts) >= 2 and parts[0] and parts[1]:
            return f"{parts[0]}__{parts[1]}"

    # fall back to stream_name + day when frame_id is absent
    stream_name = str(record.get('stream_name', '')).strip().lower()
    day = str(record.get('day', '')).strip().lower()
    if stream_name and day:
        return f"{stream_name}__{day}"

    return "__"

class FlorenceDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        manifest_row_index = record.get('manifest_row_index', idx)

        frame_id = record.get('frame_id')
        if not frame_id:
            shard_key = record.get('shard_key', get_shard_key(record))
            filename_stem = str(record.get('filename', manifest_row_index))
            frame_id = f"{shard_key}__{filename_stem}"
        
        full_path = record.get('full_path_resized', record.get('full_path'))

        try:
            img_bgr = cv2.imread(full_path)
            if img_bgr is None:
                raise ValueError(f"Failed to load image: {full_path}")
            
            # Florence-2 expects 768px input
            if img_bgr.shape[:2] != (768, 768):
                img_bgr = cv2.resize(img_bgr, (768, 768), interpolation=cv2.INTER_LANCZOS4)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
        except Exception as e:
            logging.error(f"Error loading {full_path}: {e}")
            # return a black frame so the batch doesn't break
            pil_img = Image.fromarray(np.zeros((768, 768, 3), dtype=np.uint8))

        return pil_img, frame_id, manifest_row_index

def custom_collate_fn(batch):
    pil_images = [item[0] for item in batch]
    frame_ids = [item[1] for item in batch]
    row_indices = [item[2] for item in batch]
    return pil_images, frame_ids, row_indices

def run_inference(processor, model, task_prompt, images, max_new_tokens, num_beams):
    prompts = [task_prompt] * len(images)
    inputs = processor(text=prompts, images=images, return_tensors="pt")
    inputs["input_ids"] = inputs["input_ids"].to("cuda")
    inputs["pixel_values"] = inputs["pixel_values"].to("cuda", torch.float16)
    
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=max_new_tokens,
        early_stopping=(num_beams > 1),
        do_sample=False,
        num_beams=num_beams,
        use_cache=True
    )
    
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    
    results = []
    for text, img in zip(generated_texts, images):
        parsed = processor.post_process_generation(text, task=task_prompt, image_size=(img.width, img.height))
        res = parsed.get(task_prompt, "")
        # some tasks return a dict instead of a plain string
        if isinstance(res, dict):
            res = json.dumps(res)
        results.append(str(res))
        
    return results

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    checkpoint_dir = args.checkpoint_dir or args.output_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    log_filename = f"logs/generate_florence2_captions_{datetime.now().strftime('%Y%m%d_%H%M')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Loading manifest from {args.manifest}")
    all_records = []
    with open(args.manifest, 'r') as f:
        for idx, line in enumerate(f):
            if not line.strip(): continue
            rec = json.loads(line)
            rec['manifest_row_index'] = idx
            rec['shard_key'] = get_shard_key(rec)
            all_records.append(rec)

    # one .jsonl file per shard — presence means that shard is done
    existing_jsonl = set([f for f in os.listdir(checkpoint_dir) if f.endswith('.jsonl')])
    existing_shard_keys = set([f[:-6] for f in existing_jsonl])

    if args.reprocess_existing:
        records_to_process = all_records
        logging.info("Reprocess mode enabled: existing shards will not be skipped.")
    else:
        records_to_process = [r for r in all_records if r['shard_key'] not in existing_shard_keys]
        logging.info(
            f"Skipping existing shards: {len(all_records) - len(records_to_process)} frames skipped, "
            f"{len(records_to_process)} frames pending"
        )

    if args.test_limit > 0:
        logging.info(f"TEST MODE: Limiting to first {args.test_limit} pending frames.")
        records_to_process = records_to_process[:args.test_limit]

    if args.batch_size > 1:
        logging.warning(
            "batch_size > 1 may increase throughput but can degrade Florence-2 generation quality. "
            "Use with validation before full run."
        )

    # group frames by shard so we write one file per stream/day pair
    shards_dict = defaultdict(list)
    for rec in records_to_process:
        shard_key = rec.get('shard_key', get_shard_key(rec))
        shards_dict[shard_key].append(rec)

    pending_shards = []
    for shard_key in shards_dict.keys():
        if f"{shard_key}.jsonl" in existing_jsonl and not args.reprocess_existing:
            logging.info(f"Skipping {shard_key} — already complete")
        else:
            pending_shards.append(shard_key)

    if not pending_shards:
        logging.info("All shards complete!")
        return

    total_pending_shards = len(pending_shards)
    total_pending_frames = sum(len(shards_dict[shard_key]) for shard_key in pending_shards)
    logging.info(
        f"Pending work: {total_pending_shards} shards | {total_pending_frames} frames | "
        f"batch_size={args.batch_size} | attn={args.attn_implementation}"
    )

    logging.info("Loading Model: microsoft/Florence-2-base-ft...")
    model_name = "microsoft/Florence-2-base-ft"
    
    import transformers
    import transformers.dynamic_module_utils
    # suppress dynamic module import checks that can fail in offline environments
    check_imports = lambda filename: []
    transformers.dynamic_module_utils.check_imports = check_imports
    
    if not hasattr(transformers.PretrainedConfig, "forced_bos_token_id"):
        transformers.PretrainedConfig.forced_bos_token_id = None
        
    if not hasattr(transformers.models.roberta.tokenization_roberta.RobertaTokenizer, 'additional_special_tokens'):
        transformers.models.roberta.tokenization_roberta.RobertaTokenizer.additional_special_tokens = property(lambda self: [str(t) for t in self.added_tokens_encoder.keys()])
        
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        attn_implementation=args.attn_implementation,
        torch_dtype=torch.float16, 
        trust_remote_code=True
    ).to("cuda")
    if args.compile_model:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            logging.info("torch.compile enabled with mode=reduce-overhead")
        except Exception as e:
            logging.warning(f"torch.compile unavailable, continuing without it: {e}")
    model.eval()

    total_frames = 0
    completed_shards = 0
    start_time = time.time()

    for shard_key in pending_shards:
        records = shards_dict[shard_key]
        dataset = FlorenceDataset(records)
        dataloader = DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            num_workers=args.num_workers, 
            pin_memory=True,
            collate_fn=custom_collate_fn,
            shuffle=False,
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
            persistent_workers=args.num_workers > 0
        )

        shard_results = []
        logging.info(f"Starting shard: {shard_key} ({len(records)} frames)")
        shard_start_time = time.time()
        shard_failed = False

        try:
            with torch.inference_mode(), torch.autocast("cuda", torch.float16):
                for batch_idx, (pil_images, frame_ids, row_indices) in enumerate(dataloader):
                    try:
                        # run caption and OCR as separate inference passes
                        captions = run_inference(
                            processor,
                            model,
                            "<MORE_DETAILED_CAPTION>",
                            pil_images,
                            args.caption_max_new_tokens,
                            args.num_beams,
                        )
                        ocrs = run_inference(
                            processor,
                            model,
                            "<OCR>",
                            pil_images,
                            args.ocr_max_new_tokens,
                            args.num_beams,
                        )
                        
                        for f_id, cap, ocr, r_idx in zip(frame_ids, captions, ocrs, row_indices):
                            shard_results.append({
                                'frame_id': f_id,
                                'caption': cap,
                                'ocr_text': ocr,
                                'manifest_row_index': r_idx
                            })
                    except Exception as e:
                        logging.error(f"Error in batch {batch_idx} of {shard_key}: {e}")
                        shard_failed = True
        except Exception as e:
            logging.error(f"Error processing shard {shard_key}: {e}")
            shard_failed = True
            
        if not shard_failed:
            jsonl_path = os.path.join(args.output_dir, f"{shard_key}.jsonl")
            with open(jsonl_path, 'w') as f:
                for res in shard_results:
                    f.write(json.dumps(res) + '\n')
            
            shard_time = time.time() - shard_start_time
            fps = len(records) / shard_time if shard_time > 0 else 0
            total_frames += len(records)
            completed_shards += 1
            cum_time = time.time() - start_time
            cum_fps = total_frames / cum_time if cum_time > 0 else 0
            remaining_frames = max(total_pending_frames - total_frames, 0)
            eta_seconds = remaining_frames / cum_fps if cum_fps > 0 else 0
            eta_td = str(timedelta(seconds=int(eta_seconds)))
            eta_at = datetime.utcnow() + timedelta(seconds=eta_seconds)

            logging.info(
                f"Completed {shard_key} | Frames: {len(records)} | "
                f"Progress: {completed_shards}/{total_pending_shards} shards | "
                f"Cum. Frames: {total_frames}/{total_pending_frames} | "
                f"Shard FPS: {fps:.2f} | Cum. FPS: {cum_fps:.2f} | "
                f"ETA: {eta_td} | ETA UTC: {eta_at.strftime('%Y-%m-%d %H:%M:%S')}"
            )
        else:
            logging.warning(f"Skipping save for {shard_key} due to errors.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default="index/indexing/upload_manifest.jsonl")
    parser.add_argument("--output_dir", type=str, default="index/florence_shards")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory used to detect already completed shard files")
    parser.add_argument("--batch_size", type=int, default=96, help="Inference batch size")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--caption_max_new_tokens", type=int, default=12)
    parser.add_argument("--ocr_max_new_tokens", type=int, default=8)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--test_limit", type=int, default=0, help="Process only the first N pending frames for testing")
    parser.add_argument("--reprocess_existing", action="store_true", help="Reprocess shards even if output files already exist")
    parser.add_argument("--compile_model", action="store_true", help="Enable torch.compile with reduce-overhead mode when available")
    parser.add_argument("--attn_implementation", type=str, default="sdpa", choices=["eager", "sdpa"], help="Attention backend for model loading")
    args = parser.parse_args()
    main(args)
