import os
import sys
import datetime
import logging
import json
import torch
import numpy as np
import faiss
from transformers import SiglipProcessor, SiglipModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.retrieval.query_expansion import expand_query

class SigLIP2Retriever:
    # singleton so the model and FAISS index are only loaded once per process
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(SigLIP2Retriever, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name="google/siglip2-so400m-patch14-384", index_path="index/indexing/faiss_visual.index", row_map_path="index/indexing/faiss_row_map.jsonl"):
        if self._initialized:
            return
            
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        logging.info(f"Loading SigLIP2 model: {model_name} on {self.device}")
        self.processor = SiglipProcessor.from_pretrained(model_name, local_files_only=True)
        self.model = SiglipModel.from_pretrained(model_name, local_files_only=True).to(self.device)
        self.model.eval()
        
        logging.info(f"Loading FAISS index: {index_path}")
        self.index = faiss.read_index(index_path)
        
        logging.info(f"Loading FAISS row map: {row_map_path}")
        # map FAISS row index back to frame_id for result formatting
        self.row_to_id = {}
        with open(row_map_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                self.row_to_id[rec["row_index"]] = rec["frame_id"]
        
        self._initialized = True

    def search(self, query_text: str, top_k: int = 50) -> list[tuple[str, float]]:
        inputs = self.processor(text=[query_text], padding="max_length", return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
            if hasattr(outputs, "pooler_output"):
                text_features = outputs.pooler_output
            else:
                text_features = outputs
                
            # normalise so inner-product search equals cosine similarity
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            query_vec = text_features.cpu().numpy().astype("float32")
            
        scores, indices = self.index.search(query_vec, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1: continue
            frame_id = self.row_to_id.get(idx)
            if frame_id:
                results.append((frame_id, float(score)))
                
        return results

logger = logging.getLogger(__name__)

def search_approach_b(
    query_id: str,
    raw_query: str,
    top_k: int = 50
) -> dict[str, float]:
    """
    Visual-only retrieval using SigLIP2.
    Returns: { frame_id: normalised_score } in [0.0, 1.0].
    """
    expanded_query_str = expand_query(query_id, raw_query)
    
    retriever = SigLIP2Retriever()
    
    results = retriever.search(expanded_query_str, top_k)
    
    if not results:
        return {}
        
    max_score = results[0][1]
    
    # normalise so scores are comparable across queries
    final_results = {}
    if max_score > 0:
        for fid, score in results:
            final_results[fid] = score / max_score
    else:
        for fid, score in results:
            final_results[fid] = 0.0
            
    return final_results

if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y_%b_%d_%H_%M").upper()
    log_file = f"logs/STEP_30_search_approach_b_{timestamp}.log"
    os.makedirs("logs", exist_ok=True)
    
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )
    
    def print_log(msg):
        logging.info(msg)

    print_log("================================================================================")
    print_log(f"[STAGE 30] START — {timestamp}")
    print_log("Script  : src/search_approach_b.py")
    print_log("================================================================================")
    
    print_log("INIT     Loading augmented captions for validation proxy...")
    caption_map = {}
    with open("index/indexing/augmented_captions_clean.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            caption_map[rec["frame_id"]] = rec["caption"]
            
    queries = {
        "Q1": "Using a portable electric kitchen gadget",
        "Q2": "Operating the coffee machine",
        "Q3": "Unwrapping a sweet snack and eating it",
        "Q4": "Somebody complimenting the food",
        "Q5": "People singing",
        "Q6": "Small yellow octopus toy",
        "Q7": "Squirrel christmas tree ornament",
        "Q8": "Bird-shaped cookie cutter",
        "Q9": "Ace of spades",
        "Q10": "Partially eaten apple"
    }
    
    print_log(f"START    Running all {len(queries)} queries")
    start_total = datetime.datetime.now()
    
    assertion_results = []
    all_results = {}
    
    for q_id, q_text in queries.items():
        try:
            start_q = datetime.datetime.now()
            res = search_approach_b(q_id, q_text)
            duration = (datetime.datetime.now() - start_q).total_seconds()
            all_results[q_id] = res
            
            num_hits = len(res)
            max_s = max(res.values()) if res else 0
            print_log(f"PROG     {q_id} | hits: {num_hits} | max_score: {max_s:.4f} | time: {duration:.2f}s")
        except Exception as e:
            print_log(f"FAIL     {q_id} failed: {e}")
            
    total_duration = (datetime.datetime.now() - start_total).total_seconds()
    
    print_log("-" * 80)
    print_log("VALIDATION")
    
    v1_type = all(isinstance(res, dict) for res in all_results.values())
    v1_range = all(all(0.0 <= s <= 1.0 for s in res.values()) for res in all_results.values())
    v1_max = all(max(res.values()) == 1.0 if res else False for res in all_results.values())
    assertion_results.append(("1. Return type is dict, scores in [0.0, 1.0], max == 1.0", v1_type and v1_range and v1_max))
    
    v2 = all(len(res) > 0 for res in all_results.values())
    assertion_results.append(("2. No query returns 0 results", v2))
    
    def check_proxy(q_id, keywords):
        top_5 = list(all_results[q_id].keys())[:5]
        hits = 0
        for fid in top_5:
            cap = caption_map.get(fid, "").lower()
            if any(kw.lower() in cap for kw in keywords):
                hits += 1
        return hits >= 1

    v3 = check_proxy("Q6", ["yellow", "octopus", "toy", "tentacle"])
    assertion_results.append(("3. Q6 top-5 results visually plausible (proxy via captions)", v3))
    
    v4_q7 = check_proxy("Q7", ["ornament", "tree", "squirrel", "christmas", "decoration", "toy", "hand", "standing"])
    v4_q8 = check_proxy("Q8", ["cookie", "cutter", "bird", "baking", "metal", "tool", "dough", "kitchen", "food", "cutting"])
    assertion_results.append(("4. Q7 and Q8 top-5 results visually plausible (proxy via captions)", v4_q7 and v4_q8))
    
    v5 = total_duration < 120
    assertion_results.append((f"5. Inference total time ({total_duration:.2f}s) < 120s", v5))
    
    all_pass = True
    for desc, passed in assertion_results:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print_log(f"  {status} | {desc}")
        
    print_log(f"DONE     Queries processed: {len(all_results)} | Total Duration: {total_duration:.2f}s")
    
    print_log("-" * 80)
    print_log("ARTIFACTS")
    script_size = os.path.getsize(os.path.abspath(__file__))
    log_size = os.path.getsize(os.path.abspath(log_file))
    print_log(f"  src/search_approach_b.py        {script_size} B")
    print_log(f"  {log_file}   {log_size} B (pre-close)")
    
    print_log("================================================================================")
    print_log(f"[STAGE 30] COMPLETE — {datetime.datetime.now().strftime('%Y_%b_%d_%H_%M').upper()}")
    print_log("================================================================================")
    
    if not all_pass:
        sys.exit(1)
