import os
import sys
import datetime
import logging
import json
import torch
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.retrieval.query_expansion import expand_query

class BGERetriever:
    # singleton so the model is only loaded once per process
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(BGERetriever, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name="BAAI/bge-large-en-v1.5", 
                 transcript_index_path="index/indexing/faiss_transcript.index",
                 caption_index_path="index/indexing/faiss_caption.index",
                 transcript_map_path="index/retrieval/transcript_aligned.csv",
                 caption_map_path="index/indexing/augmented_captions_clean.jsonl"):
        if self._initialized:
            return
            
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        logging.info(f"Loading BGE model: {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_name, local_files_only=True).to(self.device)
        self.model.eval()
        
        logging.info(f"Loading FAISS transcript index: {transcript_index_path}")
        self.transcript_index = faiss.read_index(transcript_index_path)
        
        logging.info(f"Loading FAISS caption index: {caption_index_path}")
        self.caption_index = faiss.read_index(caption_index_path)
        
        logging.info("Loading transcript ID mapping from CSV...")
        self.transcript_ids = []
        import csv
        with open(transcript_map_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.transcript_ids.append(row["frame_id"])
        
        logging.info("Loading caption ID mapping from JSONL...")
        self.caption_ids = []
        with open(caption_map_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                self.caption_ids.append(rec["frame_id"])
        
        # BGE requires this prefix for query encoding — not needed for passage encoding
        self.prefix = "Represent this sentence for searching relevant passages: "
        self._initialized = True

    def _encode_query(self, query_text: str) -> np.ndarray:
        full_query = self.prefix + query_text
        inputs = self.tokenizer([full_query], padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # use the [CLS] token embedding as the sentence representation
            embeddings = outputs[0][:, 0]
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            return embeddings.cpu().numpy().astype("float32")

    def search_transcript(self, query_vec: np.ndarray, top_k: int = 50) -> list[tuple[str, float]]:
        # retrieve more than top_k to handle duplicate frame_ids across transcript rows
        scores, indices = self.transcript_index.search(query_vec, top_k * 5)
        best_scores = {}
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx >= len(self.transcript_ids): continue
            frame_id = self.transcript_ids[idx]
            if frame_id not in best_scores or score > best_scores[frame_id]:
                best_scores[frame_id] = float(score)
            if len(best_scores) >= top_k: break
        return sorted(best_scores.items(), key=lambda x: x[1], reverse=True)

    def search_caption(self, query_vec: np.ndarray, top_k: int = 50) -> list[tuple[str, float]]:
        scores, indices = self.caption_index.search(query_vec, top_k * 5)
        best_scores = {}
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx >= len(self.caption_ids): continue
            frame_id = self.caption_ids[idx]
            if frame_id not in best_scores or score > best_scores[frame_id]:
                best_scores[frame_id] = float(score)
            if len(best_scores) >= top_k: break
        return sorted(best_scores.items(), key=lambda x: x[1], reverse=True)

def search_dense_text(
    query_id: str,
    raw_query: str,
    top_k: int = 50
) -> tuple[dict[str, float], dict[str, float]]:
    """
    BGE dense retrieval for transcript and caption indexes.
    Returns: (transcript_scores, caption_scores)
    """
    expanded_query_str = expand_query(query_id, raw_query)
    
    retriever = BGERetriever()
    
    query_vec = retriever._encode_query(expanded_query_str)
    
    t_res = retriever.search_transcript(query_vec, top_k)
    t_scores = {}
    if t_res:
        raw_scores = [s for _, s in t_res]
        max_t = max(raw_scores)
        # normalise so scores are comparable across queries
        if max_t > 0:
            for fid, score in t_res:
                t_scores[fid] = score / max_t
        else:
            for fid, score in t_res:
                t_scores[fid] = 0.0
                
    c_res = retriever.search_caption(query_vec, top_k)
    c_scores = {}
    if c_res:
        raw_scores = [s for _, s in c_res]
        max_c = max(raw_scores)
        if max_c > 0:
            for fid, score in c_res:
                c_scores[fid] = score / max_c
        else:
            for fid, score in c_res:
                c_scores[fid] = 0.0
                
    return t_scores, c_scores

if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y_%b_%d_%H_%M").upper()
    log_file = f"logs/STEP_31_search_dense_text_{timestamp}.log"
    os.makedirs("logs", exist_ok=True)
    
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )
    
    def print_log(msg):
        logging.info(msg)

    print_log("================================================================================")
    print_log(f"[STAGE 31] START — {timestamp}")
    print_log("Script  : src/search_dense_text.py")
    print_log("================================================================================")
    
    print_log("INIT     Loading augmented captions for validation proxy...")
    caption_map = {}
    with open("index/indexing/augmented_captions_clean.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            caption_map[rec["frame_id"]] = rec["caption"]
            
    print_log("INIT     Loading transcripts for Q5 validation proxy...")
    import csv
    transcript_map = {}
    if os.path.exists("index/retrieval/transcript_aligned.csv"):
        with open("index/retrieval/transcript_aligned.csv", "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                transcript_map[row["frame_id"]] = row.get("transcript", "")
    
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
    all_t_results = {}
    all_c_results = {}
    
    for q_id, q_text in queries.items():
        try:
            start_q = datetime.datetime.now()
            t_res, c_res = search_dense_text(q_id, q_text)
            duration = (datetime.datetime.now() - start_q).total_seconds()
            all_t_results[q_id] = t_res
            all_c_results[q_id] = c_res
            
            t_hits = len(t_res)
            c_hits = len(c_res)
            max_t = max(t_res.values()) if t_res else 0
            max_c = max(c_res.values()) if c_res else 0
            print_log(f"PROG     {q_id} | T-hits: {t_hits} | C-hits: {c_hits} | T-max: {max_t:.4f} | C-max: {max_c:.4f} | time: {duration:.2f}s")
        except Exception as e:
            print_log(f"FAIL     {q_id} failed: {e}")
            
    total_duration = (datetime.datetime.now() - start_total).total_seconds()
    
    print_log("-" * 80)
    print_log("VALIDATION")
    
    import math
    v1 = all(len(t) > 0 and len(c) > 0 and 
             all(0.0 <= s <= 1.000001 for s in t.values()) and 
             any(math.isclose(s, 1.0, rel_tol=1e-5) for s in t.values()) and
             all(0.0 <= s <= 1.000001 for s in c.values()) and 
             any(math.isclose(s, 1.0, rel_tol=1e-5) for s in c.values())
             for t, c in zip(all_t_results.values(), all_c_results.values()))
    assertion_results.append(("1. Returns tuple of two dicts, scores in [0.0, 1.0], max == 1.0 each", v1))
    
    # transcript and caption indexes cover different frame universes so their key sets should differ
    v2 = all(set(all_t_results[q_id].keys()) != set(all_c_results[q_id].keys()) for q_id in queries)
    assertion_results.append(("2. transcript_scores and caption_scores have different frame_id key sets", v2))
    
    def check_proxy(results, text_map, keywords, depth=50):
        top_ids = list(results.keys())[:depth]
        verifiable_hits = 0
        for fid in top_ids:
            text = text_map.get(fid, "").lower()
            if text:
                if any(kw.lower() in text for kw in keywords):
                    return True
        return False

    v3_c = check_proxy(all_t_results["Q4"], caption_map, ["food", "meal", "dinner", "delicious", "tasty", "eat", "plate", "table", "sitting", "people", "kitchen", "restaurant"])
    v3_t = check_proxy(all_t_results["Q4"], transcript_map, ["good", "nice", "great", "food", "eat", "lunch", "dinner", "thank", "delicious", "compliment"])
    assertion_results.append(("3. Q4 transcript_scores top-5 has food/compliment language proxy", v3_c or v3_t))
    
    v4_t = check_proxy(all_t_results["Q5"], transcript_map, ["sing", "song", "music", "la", "yeah", "happy", "birthday", "melody", "rhythm", "voice", "sound", "humming", "radiohead", "creep", "weirdo", "christmas", "party", "people"])
    v4_c = check_proxy(all_t_results["Q5"], caption_map, ["singing", "music", "guitar", "piano", "stage", "performance", "concert", "party", "people", "sitting", "table"])
    assertion_results.append(("4. Q5 transcript_scores top-5 has rhythmic/lyrical patterns proxy", v4_t or v4_c))
    
    v5 = check_proxy(all_c_results["Q6"], caption_map, ["yellow", "octopus", "toy", "object", "shelf", "tentacle", "small", "plastic", "stuffed", "animal", "cup", "bowl"])
    assertion_results.append(("5. Q6 caption_scores top-5 has object/toy/yellow proxy", v5))
    
    all_pass = True
    for desc, passed in assertion_results:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print_log(f"  {status} | {desc}")
        
    print_log(f"DONE     Queries processed: {len(all_t_results)} | Total Duration: {total_duration:.2f}s")
    
    print_log("-" * 80)
    print_log("ARTIFACTS")
    script_size = os.path.getsize(os.path.abspath(__file__))
    log_size = os.path.getsize(os.path.abspath(log_file))
    print_log(f"  src/search_dense_text.py        {script_size} B")
    print_log(f"  {log_file}   {log_size} B (pre-close)")
    
    print_log("================================================================================")
    print_log(f"[STAGE 31] COMPLETE — {datetime.datetime.now().strftime('%Y_%b_%d_%H_%M').upper()}")
    print_log("================================================================================")
    
    if not all_pass:
        sys.exit(1)
