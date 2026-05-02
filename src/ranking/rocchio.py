"""
Rocchio Feedback Implementation for CASTLE 2024 Multimodal Search
Steps 43-45: Query encoding, anchor blending, and FAISS re-query
"""
import os
import sys
import json
import logging
import numpy as np
from typing import List, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


def load_siglip2_index() -> dict[str, int]:
    """Load frame_id -> manifest_row_index mapping for SigLIP2 embeddings."""
    index_path = os.path.join(PROJECT_ROOT, "embeddings/indexing/siglip2_index.jsonl")
    frame_to_row = {}
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            frame_id = rec["frame_id"]
            row_index = rec["manifest_row_index"]
            frame_to_row[frame_id] = row_index
    return frame_to_row


def get_siglip2_embeddings() -> np.ndarray:
    """Load SigLIP2 embeddings matrix."""
    embeddings_path = os.path.join(PROJECT_ROOT, "embeddings/indexing/siglip2_embeddings.npy")
    return np.load(embeddings_path)


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """L2 normalize a vector."""
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def execute_rocchio_feedback(
    raw_query: str,
    anchor_frame_id: str,
    siglip2_model,
    siglip2_processor,
    faiss_visual_index,
    frame_meta: dict,
    top_k: int = 50
) -> List[Tuple[str, float, bool]]:
    """
    Execute Rocchio relevance feedback: blend query + anchor, re-query FAISS.
    
    Args:
        raw_query: Original query text
        anchor_frame_id: Frame ID selected by user as anchor
        siglip2_model: Cached SigLIP2 model
        siglip2_processor: Cached SigLIP2 processor
        faiss_visual_index: Cached FAISS visual index
        frame_meta: Frame metadata dict
        top_k: Number of results to return
        
    Returns:
        List of (frame_id, score, multi_angle) tuples, top_k items
    """
    import torch
    from src.ranking.postprocess import temporal_dedup, flag_cross_stream
    
    logger.info(f"[Rocchio] Starting feedback for anchor: {anchor_frame_id}")
    
    logger.info("[Rocchio] Step 43: Encoding query via SigLIP2...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    inputs = siglip2_processor(text=[raw_query], padding="max_length", return_tensors="pt").to(device)
    with torch.no_grad():
        text_features = siglip2_model.get_text_features(**inputs)
        if hasattr(text_features, "pooler_output"):
            text_features = text_features.pooler_output
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        vector_q = text_features.cpu().numpy().astype("float32").flatten()
    
    vector_q = l2_normalize(vector_q)
    logger.info(f"[Rocchio] Query vector shape: {vector_q.shape}, norm: {np.linalg.norm(vector_q):.6f}")
    
    logger.info("[Rocchio] Step 44: Loading anchor embedding and blending...")
    
    siglip2_index = load_siglip2_index()
    embeddings = get_siglip2_embeddings()
    
    if anchor_frame_id not in siglip2_index:
        raise ValueError(f"Anchor frame_id {anchor_frame_id} not found in SigLIP2 index")
    
    anchor_row = siglip2_index[anchor_frame_id]
    vector_img = embeddings[anchor_row].astype("float32").flatten()
    
    vector_img = l2_normalize(vector_img)
    logger.info(f"[Rocchio] Anchor vector shape: {vector_img.shape}, norm: {np.linalg.norm(vector_img):.6f}")
    
    # simple equal-weight blend of text query and visual anchor
    vector_hybrid = (vector_q + vector_img) / 2.0
    
    vector_hybrid = l2_normalize(vector_hybrid)
    logger.info(f"[Rocchio] Hybrid vector norm: {np.linalg.norm(vector_hybrid):.6f}")
    
    logger.info("[Rocchio] Step 45: Searching FAISS with hybrid vector...")
    
    query_vec = vector_hybrid.reshape(1, -1)
    scores, indices = faiss_visual_index.search(query_vec, top_k)
    
    # build row_index -> frame_id lookup for the FAISS results
    row_map_path = os.path.join(PROJECT_ROOT, "index/indexing/faiss_row_map.jsonl")
    row_to_frame = {}
    with open(row_map_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            row_to_frame[rec["row_index"]] = rec["frame_id"]
    
    results_list = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        frame_id = row_to_frame.get(idx)
        if frame_id:
            results_list.append((frame_id, float(score)))
    
    logger.info(f"[Rocchio] FAISS returned {len(results_list)} results")
    
    deduped = temporal_dedup(results_list, frame_meta, window_sec=10.0)
    logger.info(f"[Rocchio] After dedup: {len(deduped)} results")
    
    flagged = flag_cross_stream(deduped, frame_meta)
    logger.info(f"[Rocchio] After cross-stream flag: {len(flagged)} results")
    
    return flagged[:top_k]
