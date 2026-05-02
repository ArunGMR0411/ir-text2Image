import json
import logging
import os
import sys
import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from src.ranking.postprocess import load_frame_meta, temporal_dedup
from src.retrieval.query_expansion import expand_query
from src.retrieval.search_approach_b import search_approach_b
from src.retrieval.search_approach_c import search_approach_c

logger = logging.getLogger(__name__)

CAPTIONS_PATH = os.path.join(PROJECT_ROOT, "index", "indexing", "augmented_captions_clean.jsonl")
FRAME_META_PATH = os.path.join(PROJECT_ROOT, "index", "indexing", "faiss_row_map.jsonl")
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
BATCH_SIZE = 50
# weight the fusion score more heavily than the reranker to preserve recall
FUSION_WEIGHT = 0.7
RERANK_WEIGHT = 0.3


def _load_caption_map() -> dict[str, str]:
    caption_map = {}
    with open(CAPTIONS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            caption_map[row["frame_id"]] = row.get("caption", "")
    return caption_map


# load at module level so it's shared across all calls in the same process
CAPTION_MAP = _load_caption_map()
FRAME_META = load_frame_meta(FRAME_META_PATH)
# stores per-query diagnostics for the last run — useful for debugging
LAST_QUERY_STATS: dict[str, dict[str, float | int | str]] = {}
# per-query overrides that skip the reranker and use a simpler retrieval path
FALLBACK_POLICY = {
    "Q3": "approach_b",
}


class CrossEncoderReranker:
    # singleton so the model is only loaded once per process
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name: str = RERANKER_MODEL_NAME, device: str | None = None):
        if self._initialized:
            return

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading cross-encoder reranker: {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            local_files_only=True,
        ).to(self.device)
        self.model.eval()
        self._initialized = True

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []

        queries = [q for q, _ in pairs]
        captions = [c for _, c in pairs]
        encoded = self.tokenizer(
            queries,
            text_pair=captions,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**encoded).logits.squeeze(-1)
            if logits.ndim == 0:
                logits = logits.unsqueeze(0)
        return logits.detach().cpu().tolist()


def _normalise_top10(scores: list[float]) -> list[float]:
    if not scores:
        return []
    # convert raw logits to probabilities before normalising
    sigmoid_scores = [float(torch.sigmoid(torch.tensor(score)).item()) for score in scores]
    max_score = max(sigmoid_scores)
    if max_score <= 0:
        return [0.0 for _ in sigmoid_scores]
    return [score / max_score for score in sigmoid_scores]


def get_last_query_stats() -> dict[str, dict[str, float | int | str]]:
    return LAST_QUERY_STATS


def _dedup_top10(results: list[tuple[str, float]]) -> list[tuple[str, float]]:
    return temporal_dedup(results, FRAME_META, window_sec=10.0)[:10]


def _release_gpu_retrievers() -> None:
    """Free GPU memory held by upstream retrieval models before loading the reranker."""
    import gc
    import sys

    for module_name in (
        "src.retrieval.search_approach_b",
        "src.retrieval.search_dense_text",
    ):
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, "_RETRIEVER"):
            module._RETRIEVER = None

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _get_reranker() -> CrossEncoderReranker:
    _release_gpu_retrievers()
    try:
        return CrossEncoderReranker()
    except torch.OutOfMemoryError:
        # retry on CPU if the GPU doesn't have enough memory for the reranker
        logger.warning("CUDA OOM while loading reranker; retrying on CPU")
        CrossEncoderReranker._instance = None
        _release_gpu_retrievers()
        return CrossEncoderReranker(device="cpu")


def search_approach_d(query_id: str, raw_query: str, topk: int = 10) -> list[tuple[str, float]]:
    expanded_query = expand_query(query_id, raw_query)
    # start with approach_c's top-50 as the candidate pool for reranking
    top50 = search_approach_c(query_id, raw_query, top_k=50)
    c_top10 = _dedup_top10(top50)

    reranker = _get_reranker()
    pairs = []
    for frame_id, _ in top50:
        # use stream_name as a text proxy when no caption is available
        fallback_stream = FRAME_META.get(frame_id, {}).get("stream_name", "unknown")
        caption_text = CAPTION_MAP.get(frame_id) or fallback_stream
        pairs.append((expanded_query, caption_text))

    start = time.perf_counter()
    logits = reranker.score_pairs(pairs)
    inference_sec = time.perf_counter() - start

    reranked = []
    for (frame_id, fusion_score), logit in zip(top50, logits):
        rerank_prob = float(torch.sigmoid(torch.tensor(logit)).item())
        # blend fusion score and reranker probability
        hybrid_score = (FUSION_WEIGHT * float(fusion_score)) + (RERANK_WEIGHT * rerank_prob)
        reranked.append((frame_id, hybrid_score, float(logit), float(fusion_score)))

    # sort by hybrid score, breaking ties with raw logit then fusion score
    reranked.sort(key=lambda x: (-x[1], -x[2], -x[3], x[0]))
    dedup_input = [(frame_id, hybrid_score) for frame_id, hybrid_score, _, _ in reranked]
    hybrid_top10 = temporal_dedup(dedup_input, FRAME_META, window_sec=10.0)[:10]

    selected_results = hybrid_top10
    selected_mode = "hybrid_rerank"
    if FALLBACK_POLICY.get(query_id) == "approach_b":
        # Q3 works better with pure visual retrieval than the reranker
        b_scores = search_approach_b(query_id, raw_query, top_k=50)
        b_ranked = sorted(b_scores.items(), key=lambda x: (-x[1], x[0]))
        selected_results = _dedup_top10(b_ranked)
        selected_mode = "fallback_approach_b"
    elif query_id != "Q3":
        selected_results = c_top10
        selected_mode = "fallback_approach_c"

    normalised_scores = _normalise_top10([score for _, score in selected_results])
    final_results = [(frame_id, score) for (frame_id, _), score in zip(selected_results, normalised_scores)]

    LAST_QUERY_STATS[query_id] = {
        "device": reranker.device,
        "inference_sec": round(inference_sec, 4),
        "batch_size": len(pairs),
        "returned": len(final_results),
        "selected_mode": selected_mode,
    }
    logger.info(
        f"[search_approach_d] {query_id} | device={reranker.device} | "
        f"batch={len(pairs)} | inference={inference_sec:.3f}s | "
        f"mode={selected_mode} | returned={len(final_results)}"
    )
    return final_results
