"""
Text2Image - Multi Modal Search - Streamlit Interactive Interface
Steps 40-45: Interactive System + Rocchio Feedback
"""
import os
import sys
import json
import csv
import logging
from datetime import datetime, timedelta

import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

from src.retrieval.query_expansion import RAW_QUERIES
from src.ranking.postprocess import load_frame_meta, temporal_dedup, flag_cross_stream
from src.ranking.rocchio import execute_rocchio_feedback



@st.cache_resource
def get_frame_meta():
    """Load frame metadata once at startup."""
    meta_path = os.path.join(PROJECT_ROOT, "index/indexing/faiss_row_map.jsonl")
    return load_frame_meta(meta_path)


@st.cache_resource
def _iter_manifest_records():
    """Yield manifest records from the current module-aligned path, with legacy fallback."""
    candidate_paths = [
        os.path.join(PROJECT_ROOT, "index", "ingestion", "manifest.csv"),
        os.path.join(PROJECT_ROOT, "index", "ingestion", "manifest.jsonl"),
        os.path.join(PROJECT_ROOT, "index", "manifest.csv"),
        os.path.join(PROJECT_ROOT, "index", "manifest.jsonl"),
    ]

    manifest_path = next((path for path in candidate_paths if os.path.exists(path)), None)
    if manifest_path is None:
        raise FileNotFoundError(
            "No manifest file found. Tried: " + ", ".join(candidate_paths)
        )

    if manifest_path.endswith(".csv"):
        with open(manifest_path, "r", encoding="utf-8", newline="") as f:
            yield from csv.DictReader(f)
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


@st.cache_resource
def get_manifest_lookup():
    """Build frame_id -> full_path lookup from manifest."""
    lookup = {}
    for rec in _iter_manifest_records():
        stream = rec["stream_name"]
        day = rec["day"]
        filename = rec["filename"]
        frame_id = f"{stream}__{day}__{filename}"
        lookup[frame_id] = os.path.join(PROJECT_ROOT, rec["full_path"])
    return lookup


@st.cache_resource
def load_search_modules():
    """Lazy load search modules (some load heavy models at import)."""
    from src.retrieval.search_approach_a import search_approach_a
    from src.retrieval.search_approach_b import search_approach_b
    from src.retrieval.search_approach_c import search_approach_c
    return {
        "A": search_approach_a,
        "B": search_approach_b,
        "C": search_approach_c
    }


@st.cache_resource
def load_siglip2_resources():
    """Load SigLIP2 model, processor, and FAISS index for Rocchio feedback."""
    import torch
    import faiss
    import gc
    from transformers import SiglipProcessor, SiglipModel
    
    # clear GPU cache before loading a large model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info("[Resources] Cleared GPU cache before SigLIP2 load")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"[Resources] Loading SigLIP2 model on {device}")
    
    try:
        model_name = "google/siglip2-so400m-patch14-384"
        processor = SiglipProcessor.from_pretrained(model_name)
        model = SiglipModel.from_pretrained(model_name).to(device)
        model.eval()
        
        logger.info("[Resources] Loading FAISS visual index...")
        faiss_index_path = os.path.join(PROJECT_ROOT, "index/indexing/faiss_visual.index")
        faiss_index = faiss.read_index(faiss_index_path)
        
        logger.info("[Resources] SigLIP2 resources loaded successfully")
        return model, processor, faiss_index
    except Exception as e:
        logger.error(f"[Resources] Failed to load SigLIP2 resources: {e}")
        raise



def format_time_with_overflow(hour: int, time_offset_sec: float) -> str:
    """
    Convert hour + time_offset_sec to HH:MM:SS format.
    Handles overflow: if offset pushes into next hour, carry over.
    """
    total_seconds = int(time_offset_sec)
    additional_hours = total_seconds // 3600
    remaining_seconds = total_seconds % 3600
    
    final_hour = hour + additional_hours
    minutes = remaining_seconds // 60
    secs = remaining_seconds % 60
    
    return f"{final_hour:02d}:{minutes:02d}:{secs:02d}"


def get_time_window_with_overflow(hour: int, time_offset_sec: float, window: float = 5.0) -> tuple[str, str]:
    """
    Get start/end time with ±window seconds, handling hour overflow.
    start = max(0, time_offset_sec - window)
    end = time_offset_sec + window
    """
    # clamp start to zero so we don't produce negative timestamps
    start_offset = max(0.0, time_offset_sec - window)
    end_offset = time_offset_sec + window
    
    start_str = format_time_with_overflow(hour, start_offset)
    end_str = format_time_with_overflow(hour, end_offset)
    
    return start_str, end_str


def run_search(approach: str, query_id: str, raw_query: str) -> list[tuple]:
    """
    Execute search with selected approach, apply dedup and cross-stream flag.
    Returns list of (frame_id, score, multi_angle) tuples.
    """
    search_modules = load_search_modules()
    frame_meta = get_frame_meta()
    
    if approach == "A":
        results_dict = search_modules["A"](query_id, raw_query, top_k=50)
        results_list = sorted(results_dict.items(), key=lambda x: (-x[1], x[0]))
    elif approach == "B":
        results_dict = search_modules["B"](query_id, raw_query, top_k=50)
        results_list = sorted(results_dict.items(), key=lambda x: (-x[1], x[0]))
    else:
        results_list = search_modules["C"](query_id, raw_query, top_k=50)
    
    if not results_list:
        return []
    
    # remove near-duplicate frames from the same stream within 10 seconds
    deduped = temporal_dedup(results_list, frame_meta, window_sec=10.0)
    
    flagged = flag_cross_stream(deduped, frame_meta)
    
    return flagged[:10]


def run_rocchio_search(raw_query: str, anchor_frame_id: str) -> list[tuple]:
    """
    Execute Rocchio feedback search with anchor frame.
    Returns list of (frame_id, score, multi_angle) tuples.
    """
    siglip2_model, siglip2_processor, faiss_index = load_siglip2_resources()
    frame_meta = get_frame_meta()
    
    results = execute_rocchio_feedback(
        raw_query=raw_query,
        anchor_frame_id=anchor_frame_id,
        siglip2_model=siglip2_model,
        siglip2_processor=siglip2_processor,
        faiss_visual_index=faiss_index,
        frame_meta=frame_meta,
        top_k=50
    )
    
    return results[:10]



def init_session_state():
    """Initialize Streamlit session state variables."""
    if "refinement_anchor" not in st.session_state:
        st.session_state.refinement_anchor = None
    if "last_results" not in st.session_state:
        st.session_state.last_results = None
    if "last_query_id" not in st.session_state:
        st.session_state.last_query_id = None
    if "last_raw_query" not in st.session_state:
        st.session_state.last_raw_query = None
    if "last_approach" not in st.session_state:
        st.session_state.last_approach = None
    if "last_is_refinement" not in st.session_state:
        st.session_state.last_is_refinement = False
    if "last_refinement_anchor" not in st.session_state:
        st.session_state.last_refinement_anchor = None


def main():
    st.set_page_config(
        page_title="Text2Image - Multi Modal Search",
        page_icon="🏰",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
        <style>
          footer {visibility: hidden;}
          header {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)
    
    st.title("Text2Image - Multi Modal Search")
    st.caption("Egocentric · Exocentric · 4-Stream Late Fusion")
    st.markdown("---")
    
    init_session_state()
    
    with st.sidebar:
        st.header("🔍 Search Controls")
        
        # sort query IDs numerically so Q10 comes after Q9
        sorted_query_ids = sorted(
            RAW_QUERIES.keys(),
            key=lambda qid: int(qid[1:]) if qid.startswith("Q") and qid[1:].isdigit() else qid,
        )
        query_options = [f"{qid}: {RAW_QUERIES[qid]}" for qid in sorted_query_ids]
        selected_query_str = st.selectbox(
            "Quick Select (10 Assignment Queries)",
            options=query_options,
            index=0
        )
        query_id_preset = selected_query_str.split(":")[0]
        raw_query_preset = RAW_QUERIES[query_id_preset]
        
        st.markdown("---")
        custom_query = st.text_input(
            "Or enter a custom query",
            placeholder="Type your own search query here...",
            help="If provided, this overrides the Quick Select above"
        )
        
        # custom query takes precedence over the preset selector
        if custom_query.strip():
            query_id = "CUSTOM"
            raw_query = custom_query.strip()
        else:
            query_id = query_id_preset
            raw_query = raw_query_preset
        
        st.markdown("---")
        
        approach = st.radio(
            "Search Approach",
            options=["A", "B", "C"],
            format_func=lambda x: {
                "A": "Approach A (BM25)",
                "B": "Approach B (Visual SigLIP2)",
                "C": "Approach C (Fusion)"
            }[x],
            index=0
        )
        
        st.markdown("---")
        
        search_clicked = st.button("🔍 Search", type="primary", use_container_width=True)
        
        if st.session_state.refinement_anchor:
            st.markdown("---")
            if st.button("✖ Clear Refinement", use_container_width=True):
                st.session_state.refinement_anchor = None
                st.session_state.last_is_refinement = False
                st.session_state.last_refinement_anchor = None
                st.rerun()
    
    if search_clicked:
        with st.spinner("Searching..."):
            try:
                results = run_search(approach, query_id, raw_query)
                
                st.session_state.last_results = results
                st.session_state.last_query_id = query_id
                st.session_state.last_raw_query = raw_query
                st.session_state.last_approach = approach
                st.session_state.refinement_anchor = None
                st.session_state.last_is_refinement = False
                st.session_state.last_refinement_anchor = None
                
                if not results:
                    st.warning("No results found for this query.")
                else:
                    display_results(results, query_id, approach, raw_query)
                    
            except Exception as e:
                st.error(f"Search failed: {str(e)}")
                logger.error(f"Search error: {e}", exc_info=True)
    
    elif st.session_state.refinement_anchor and st.session_state.last_raw_query:
        anchor_id = st.session_state.refinement_anchor
        raw_query = st.session_state.last_raw_query
        
        display_refinement_banner(anchor_id)
        
        with st.spinner("Refining search with Rocchio feedback..."):
            try:
                results = run_rocchio_search(raw_query, anchor_id)
                st.session_state.last_results = results
                st.session_state.last_query_id = "CUSTOM" if st.session_state.last_query_id == "CUSTOM" else st.session_state.last_query_id
                st.session_state.last_approach = "Refined"
                st.session_state.last_is_refinement = True
                st.session_state.last_refinement_anchor = anchor_id
                
                if not results:
                    st.warning("No refined results found.")
                else:
                    display_results(results, "Rocchio", "Refined", raw_query, is_refinement=True)
                    
            except Exception as e:
                st.error(f"Refinement failed: {str(e)}")
                logger.error(f"Rocchio error: {e}", exc_info=True)
    
    elif st.session_state.last_results:
        # re-display the last results so they persist across reruns
        if st.session_state.last_is_refinement and st.session_state.last_refinement_anchor:
            display_refinement_banner(st.session_state.last_refinement_anchor)
            display_results(
                st.session_state.last_results,
                "Rocchio",
                "Refined",
                st.session_state.last_raw_query or "",
                is_refinement=True,
            )
        else:
            display_results(
                st.session_state.last_results,
                st.session_state.last_query_id or query_id,
                st.session_state.last_approach or approach,
                st.session_state.last_raw_query or raw_query,
            )

    else:
        st.info("👈 Configure your search in the sidebar and click 'Search' to begin.")
        
        st.subheader("Selected Query Details")
        st.write(f"**Query ID:** {query_id}")
        st.write(f"**Query Text:** {raw_query}")


def display_refinement_banner(anchor_frame_id: str):
    """Display refinement banner with anchor image (Step 42)."""
    manifest_lookup = get_manifest_lookup()
    
    img_path = manifest_lookup.get(anchor_frame_id)
    
    st.info(f"🔁 Refining search using frame **{anchor_frame_id}** as anchor")
    
    if img_path and os.path.exists(img_path):
        st.image(img_path, width=200)


def display_results(results: list[tuple], query_id: str, approach: str, raw_query: str = "", is_refinement: bool = False):
    """
    Display search results in a 2x5 grid layout (Step 41 enhancement).
    
    Args:
        results: List of (frame_id, score, multi_angle) tuples
        query_id: Query identifier
        approach: Search approach (A, B, C, or "Rocchio")
        raw_query: Original query text (for Rocchio refinement)
        is_refinement: True if showing Rocchio-refined results
    """
    manifest_lookup = get_manifest_lookup()
    frame_meta = get_frame_meta()
    
    if is_refinement:
        approach_label = "Rocchio Refined"
        st.subheader(f"🔁 Top {len(results)} Results — {approach_label} | Query: CUSTOM")
    else:
        approach_labels = {"A": "BM25 Text", "B": "Visual SigLIP2", "C": "Fusion"}
        approach_label = approach_labels.get(approach, approach)
        st.subheader(f"Top {len(results)} Results — {approach_label} | Query: {query_id}")
    
    st.markdown(f"**Top {len(results)} results after temporal deduplication:**")
    
    # display results in a 2-row × 5-column grid
    for row_idx in range(2):
        cols = st.columns(5)
        
        for col_idx in range(5):
            result_idx = row_idx * 5 + col_idx
            
            if result_idx >= len(results):
                break
            
            frame_id, score, multi_angle = results[result_idx]
            
            with cols[col_idx]:
                img_path = manifest_lookup.get(frame_id)
                
                meta = frame_meta.get(frame_id, {})
                stream_name = meta.get("stream_name", "unknown")
                time_offset = meta.get("time_offset_sec", 0)
                hour = meta.get("hour", 0)
                
                start_time, end_time = get_time_window_with_overflow(hour, time_offset)
                
                if img_path and os.path.exists(img_path):
                    st.image(img_path, use_container_width=True)
                else:
                    st.image(
                        "https://via.placeholder.com/300x200?text=Image+Unavailable",
                        caption="Image unavailable",
                        use_container_width=True
                    )
                
                st.markdown(f"**{stream_name}**")
                
                frame_parts = frame_id.split("__")
                if len(frame_parts) >= 3:
                    day = frame_parts[1]
                    frame_num = frame_parts[2]
                    st.markdown(f"📅 Day: {day} | Frame: {frame_num}")
                
                st.markdown(f"⏱️ {start_time} – {end_time}")
                
                # colour-code the score badge so relevance is visible at a glance
                if score >= 0.7:
                    score_emoji = "🟢"
                elif score >= 0.4:
                    score_emoji = "🟡"
                else:
                    score_emoji = "🔴"
                st.markdown(f"{score_emoji} **{score:.4f}**")
                
                if multi_angle:
                    st.markdown("📡 **Multi-angle**")
                
                if st.button("🔍 Refine Search", key=f"refine_{frame_id}", use_container_width=True):
                    st.session_state.refinement_anchor = frame_id
                    st.rerun()
                
                st.markdown("---")



if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y_%b_%d_%H_%M").upper()
    logger.info("=" * 80)
    logger.info(f"[STEP 40] Text2Image - Multi Modal Search App Startup — {timestamp}")
    logger.info("Script  : src/app.py")
    logger.info("=" * 80)
    
    logger.info("INIT     Running pre-flight checks...")
    
    checks = []
    
    try:
        meta = get_frame_meta()
        checks.append(("Frame metadata load", len(meta) > 0))
        logger.info(f"  PASS   Frame metadata: {len(meta)} entries")
    except Exception as e:
        checks.append(("Frame metadata load", False))
        logger.error(f"  FAIL   Frame metadata: {e}")
    
    try:
        lookup = get_manifest_lookup()
        checks.append(("Manifest lookup load", len(lookup) > 0))
        logger.info(f"  PASS   Manifest lookup: {len(lookup)} entries")
    except Exception as e:
        checks.append(("Manifest lookup load", False))
        logger.error(f"  FAIL   Manifest lookup: {e}")
    
    try:
        modules = load_search_modules()
        checks.append(("Search modules import", all(m is not None for m in modules.values())))
        logger.info("  PASS   Search modules imported")
    except Exception as e:
        checks.append(("Search modules import", False))
        logger.error(f"  FAIL   Search modules: {e}")
    
    try:
        checks.append(("Queries defined", len(RAW_QUERIES) == 10))
        logger.info(f"  PASS   Queries: {len(RAW_QUERIES)} defined")
    except Exception as e:
        checks.append(("Queries defined", False))
        logger.error(f"  FAIL   Queries: {e}")
    
    all_pass = all(passed for _, passed in checks)
    
    if all_pass:
        logger.info("INIT     All pre-flight checks PASSED — starting Streamlit")
        logger.info("=" * 80)
    else:
        logger.error("INIT     Some pre-flight checks FAILED — app may not function correctly")
        logger.info("=" * 80)
    
    main()
