# MOS — Multimodal Object Search over CASTLE 2024

A multimodal information retrieval system for the [CASTLE 2024](https://castle-dataset.github.io) egocentric lifelogging dataset. The system indexes 416,542 keyframes across four recording days and retrieves relevant frames for natural-language queries using three complementary approaches: sparse BM25 text retrieval, dense SigLIP2 visual retrieval, and a hybrid late-fusion pipeline with cross-encoder re-ranking.

---

## Results

| Approach | Description | Mean P@10 |
|---|---|---|
| A | BM25 sparse text retrieval | 0.070 |
| B | SigLIP2 dense visual retrieval | 0.620 |
| **C** | **Hybrid late fusion + cross-encoder re-ranking** | **0.640** |

Evaluated on 10 queries against 71 manually judged relevant frames.

---

## System Architecture

The pipeline is organised into seven layers:

```
CASTLE 2024 keyframes (416,542 frames)
        │
        ▼
Preprocessing
  ├── Resize to 768×768 WebP (107.61 GB → 41.28 GB)
  └── QAFF: CLIP ViT-B/32 scoring + pHash decimation
        → 244,966 frames selected for captioning
        │
        ▼
Feature Extraction
  ├── SigLIP2 SO400M visual embeddings  (416,542 × 1152, 1.79 GiB)
  ├── Florence-2 captions + OCR         (244,966 records)
  ├── OCR Hallucination Gate            (178,914 records stripped)
  └── BGE-large-en-v1.5 text embeddings
        ├── Transcript corpus  (145,140 × 1024, 0.55 GiB)
        └── Caption corpus     (244,966 × 1024, 0.93 GiB)
        │
        ▼
Indexes
  ├── FAISS IndexFlatIP — visual        (416,542 vectors, 1152-dim)
  ├── FAISS IndexFlatIP — transcripts   (145,140 vectors, 1024-dim)
  ├── FAISS IndexFlatIP — captions      (244,966 vectors, 1024-dim)
  └── Whoosh BM25                       (416,542 documents, 7 fields)
        │
        ▼
Retrieval
  ├── Approach A: BM25 (caption-only for visual queries,
  │              transcript+caption for speech queries)
  ├── Approach B: SigLIP2 FAISS search
  └── Approach C: Weighted late fusion of A + B + BGE dense text
                  → cross-encoder re-ranking (bge-reranker-v2-m3, top-50)
        │
        ▼
Post-Processing
  ├── Temporal deduplication (±10s window per stream)
  └── Cross-stream flagging (multi-angle detection)
        │
        ▼
Output: top-10 ranked frames with scores and timestamps
```

---

## Dataset

The [CASTLE 2024](https://castle-dataset.github.io) dataset is a 4-day egocentric lifelogging corpus:

| Metric | Value |
|---|---|
| Total keyframes | 416,542 |
| Recording days | 4 |
| Member streams (egocentric) | 10 |
| Fixed room cameras | 5 (kitchen, living1, living2, meeting, reading) |
| Total streams | 60 |
| Frame sampling interval | 5.0 seconds |
| Transcript JSON files | 667 |
| Transcript-aligned frame pairs | 145,140 |
| Original dataset size | 107.61 GB |
| Resized dataset size (768×768 WebP) | 41.28 GB |

Keyframe filenames follow the `HH_NNNN.webp` convention where `HH` is the hour of day and `NNNN` is the frame index within that hour. The time offset formula is:

```
time_offset_sec = (NNNN - 1) × 5
```

---

## Project Structure

```
MOS/
├── src/
│   ├── ingestion/          # Manifest building, transcript parsing and cleaning
│   ├── captioning/         # Florence-2 inference, OCR gate, caption filtering
│   ├── indexing/           # QAFF scoring, SigLIP2/BGE embeddings, FAISS/Whoosh
│   ├── retrieval/          # Approaches A, B, C, D; query expansion; weights
│   ├── ranking/            # Score aggregation, prefusion gate, postprocessing, Rocchio
│   ├── evaluation/         # Metrics, TSV export, candidate pooling, experiments
│   ├── ui/                 # Streamlit search interface
│   └── utils/              # Architecture diagram generator
├── report/
│   ├── main.tex            # ACM sigconf report
│   ├── report.bib          # Bibliography
│   ├── sections/           # Abstract, introduction, indexing, ranking, evaluation, conclusions
│   ├── tables/             # CSV data tables for LaTeX
│   └── figures/            # Architecture diagram, UI screenshot
├── evaluation/
│   ├── ground_truth.csv    # 71 manually judged relevant frames
│   ├── results_tsv/        # Q1–Q10 submission TSV files (Approach D)
│   └── precision_at_10.csv # Final P@10 results
├── requirements.txt
└── README.md
```

> **Not tracked in git:** `dataset/`, `embeddings/`, `index/`, `logs/`, `audit_log.md`, `venv/`

---

## Installation

```bash
git clone git@github.com:ArunGMR0411/ir-text2Image.git
cd ir-text2Image

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

NLTK data required for transcript cleaning:

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
```

---

## Pipeline Execution

Run the steps in order. Each script is self-contained and resume-safe.

### 1. Build the frame manifest

```bash
python src/ingestion/build_manifest.py
# Output: index/ingestion/manifest.jsonl, index/ingestion/manifest.csv
```

### 2. Parse and align transcripts

```bash
python src/ingestion/parse_transcripts.py
python src/ingestion/clean_transcripts.py
# Output: index/retrieval/transcript_aligned.csv
#         index/indexing/transcript_cleaned.csv
```

### 3. QAFF frame scoring and decimation

```bash
python src/indexing/qaff_score.py
python src/indexing/qaff_decimate.py
# Output: index/indexing/qaff_classified.jsonl
#         index/indexing/upload_manifest.jsonl  (244,966 frames)
```

### 4. Generate SigLIP2 visual embeddings

```bash
python src/indexing/generate_siglip2_embeddings.py
# Output: embeddings/indexing/siglip2_embeddings.npy  (1.79 GiB)
#         embeddings/indexing/siglip2_index.jsonl
```

### 5. Generate Florence-2 captions

```bash
python src/captioning/generate_florence2_captions.py \
    --manifest index/indexing/upload_manifest.jsonl \
    --output_dir index/florence_shards \
    --batch_size 96 \
    --attn_implementation sdpa
# Then merge shards and filter:
python src/captioning/filter_captions.py
```

### 6. Run the OCR hallucination gate

```bash
python src/captioning/ocr_hallucination_gate.py
# Output: index/indexing/augmented_captions_clean.jsonl
```

### 7. Generate BGE text embeddings

```bash
python src/indexing/generate_bge_embeddings.py
# Output: embeddings/indexing/bge_transcript_embeddings.npy  (0.55 GiB)
#         embeddings/indexing/bge_caption_embeddings.npy     (0.93 GiB)
```

### 8. Build FAISS and Whoosh indexes

```bash
python src/indexing/init_faiss_indexes.py
python src/indexing/populate_faiss_indexes.py
python src/indexing/init_whoosh_index.py
python src/indexing/ingest_whoosh.py
```

### 9. Run evaluation

```bash
python src/evaluation/run_evaluation.py
python src/evaluation/calculate_metrics.py
python src/evaluation/export_tsv.py
```

### 10. Launch the search UI

```bash
streamlit run src/ui/app.py
```

---

## Retrieval Approaches

### Approach A — BM25 Sparse Text Retrieval

- **Index:** Whoosh BM25 over transcript and caption text
- **Query type routing:** visual queries search caption field only; speech queries (Q4, Q5) use transcript (0.7) + caption (0.3)
- **Query expansion:** synonyms, object hints, colour hints per query
- **Location filter:** 50% score penalty for frames from wrong-room streams
- **Mean P@10:** 0.070

### Approach B — SigLIP2 Dense Visual Retrieval

- **Model:** `google/siglip2-so400m-patch14-384` (1152-dim)
- **Index:** FAISS IndexFlatIP over 416,542 L2-normalised frame embeddings
- **Query encoding:** expanded query text → SigLIP2 text encoder → FAISS search
- **Mean P@10:** 0.620

### Approach C — Hybrid Late Fusion with Cross-Encoder Re-ranking

Combines all four signals through weighted late fusion:

```
s_final = w_vis × V + w_text × (0.5 × BM25 + 0.5 × (0.5 × cap + 0.5 × trans))
```

Where `V` = SigLIP2 visual score, `BM25` = sparse score, `cap` = BGE caption score, `trans` = BGE transcript score.

Per-query weights (`w_vis` / `w_text`):

| Query | Description | w_vis | w_text |
|---|---|---|---|
| Q1 | Portable electric kitchen gadget | 0.7 | 0.3 |
| Q2 | Operating the coffee machine | 0.6 | 0.4 |
| Q3 | Unwrapping a sweet snack | 0.6 | 0.4 |
| Q4 | Complimenting the food | 0.7 | 0.3 |
| Q5 | People singing | 0.7 | 0.3 |
| Q6 | Small yellow octopus toy | 0.8 | 0.2 |
| Q7 | Squirrel christmas tree ornament | 0.8 | 0.2 |
| Q8 | Bird-shaped cookie cutter | 0.8 | 0.2 |
| Q9 | Ace of spades | 0.6 | 0.4 |
| Q10 | Partially eaten apple | 0.8 | 0.2 |

Top-50 candidates are then re-ranked with `BAAI/bge-reranker-v2-m3`:

```
hybrid_score = 0.7 × fusion_score + 0.3 × sigmoid(reranker_logit)
```

**Mean P@10:** 0.640

---

## OCR Hallucination Gate

Florence-2 frequently appended OCR-bearing strings to frames with no visible text. The gate encodes each OCR string via the SigLIP2 text encoder and computes cosine similarity against the frame's visual embedding. Strings below the 0.20 threshold are stripped.

| Metric | Value |
|---|---|
| Records with non-empty OCR text | 178,914 |
| OCR records stripped | 178,914 (100%) |
| Cosine similarity range | −0.081 to 0.137 (mean 0.043) |
| Gate threshold | 0.20 |

All 178,914 records fell below the threshold — consistent with SigLIP2's sigmoid calibration, where the theoretical breakeven cosine (~3.39) is impossible for unit-normalised vectors.

---

## Evaluation Queries

| ID | Query |
|---|---|
| Q1 | Somebody using a portable electric kitchen gadget |
| Q2 | Somebody operating the coffee machine |
| Q3 | Somebody unwrapping a sweet snack and eating it |
| Q4 | Somebody complimenting the food |
| Q5 | People singing |
| Q6 | Small yellow octopus toy |
| Q7 | Squirrel christmas tree ornament |
| Q8 | Bird-shaped cookie cutter |
| Q9 | Ace of spades |
| Q10 | Partially eaten apple |

Ground truth: 71 relevant frames across all 10 queries, collected via TREC-style depth-20 pooling.

---

## Interactive UI

The Streamlit interface provides:

- **Query selector** — all 10 assignment queries plus free-text input
- **Approach selector** — A (BM25), B (SigLIP2), C (Hybrid), D (Re-ranked)
- **Result cards** — frame thumbnail, score badge (🟢 ≥0.7 / 🟡 ≥0.4 / 🔴 <0.4), stream, day
- **Rocchio feedback** — click any result frame to blend its embedding with the query vector and re-search

```bash
streamlit run src/ui/app.py
```

---

## Models Used

| Model | Role | Dimension |
|---|---|---|
| `google/siglip2-so400m-patch14-384` | Visual embeddings + OCR gate | 1152 |
| `microsoft/Florence-2-base-ft` | Caption and OCR generation | — |
| `BAAI/bge-large-en-v1.5` | Dense text embeddings | 1024 |
| `BAAI/bge-reranker-v2-m3` | Cross-encoder re-ranking | — |
| `openai/clip-vit-b-32` | QAFF frame relevance scoring | 512 |
| `openai/clip-vit-l-14` | Initial keyframe embeddings | 768 |

---

## Dependencies

```
torch / torchvision      — GPU inference
transformers             — Florence-2, SigLIP2, BGE, reranker
sentence-transformers    — BGE embedding wrapper
faiss-cpu                — Exact vector search
whoosh                   — BM25 sparse index
streamlit                — Search UI
opencv-python            — Image decoding and resizing
pillow                   — PIL image handling
imagehash                — Perceptual hash for frame decimation
openai-clip              — QAFF scoring
nltk                     — Transcript tokenisation and stopword removal
numpy / pandas           — Data handling
tqdm                     — Progress bars
pytest                   — Test suite
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## References

- Rossetto et al. (2025). *The CASTLE 2024 Dataset: Advancing the Art of Multimodal Understanding*. ACM MM 2025. [castle-dataset.github.io](https://castle-dataset.github.io)
- Zhai et al. (2023). *SigLIP: Scaling Vision-Language Models with Sigmoid Loss*. arXiv:2303.15343
- Tschannen et al. (2025). *SigLIP 2*. arXiv:2502.14786
- Chen et al. (2024). *BGE: Retrieval-Enhanced Pre-Training for Dense Text Retrieval*. arXiv:2309.07597
- Robertson & Zaragoza (2009). *The Probabilistic Relevance Framework: BM25 and Beyond*
- Johnson et al. (2021). *Billion-Scale Similarity Search with GPUs (FAISS)*. IEEE TBMD
- Liu et al. (2024). *Grounding DINO*. arXiv:2303.05499
- Rocchio (1971). *Relevance Feedback in Information Retrieval*

---

## Author

**Arun Narayanan** — Dublin City University
