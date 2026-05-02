import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import matplotlib.path as mpath
import os

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Inter', 'Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(12, 17.5), dpi=100)
ax.set_xlim(0, 1200)
ax.set_ylim(1750, 0)
ax.axis('off')

fig.patch.set_facecolor('#F8F9FA')
ax.set_facecolor('#F8F9FA')

# each layer describes one stage of the pipeline
LAYERS = [
    {
        'header': 'LAYER 1 — INPUT DATA',
        'color': '#3C5B74',
        'boxes': [
            ['CASTLE Dataset', 'Visual keyframes collection', '416,542 WebP frames'],
            ['ASR Transcripts', 'Speech recognition output', '667 JSON transcript files'],
            ['Multi-Stream Capture', 'Egocentric + exocentric', '10 members + 5 fixed cams']
        ],
        'flow': 'horizontal'
    },
    {
        'header': 'LAYER 2 — PREPROCESSING',
        'color': '#466782',
        'boxes': [
            ['Frame Filtering', 'Quality-aware filtering', 'QAFF + pHash → 244,966'],
            ['Image Resizing', 'Standardize dimensions', 'Resize 768×768 WebP'],
            ['Text Alignment', 'Temporal synchronization', '20s sliding window']
        ],
        'flow': 'horizontal'
    },
    {
        'header': 'LAYER 3 — FEATURE EXTRACTION',
        'color': '#507390',
        'boxes': [
            ['Visual Embeddings', 'Image representation', 'SigLIP2 [416K×1152]'],
            ['Caption Generation', 'Scene understanding', 'Florence-2-base-ft [244K]'],
            ['Semantic Text (Transcripts)', 'Dense encoding', 'BGE-large [145K×1024]'],
            ['Semantic Text (Captions)', 'Dense encoding', 'BGE-large [244K×1024]']
        ],
        'flow': 'vertical_out'
    },
    {
        'header': 'LAYER 4 — INDEXING',
        'color': '#5A7F9E',
        'boxes': [
            ['Inverted Index', 'Text search structure', 'Whoosh BM25'],
            ['Visual Index', 'Vector similarity search', 'FAISS IndexFlatIP [416K]'],
            ['BM25 Tokenization', 'Sparse indexing', 'BM25 index creation'],
            ['Semantic Index (Transcripts)', 'Dense retrieval store', 'FAISS IndexFlatIP [145K]'],
            ['Semantic Index (Captions)', 'Dense retrieval store', 'FAISS IndexFlatIP [244K]']
        ],
        'flow': 'none',
        'custom_layout': 'two_rows'
    },
    {
        'header': 'LAYER 5 — RETRIEVAL',
        'color': '#648CAC',
        'boxes': [
            ['Approach A: BM25', 'Sparse text retrieval', 'Keyword matching'],
            ['Approach B: SigLIP2', 'Dense vision search', 'Visual similarity'],
            ['Approach C: Hybrid', 'Late fusion strategy', 'Multi-modal ensemble']
        ],
        'flow': 'horizontal',
        # highlight the fusion approach as the primary submission
        'highlight_box': 2
    },
    {
        'header': 'LAYER 6 — POST-PROCESSING',
        'color': '#42705D',
        'boxes': [
            ['Temporal Dedup', 'Remove near-duplicates', '±10s window filtering'],
            ['Multi-Angle Merge', 'Cross-stream fusion', 'Same event detection'],
            ['Result Ranking', 'Final score normalization', 'Top-10 truncation']
        ],
        'flow': 'horizontal'
    },
    {
        'header': 'LAYER 7 — OUTPUT',
        'color': '#4A7C68',
        'boxes': [
            ['Interactive UI', 'Web-based frontend', 'Streamlit application'],
            ['TREC Format', 'Standardized results', 'TSV output files'],
            ['Evaluation Metrics', 'Performance assessment', 'Precision@10 scoring']
        ],
        'flow': 'none'
    }
]

def add_shadow(ax, x, y, w, h, offset=4, alpha=0.08, rounding_size=12):
    shadow = FancyBboxPatch((x+offset, y+offset), w, h, boxstyle=f"round,pad=0,rounding_size={rounding_size}", 
                         facecolor='#000000', edgecolor='none', alpha=alpha)
    ax.add_patch(shadow)

def draw_box(ax, x, y, w, h, data, highlight=False):
    title, desc, detail = data
    border_color = '#C5A059' if highlight else '#9BA4B5'
    border_width = 3 if highlight else 1.2
    
    add_shadow(ax, x, y, w, h, offset=6, alpha=0.08, rounding_size=12)
    
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=12", 
                         facecolor='#FFFFFF', edgecolor=border_color, linewidth=border_width)
    ax.add_patch(box)
    
    cx = x + w/2
    ax.text(cx, y + h*0.28, title, fontsize=10.5, fontweight='bold', ha='center', va='center', color='#1A202C')
    ax.text(cx, y + h*0.52, desc, fontsize=9.5, style='italic', ha='center', va='center', color='#4A5568')
    ax.text(cx, y + h*0.75, detail, fontsize=9, ha='center', va='center', color='#718096')

LAYER_W = 950
MARGIN_X = 100
CENTER_X = MARGIN_X + LAYER_W/2

ax.text(CENTER_X, 50, 'Multimodal Object Search System Architecture', fontsize=22, fontweight='bold', ha='center', va='center', color='#1A202C')
ax.text(CENTER_X, 85, 'CASTLE 2024 Dataset  ·  Three Approaches  ·  TREC', fontsize=14, style='italic', ha='center', va='center', color='#718096')

Y_START = 130
GAP = 45
y_curr = Y_START
box_centers = {}

for idx, layer in enumerate(LAYERS):
    is_two_rows = layer.get('custom_layout') == 'two_rows'
    content_h = 240 if is_two_rows else 130
    layer_h = 40 + content_h
    
    shadow_rect = FancyBboxPatch((MARGIN_X+6, y_curr+6), LAYER_W, layer_h, boxstyle="round,pad=0,rounding_size=16", facecolor='#000000', alpha=0.04, edgecolor='none')
    ax.add_patch(shadow_rect)
    
    layer_bg = FancyBboxPatch((MARGIN_X, y_curr), LAYER_W, layer_h, boxstyle="round,pad=0,rounding_size=16", facecolor='#FCFDFD', edgecolor='none')
    ax.add_patch(layer_bg)
    
    header_top = FancyBboxPatch((MARGIN_X, y_curr), LAYER_W, 40, boxstyle="round,pad=0,rounding_size=16", facecolor=layer['color'], edgecolor='none')
    ax.add_patch(header_top)
    header_bottom = Rectangle((MARGIN_X, y_curr+20), LAYER_W, 20, facecolor=layer['color'], edgecolor='none')
    ax.add_patch(header_bottom)
    
    ax.text(CENTER_X, y_curr + 20, layer['header'], color='#FFFFFF', fontsize=13, fontweight='bold', ha='center', va='center')
    
    layer_border = FancyBboxPatch((MARGIN_X, y_curr), LAYER_W, layer_h, boxstyle="round,pad=0,rounding_size=16", facecolor='none', edgecolor=layer['color'], linewidth=1.5)
    ax.add_patch(layer_border)
    
    box_y = y_curr + 40 + 20
    box_h = 90
    
    layer_box_centers = []
    
    if is_two_rows:
        # first row: 3 boxes
        n1 = 3
        spacing1 = 30
        w1 = (LAYER_W - 40 - (n1-1)*spacing1) / n1
        for i in range(n1):
            bx = MARGIN_X + 20 + i*(w1 + spacing1)
            draw_box(ax, bx, box_y, w1, box_h, layer['boxes'][i])
            layer_box_centers.append((bx + w1/2, box_y + box_h/2))
            
        # second row: 2 wider boxes centred under the first row
        w2 = LAYER_W * 0.4
        by2 = box_y + box_h + 20
        spacing2 = 40
        bx_start = MARGIN_X + (LAYER_W - (w2*2 + spacing2)) / 2
        for i in range(2):
            bx = bx_start + i*(w2 + spacing2)
            draw_box(ax, bx, by2, w2, box_h, layer['boxes'][3+i])
            layer_box_centers.append((bx + w2/2, by2 + box_h/2))
            
    else:
        boxes = layer['boxes']
        n = len(boxes)
        spacing = 20 if n == 4 else 40
        w = (LAYER_W - 40 - (n-1)*spacing) / n
        for i in range(n):
            bx = MARGIN_X + 20 + i*(w + spacing)
            highlight = (layer.get('highlight_box') == i)
            draw_box(ax, bx, box_y, w, box_h, boxes[i], highlight)
            layer_box_centers.append((bx + w/2, box_y + box_h/2))
            
            if layer.get('flow') == 'horizontal' and i < n - 1:
                arr_x1 = bx + w + 5
                arr_x2 = bx + w + spacing - 5
                arr_y = box_y + box_h/2
                ax.annotate('', xy=(arr_x2, arr_y), xytext=(arr_x1, arr_y),
                            arrowprops=dict(arrowstyle='-|>', lw=2, color='#A0AEC0', mutation_scale=15))
                
    box_centers[idx] = layer_box_centers
    
    if idx < len(LAYERS) - 1:
        y_bottom = y_curr + layer_h
        y_next = y_curr + layer_h + GAP
        ax.annotate('', xy=(CENTER_X, y_next), xytext=(CENTER_X, y_bottom),
                    arrowprops=dict(arrowstyle='-|>', lw=2.5, color='#4A5568', mutation_scale=20))
    
    y_curr += layer_h + GAP

# draw the Rocchio feedback loop as a curved arrow on the right side
y5_center = box_centers[4][-1][1]
y6_center = box_centers[5][-1][1]

right_edge = MARGIN_X + LAYER_W
path_data = [
    (mpath.Path.MOVETO, (right_edge, y6_center)),
    (mpath.Path.CURVE3, (right_edge + 50, (y5_center + y6_center)/2)),
    (mpath.Path.CURVE3, (right_edge, y5_center))
]
codes, verts = zip(*path_data)
path = mpath.Path(verts, codes)
feedback_arrow = FancyArrowPatch(path=path, linestyle='--', color='#C5A059', lw=2.5, arrowstyle='-|>', mutation_scale=20)
ax.add_patch(feedback_arrow)

ax.text(right_edge - 10, (y5_center + y6_center)/2, 'Rocchio Feedback\n(User Relevance Judgments)', 
        fontsize=10.5, ha='right', va='center', color='#B2863B', fontweight='bold')

plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

os.makedirs('report', exist_ok=True)
plt.savefig('report/architecture_diagram_v2.png', dpi=100, facecolor=fig.get_facecolor(), edgecolor='none')
