#!/usr/bin/env python3
"""
generate_figures_matplotlib.py
Generates publication-grade, ultra-high-resolution vector figures for OrthoStyle using Matplotlib:
  1. qualitative_matrix_teaser.pdf / .png (5 contents x 15 styles + reference headers)
  2. qualitative_comparison_baselines.pdf / .png (12 curated pairs x 9 columns)
     & qualitative_comparison_baselines_cut.pdf / .png (5 curated pairs x 9 columns)
  3. ablation_components_cut.pdf / .png (3 pairs x 10 columns: A-F + 2 prompt levels)
     & ablation_components_.pdf / .png

Key advantage over PIL canvas:
- True vector headers, borders, and text labels (Type 42 / TrueType vector fonts)
- Embedded raster images retain 100% full original resolution (1024x1024 / 1440x1440)
- Zero blurriness / pixelation ("không bị bể ảnh") when zooming in PDF readers
"""

import os
import json
import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image

# Ensure TrueType vector fonts in PDF
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'DejaVu Sans'

WORKSPACE = '/mnt/wav2vec2/khoan/source_code/RB-Ortho/OthoStyle'
FIGS_DIR = os.path.join(WORKSPACE, 'Latex-Template-for-Springer/figs')
REPO_FIGS_DIR = os.path.join(WORKSPACE, 'figures')

os.makedirs(FIGS_DIR, exist_ok=True)
os.makedirs(REPO_FIGS_DIR, exist_ok=True)

def make_square(im):
    """Center crops any image to a 1:1 square without downsampling."""
    w, h = im.size
    if w == h:
        return im
    min_dim = min(w, h)
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    return im.crop((left, top, left + min_dim, top + min_dim))

def load_np_square(path):
    """Loads image, center-crops to 1:1 square, returns RGB numpy array."""
    if not os.path.exists(path):
        return None
    im = Image.open(path).convert("RGB")
    im_sq = make_square(im)
    return np.array(im_sq)

def save_mpl_figure(fig, base_name, dpi=300):
    """Saves figure to both Springer figs/ and repo figures/ directories as PDF and PNG."""
    out_pdf = os.path.join(FIGS_DIR, f"{base_name}.pdf")
    repo_pdf = os.path.join(REPO_FIGS_DIR, f"{base_name}.pdf")
    out_png = os.path.join(FIGS_DIR, f"{base_name}.png")
    repo_png = os.path.join(REPO_FIGS_DIR, f"{base_name}.png")
    
    t0 = time.time()
    fig.savefig(out_pdf, format='pdf', dpi=dpi, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(repo_pdf, format='pdf', dpi=dpi, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(out_png, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(repo_png, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    t1 = time.time()
    print(f"  [OK] Saved {base_name}.pdf & {base_name}.png (PDF size: {os.path.getsize(out_pdf)/1024:.1f} KB) in {t1-t0:.2f}s")

# ==============================================================================
# 1. QUALITATIVE MATRIX TEASER (6 Rows x 16 Columns)
# ==============================================================================
def generate_teaser_matrix_mpl():
    print("\n--- 1/4: Generating Qualitative Teaser Matrix via Matplotlib ---")
    contents = ['07_vintage_clock', '06_cat_sitting', '09_dog_sitting', '10_rubber_duck', '08_colorful_sneaker']
    styles = sorted([os.path.splitext(f)[0] for f in os.listdir(f'{WORKSPACE}/data/style') if f.endswith('.png')])

    num_rows = 1 + len(contents)
    num_cols = 1 + len(styles)

    # 16 cols x 6 rows -> width 26, height 10.2
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(26, 10.2))
    
    # Corner cell (0, 0): Clean diagonal indication
    corner_ax = axes[0, 0]
    corner_ax.set_facecolor('#f1f5f9')
    corner_ax.set_xticks([])
    corner_ax.set_yticks([])
    corner_ax.set_aspect('equal')
    corner_ax.plot([0, 1], [1, 0], color='#cbd5e1', transform=corner_ax.transAxes, linewidth=1.5)
    corner_ax.text(0.72, 0.72, "Style →", transform=corner_ax.transAxes, fontsize=10, fontweight='bold',
                   color='#1e293b', ha='center', va='center')
    corner_ax.text(0.28, 0.28, "↓ Content", transform=corner_ax.transAxes, fontsize=10, fontweight='bold',
                   color='#1e293b', ha='center', va='center')
    for spine in corner_ax.spines.values():
        spine.set_color('#cbd5e1')
        spine.set_linewidth(1.0)

    # Row 0: Style Reference Images
    for j, s in enumerate(styles, start=1):
        ax = axes[0, j]
        s_path = f"{WORKSPACE}/data/style/{s}.png"
        arr = load_np_square(s_path)
        if arr is not None:
            ax.imshow(arr, interpolation='lanczos')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        for spine in ax.spines.values():
            spine.set_color('#cbd5e1')
            spine.set_linewidth(1.0)

    # Col 0: Content Reference Images
    for i, c in enumerate(contents, start=1):
        ax = axes[i, 0]
        c_path = f"{WORKSPACE}/data/content/{c}.png"
        arr = load_np_square(c_path)
        if arr is not None:
            ax.imshow(arr, interpolation='lanczos')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        for spine in ax.spines.values():
            spine.set_color('#cbd5e1')
            spine.set_linewidth(1.0)

    # Matrix Cells (5 x 15): OrthoStyle Generated Results
    for i, c in enumerate(contents, start=1):
        for j, s in enumerate(styles, start=1):
            ax = axes[i, j]
            out_path = f"{WORKSPACE}/output/benchmark/level1_null/ortho_{c}_{s}.png"
            arr = load_np_square(out_path)
            if arr is not None:
                ax.imshow(arr, interpolation='lanczos')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')
            for spine in ax.spines.values():
                spine.set_color('#e2e8f0')
                spine.set_linewidth(0.8)

    plt.subplots_adjust(wspace=0.03, hspace=0.03, left=0.01, right=0.99, top=0.99, bottom=0.01)
    save_mpl_figure(fig, "qualitative_matrix_teaser", dpi=300)

# ==============================================================================
# 2. BASELINES QUALITATIVE COMPARISON (12 Pairs or 5 Pairs Cut x 9 Columns)
# ==============================================================================
def generate_baselines_comparison_mpl(cut=False):
    suffix = "_cut" if cut else ""
    print(f"\n--- 2/4: Generating Baselines Qualitative Comparison{suffix} via Matplotlib ---")

    styles = sorted([os.path.splitext(f)[0] for f in os.listdir(f'{WORKSPACE}/data/style') if f.endswith('.png')])

    if cut:
        pairs = [
            ('11_fancy_boot', '01_antimonocromatismo'),
            ('06_cat_sitting', '10_starry_night_sketch'),
            ('10_rubber_duck', '11_abstract_3d_render'),
            ('09_dog_sitting', '04_flat_illustration'),
            ('02_bear_plushie', '03_cyberpunk')
        ]
    else:
        with open(f"{WORKSPACE}/curated_paper_candidates.json") as f:
            curated = json.load(f)
        pairs = []
        for item in curated:
            parts = item.split('_')
            s_idx = -1
            for i in range(1, len(parts)):
                if any(s.startswith(parts[i]) for s in styles):
                    cand_s = '_'.join(parts[i:])
                    if cand_s in styles:
                        s_idx = i
                        break
            c_id = '_'.join(parts[:s_idx])
            s_id = '_'.join(parts[s_idx:])
            pairs.append((c_id, s_id))

    cols = [
        {"id": "content", "label": "Content", "bg": "#f1f5f9", "fg": "#0f172a"},
        {"id": "style", "label": "Style", "bg": "#f1f5f9", "fg": "#0f172a"},
        {"id": "styleid", "label": "StyleID", "bg": "#f8fafc", "fg": "#334155"},
        {"id": "attenst", "label": "AttenST", "bg": "#f8fafc", "fg": "#334155"},
        {"id": "diffuseIT", "label": "DiffuseIT", "bg": "#f8fafc", "fg": "#334155"},
        {"id": "instantstyle", "label": "InstantStyle", "bg": "#f8fafc", "fg": "#334155"},
        {"id": "itoc", "label": "ITOC", "bg": "#f8fafc", "fg": "#334155"},
        {"id": "rb", "label": "RB-Modulation", "bg": "#f8fafc", "fg": "#334155"},
        {"id": "ours", "label": "OrthoStyle (Ours)", "bg": "#e0e7ff", "fg": "#4338ca", "is_ours": True},
    ]

    num_rows = len(pairs)
    num_cols = len(cols)

    fig_w = 18.0
    fig_h = (num_rows * 2.05) + 0.8
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_w, fig_h))

    for r_idx, (c_id, s_id) in enumerate(pairs):
        method_paths = {
            "content": f"{WORKSPACE}/data/content/{c_id}.png",
            "style": f"{WORKSPACE}/data/style/{s_id}.png",
            "styleid": f"{WORKSPACE}/baseline_results/outputs_styleid_soictdata/styleid_{c_id}_{s_id}.png",
            "attenst": f"{WORKSPACE}/baseline_results/outputs_attenst_soictdata/attenst_{c_id}_{s_id}.png",
            "diffuseIT": f"{WORKSPACE}/baseline_results/outputs_diffuseIT_soictdata/DiffuseIT_{c_id}_{s_id}.png",
            "instantstyle": f"{WORKSPACE}/baseline_results/outputs_instantstyle_soictdata/instantstyle_{c_id}_{s_id}.png",
            "itoc": f"{WORKSPACE}/baseline_results/outputs_itoc_soictdata/itoc_{c_id}_{s_id}.png",
            "rb": f"{WORKSPACE}/baseline_results/outputs_rb_soictdata/outputs_rb_soictdata_{c_id}_{s_id}.png",
            "ours": f"{WORKSPACE}/output/benchmark/level1_null/ortho_{c_id}_{s_id}.png",
        }

        for c_idx, col in enumerate(cols):
            ax = axes[r_idx, c_idx]
            mid = col["id"]
            p = method_paths.get(mid)
            arr = load_np_square(p)
            if arr is not None:
                ax.imshow(arr, interpolation='lanczos')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')

            is_ours = col.get("is_ours", False)
            for spine in ax.spines.values():
                if is_ours:
                    spine.set_color('#6366f1')
                    spine.set_linewidth(2.2)
                else:
                    spine.set_color('#cbd5e1')
                    spine.set_linewidth(0.9)

            # Vector Column Headers on Top Row
            if r_idx == 0:
                outline_color = '#6366f1' if is_ours else '#cbd5e1'
                lw = 1.6 if is_ours else 1.0
                ax.set_title(col["label"], fontsize=11, fontweight='bold', pad=10,
                             color=col["fg"],
                             bbox=dict(boxstyle='round,pad=0.4,rounding_size=0.25',
                                       facecolor=col["bg"],
                                       edgecolor=outline_color,
                                       linewidth=lw))

    plt.subplots_adjust(wspace=0.035, hspace=0.04, left=0.01, right=0.99, top=0.94 if cut else 0.97, bottom=0.01)
    base_name = f"qualitative_comparison_baselines{suffix}"
    save_mpl_figure(fig, base_name, dpi=300)

# ==============================================================================
# 3. ABLATION ARCHITECTURAL COMPONENTS CUT (3 Pairs x 10 Columns)
# ==============================================================================
def generate_ablation_components_cut_mpl():
    print("\n--- 3/4: Generating Ablation Architectural Components Cut via Matplotlib ---")
    pairs = [
        ('01_backpack_dog', '06_historical_oil'),
        ('03_berry_bowl', '04_flat_illustration'),
        ('04_can', '13_3d_isometric')
    ]

    cols = [
        {"id": "content", "label": "Content", "bg": "#f1f5f9", "fg": "#0f172a"},
        {"id": "style", "label": "Style", "bg": "#f1f5f9", "fg": "#0f172a"},
        {"id": "full", "label": "(A) Full OrthoStyle", "bg": "#e0e7ff", "fg": "#4338ca", "is_full": True},
        {"id": "pure_mean", "label": "(B) Pure Mean", "bg": "#f8fafc", "fg": "#1e293b"},
        {"id": "raw_style", "label": "(C) Raw Style", "bg": "#f8fafc", "fg": "#1e293b"},
        {"id": "no_guidance", "label": "(D) w/o Score Guidance", "bg": "#f8fafc", "fg": "#1e293b"},
        {"id": "no_pushforward", "label": "(E) w/o Pushforward", "bg": "#f8fafc", "fg": "#1e293b"},
        {"id": "no_gating", "label": "(F) w/o Gated Canny", "bg": "#f8fafc", "fg": "#1e293b"},
        {"id": "level2_object", "label": "Object Prompt (\"a <obj>\")", "bg": "#eef2ff", "fg": "#4338ca", "is_prompt": True},
        {"id": "level3_style_desc", "label": "Style Prompt (\"<obj> in <sty>\")", "bg": "#eef2ff", "fg": "#4338ca", "is_prompt": True},
    ]

    num_rows = len(pairs)
    num_cols = len(cols)

    fig_w = 20.0
    fig_h = (num_rows * 2.05) + 0.8
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_w, fig_h))

    for r_idx, (c_id, s_id) in enumerate(pairs):
        ablation_paths = {
            "content": f"{WORKSPACE}/data/content/{c_id}.png",
            "style": f"{WORKSPACE}/data/style/{s_id}.png",
            "full": f"{WORKSPACE}/output/benchmark/level1_null/ortho_{c_id}_{s_id}.png",
            "pure_mean": f"{WORKSPACE}/output/benchmark/ablation_B_pure_mean/level1_null/ortho_{c_id}_{s_id}.png",
            "raw_style": f"{WORKSPACE}/output/benchmark/ablation_C_raw_style/level1_null/ortho_{c_id}_{s_id}.png",
            "no_guidance": f"{WORKSPACE}/output/benchmark/ablation_D_no_guidance/level1_null/ortho_{c_id}_{s_id}.png",
            "no_pushforward": f"{WORKSPACE}/output/benchmark/ablation_E_no_pushforward/level1_null/ortho_{c_id}_{s_id}.png",
            "no_gating": f"{WORKSPACE}/output/benchmark/ablation_F_no_gating/level1_null/ortho_{c_id}_{s_id}.png",
            "level2_object": f"{WORKSPACE}/output/benchmark/level2_object/ortho_{c_id}_{s_id}.png",
            "level3_style_desc": f"{WORKSPACE}/output/benchmark/level3_style_desc/ortho_{c_id}_{s_id}.png",
        }

        for c_idx, col in enumerate(cols):
            ax = axes[r_idx, c_idx]
            mid = col["id"]
            p = ablation_paths.get(mid)
            arr = load_np_square(p)
            if arr is not None:
                ax.imshow(arr, interpolation='lanczos')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')

            is_full = col.get("is_full", False)
            is_prompt = col.get("is_prompt", False)

            for spine in ax.spines.values():
                if is_full:
                    spine.set_color('#6366f1')
                    spine.set_linewidth(2.2)
                elif is_prompt:
                    spine.set_color('#818cf8')
                    spine.set_linewidth(1.5)
                else:
                    spine.set_color('#cbd5e1')
                    spine.set_linewidth(0.9)

            # Vector Column Headers on Top Row
            if r_idx == 0:
                outline_color = '#6366f1' if is_full else ('#818cf8' if is_prompt else '#cbd5e1')
                lw = 1.6 if (is_full or is_prompt) else 1.0
                ax.set_title(col["label"], fontsize=9.2, fontweight='bold', pad=10,
                             color=col["fg"],
                             bbox=dict(boxstyle='round,pad=0.38,rounding_size=0.25',
                                       facecolor=col["bg"],
                                       edgecolor=outline_color,
                                       linewidth=lw))

    plt.subplots_adjust(wspace=0.035, hspace=0.04, left=0.01, right=0.99, top=0.90, bottom=0.01)
    
    # Save to both ablation_components_cut and ablation_components_
    out_pdf = os.path.join(FIGS_DIR, "ablation_components_cut.pdf")
    repo_pdf = os.path.join(REPO_FIGS_DIR, "ablation_components_cut.pdf")
    out_png = os.path.join(FIGS_DIR, "ablation_components_cut.png")
    repo_png = os.path.join(REPO_FIGS_DIR, "ablation_components_cut.png")
    
    t0 = time.time()
    fig.savefig(out_pdf, format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(repo_pdf, format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(out_png, format='png', dpi=300, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(repo_png, format='png', dpi=300, bbox_inches='tight', pad_inches=0.02)
    
    # Also duplicate to ablation_components_ for compatibility
    for dst in [os.path.join(FIGS_DIR, "ablation_components_.pdf"), os.path.join(REPO_FIGS_DIR, "ablation_components_.pdf")]:
        import shutil
        shutil.copyfile(out_pdf, dst)
    for dst in [os.path.join(FIGS_DIR, "ablation_components_.png"), os.path.join(REPO_FIGS_DIR, "ablation_components_.png")]:
        import shutil
        shutil.copyfile(out_png, dst)
    
    plt.close(fig)
    t1 = time.time()
    print(f"  [OK] Saved ablation_components_cut.pdf & .png (PDF size: {os.path.getsize(out_pdf)/1024:.1f} KB) in {t1-t0:.2f}s")

if __name__ == "__main__":
    t_start = time.time()
    generate_teaser_matrix_mpl()
    generate_baselines_comparison_mpl(cut=False)
    generate_baselines_comparison_mpl(cut=True)
    generate_ablation_components_cut_mpl()
    print(f"\n=== All High-Resolution Vector Figures Generated Successfully in {time.time()-t_start:.1f}s! ===")
