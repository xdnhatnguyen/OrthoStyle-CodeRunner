#!/usr/bin/env python3
"""
generate_paper_figures.py
Generates 4 publication-grade figures for OrthoStyle with strictly 1:1 aspect ratio:
  1. qualitative_matrix_teaser.png / .pdf (5 contents x 15 styles + reference headers, 1:1 square cells, no banners)
  2. qualitative_comparison_baselines.png / .pdf (Curated 12 pairs vs 6 baselines, 1:1 square cells, clean headers)
  3. ablation_prompt_levels.png / .pdf (3 prompt levels for 3 selected pairs)
  4. ablation_components.png / .pdf (Architectural components ablation for 5 selected pairs)
"""

import os
import json
import time
from PIL import Image, ImageDraw, ImageFont

WORKSPACE = '/mnt/wav2vec2/khoan/source_code/RB-Ortho/OthoStyle'
FIGS_DIR = os.path.join(WORKSPACE, 'Latex-Template-for-Springer/figs')
REPO_FIGS_DIR = os.path.join(WORKSPACE, 'figures')

os.makedirs(FIGS_DIR, exist_ok=True)
os.makedirs(REPO_FIGS_DIR, exist_ok=True)

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_REGULAR_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

def get_font(size, bold=False):
    path = FONT_PATH if bold else FONT_REGULAR_PATH
    if os.path.exists(path):
        return ImageFont.truetype(path, size)
    return ImageFont.load_default()

def make_square(im):
    """Center crops any image to a 1:1 square to prevent any aspect ratio distortion."""
    w, h = im.size
    if w == h:
        return im
    min_dim = min(w, h)
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    return im.crop((left, top, left + min_dim, top + min_dim))

def save_figure(canvas, base_name):
    out_png = os.path.join(FIGS_DIR, f"{base_name}.png")
    repo_png = os.path.join(REPO_FIGS_DIR, f"{base_name}.png")
    out_pdf = os.path.join(FIGS_DIR, f"{base_name}.pdf")
    repo_pdf = os.path.join(REPO_FIGS_DIR, f"{base_name}.pdf")
    
    t0 = time.time()
    canvas.save(out_png)
    canvas.save(repo_png)
    canvas.save(out_pdf, "PDF", resolution=300.0)
    canvas.save(repo_pdf, "PDF", resolution=300.0)
    t1 = time.time()
    print(f"  [OK] Saved {base_name}.png & {base_name}.pdf ({canvas.size[0]}x{canvas.size[1]}) in {t1-t0:.2f}s")

# ==============================================================================
# 1. QUALITATIVE TEASER MATRIX (5 Contents x 15 Styles, 1:1 Strict Square, No Banners)
# ==============================================================================
def generate_teaser_matrix():
    print("\n--- 1/4: Generating Qualitative Teaser Matrix (1:1 Strict Squares) ---")
    contents = ['07_vintage_clock', '06_cat_sitting', '09_dog_sitting', '10_rubber_duck', '08_colorful_sneaker']
    styles = sorted([os.path.splitext(f)[0] for f in os.listdir(f'{WORKSPACE}/data/style') if f.endswith('.png')])

    cell_size = 240
    pad = 4

    # Grid: 1 header col (Content) + 15 styles cols = 16 cols
    #       1 header row (Style)   + 5 contents rows = 6 rows
    cols_count = 1 + len(styles)
    rows_count = 1 + len(contents)

    total_w = cols_count * (cell_size + pad) + pad
    total_h = rows_count * (cell_size + pad) + pad

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    corner_font = get_font(18, bold=True)

    # Top-Left Corner Cell (0, 0): Clean label indicating Content vs Style
    corner_x = pad
    corner_y = pad
    draw.rounded_rectangle([corner_x, corner_y, corner_x + cell_size, corner_y + cell_size], radius=4, fill=(241, 245, 249), outline=(203, 213, 225), width=1)
    
    # Draw subtle diagonal line or clean labels
    draw.line([corner_x + 10, corner_y + 10, corner_x + cell_size - 10, corner_y + cell_size - 10], fill=(203, 213, 225), width=2)
    draw.text((corner_x + cell_size - 85, corner_y + 25), "Style →", font=corner_font, fill=(30, 41, 59))
    draw.text((corner_x + 20, corner_y + cell_size - 50), "↓ Content", font=corner_font, fill=(30, 41, 59))

    # Row 0: Style reference images (strict 1:1 square, no captions)
    for j, s in enumerate(styles, start=1):
        x = j * (cell_size + pad) + pad
        y = pad
        s_img_path = f"{WORKSPACE}/data/style/{s}.png"
        if os.path.exists(s_img_path):
            s_img = Image.open(s_img_path).convert("RGB")
            s_img = make_square(s_img).resize((cell_size, cell_size), Image.Resampling.LANCZOS)
            canvas.paste(s_img, (x, y))
            draw.rectangle([x, y, x + cell_size, y + cell_size], outline=(203, 213, 225), width=1)

    # Column 0: Content reference images (strict 1:1 square, no captions)
    for i, c in enumerate(contents, start=1):
        x = pad
        y = i * (cell_size + pad) + pad
        c_img_path = f"{WORKSPACE}/data/content/{c}.png"
        if os.path.exists(c_img_path):
            c_img = Image.open(c_img_path).convert("RGB")
            c_img = make_square(c_img).resize((cell_size, cell_size), Image.Resampling.LANCZOS)
            canvas.paste(c_img, (x, y))
            draw.rectangle([x, y, x + cell_size, y + cell_size], outline=(203, 213, 225), width=1)

    # Matrix Cells (5 x 15): OrthoStyle Generated Results
    for i, c in enumerate(contents, start=1):
        y = i * (cell_size + pad) + pad
        for j, s in enumerate(styles, start=1):
            x = j * (cell_size + pad) + pad
            out_path = f"{WORKSPACE}/output/benchmark/level1_null/ortho_{c}_{s}.png"
            if os.path.exists(out_path):
                out_img = Image.open(out_path).convert("RGB")
                out_img = make_square(out_img).resize((cell_size, cell_size), Image.Resampling.LANCZOS)
                canvas.paste(out_img, (x, y))
                draw.rectangle([x, y, x + cell_size, y + cell_size], outline=(226, 232, 240), width=1)

    save_figure(canvas, "qualitative_matrix_teaser")

# ==============================================================================
# 2. BASELINES QUALITATIVE COMPARISON (Curated 12 Pairs x 9 Columns, 1:1 Squares)
# ==============================================================================
def generate_baselines_comparison():
    print("\n--- 2/4: Generating Baselines Qualitative Comparison (1:1 Strict Squares) ---")
    with open(f"{WORKSPACE}/curated_paper_candidates.json") as f:
        curated = json.load(f)

    styles = sorted([os.path.splitext(f)[0] for f in os.listdir(f'{WORKSPACE}/data/style') if f.endswith('.png')])

    # Column definitions: clean labels, no cluttered sub-captions
    cols = [
        {"id": "content", "label": "Content", "bg": (241, 245, 249), "fg": (15, 23, 42)},
        {"id": "style", "label": "Style", "bg": (241, 245, 249), "fg": (15, 23, 42)},
        {"id": "styleid", "label": "StyleID", "bg": (248, 250, 252), "fg": (51, 65, 85)},
        {"id": "attenst", "label": "AttenST", "bg": (248, 250, 252), "fg": (51, 65, 85)},
        {"id": "diffuseIT", "label": "DiffuseIT", "bg": (248, 250, 252), "fg": (51, 65, 85)},
        {"id": "instantstyle", "label": "InstantStyle", "bg": (248, 250, 252), "fg": (51, 65, 85)},
        {"id": "itoc", "label": "ITOC", "bg": (248, 250, 252), "fg": (51, 65, 85)},
        {"id": "rb", "label": "RB-Modulation", "bg": (248, 250, 252), "fg": (51, 65, 85)},
        {"id": "ours", "label": "OrthoStyle (Ours)", "bg": (224, 231, 255), "fg": (67, 56, 202), "is_ours": True},
    ]

    cell_size = 256
    pad = 5
    top_header = 45

    total_w = len(cols) * (cell_size + pad) + pad
    total_h = top_header + len(curated) * (cell_size + pad) + pad

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    header_font = get_font(16, bold=True)

    # Top Column Headers
    for c_idx, col in enumerate(cols):
        x = c_idx * (cell_size + pad) + pad
        is_ours = col.get("is_ours", False)
        
        banner_w = cell_size
        banner_h = top_header - 10
        draw.rounded_rectangle([x, 5, x + banner_w, 5 + banner_h], radius=4, fill=col["bg"], outline=(99, 102, 241) if is_ours else (203, 213, 225), width=2 if is_ours else 1)
        
        bbox = header_font.getbbox(col["label"])
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text((x + (banner_w - tw)//2, 5 + (banner_h - th)//2 - 2), col["label"], font=header_font, fill=col["fg"])

    # Rows (1:1 strict square cells)
    for r_idx, item in enumerate(curated):
        y = top_header + r_idx * (cell_size + pad) + pad
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

        # 1. Content
        c_path = f"{WORKSPACE}/data/content/{c_id}.png"
        if os.path.exists(c_path):
            img = Image.open(c_path).convert("RGB")
            img = make_square(img).resize((cell_size, cell_size), Image.Resampling.LANCZOS)
            x = 0 * (cell_size + pad) + pad
            canvas.paste(img, (x, y))
            draw.rectangle([x, y, x + cell_size, y + cell_size], outline=(203, 213, 225), width=1)

        # 2. Style (strict 1:1 square)
        s_path = f"{WORKSPACE}/data/style/{s_id}.png"
        if os.path.exists(s_path):
            img = Image.open(s_path).convert("RGB")
            img = make_square(img).resize((cell_size, cell_size), Image.Resampling.LANCZOS)
            x = 1 * (cell_size + pad) + pad
            canvas.paste(img, (x, y))
            draw.rectangle([x, y, x + cell_size, y + cell_size], outline=(203, 213, 225), width=1)

        # Baselines & Ours
        method_paths = {
            "styleid": f"{WORKSPACE}/baseline_results/outputs_styleid_soictdata/styleid_{c_id}_{s_id}.png",
            "attenst": f"{WORKSPACE}/baseline_results/outputs_attenst_soictdata/attenst_{c_id}_{s_id}.png",
            "diffuseIT": f"{WORKSPACE}/baseline_results/outputs_diffuseIT_soictdata/DiffuseIT_{c_id}_{s_id}.png",
            "instantstyle": f"{WORKSPACE}/baseline_results/outputs_instantstyle_soictdata/instantstyle_{c_id}_{s_id}.png",
            "itoc": f"{WORKSPACE}/baseline_results/outputs_itoc_soictdata/itoc_{c_id}_{s_id}.png",
            "rb": f"{WORKSPACE}/baseline_results/outputs_rb_soictdata/outputs_rb_soictdata_{c_id}_{s_id}.png",
            "ours": f"{WORKSPACE}/output/benchmark/level1_null/ortho_{c_id}_{s_id}.png",
        }

        for c_idx, col in enumerate(cols[2:], start=2):
            mid = col["id"]
            p = method_paths.get(mid)
            if p and os.path.exists(p):
                img = Image.open(p).convert("RGB")
                img = make_square(img).resize((cell_size, cell_size), Image.Resampling.LANCZOS)
                x = c_idx * (cell_size + pad) + pad
                canvas.paste(img, (x, y))
                if col.get("is_ours"):
                    draw.rectangle([x, y, x + cell_size, y + cell_size], outline=(99, 102, 241), width=3)
                else:
                    draw.rectangle([x, y, x + cell_size, y + cell_size], outline=(226, 232, 240), width=1)

    save_figure(canvas, "qualitative_comparison_baselines")

# ==============================================================================
# 3. ABLATION PROMPT LEVELS (3 Pairs x 5 Columns, 1:1 Squares)
# ==============================================================================
def generate_ablation_prompt_levels():
    print("\n--- 3/4: Generating Ablation Prompt Levels ---")
    pairs = [
        ('01_backpack_dog', '15_oil_pastels'),
        ('02_bear_plushie', '14_crayon_drawing'),
        ('07_vintage_clock', '08_pencil_sketch')
    ]

    cols = [
        {"id": "content", "label": "Content", "bg": (241, 245, 249)},
        {"id": "style", "label": "Style", "bg": (241, 245, 249)},
        {"id": "level1", "label": "Null Prompt (\"\")", "bg": (238, 242, 255)},
        {"id": "level2", "label": "Object Prompt (\"a <obj>\")", "bg": (238, 242, 255)},
        {"id": "level3", "label": "Style Prompt (\"<obj> in <sty>\")", "bg": (238, 242, 255)},
    ]

    cell_size = 280
    pad = 6
    top_header = 45

    total_w = len(cols) * (cell_size + pad) + pad
    total_h = top_header + len(pairs) * (cell_size + pad) + pad

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    header_font = get_font(15, bold=True)

    # Headers
    for c_idx, col in enumerate(cols):
        x = c_idx * (cell_size + pad) + pad
        banner_w = cell_size
        banner_h = top_header - 10
        draw.rounded_rectangle([x, 5, x + banner_w, 5 + banner_h], radius=4, fill=col["bg"], outline=(203, 213, 225), width=1)
        
        bbox = header_font.getbbox(col["label"])
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text((x + (banner_w - tw)//2, 5 + (banner_h - th)//2 - 2), col["label"], font=header_font, fill=(15, 23, 42))

    # Rows
    for r_idx, (c_id, s_id) in enumerate(pairs):
        y = top_header + r_idx * (cell_size + pad) + pad

        # Content
        c_path = f"{WORKSPACE}/data/content/{c_id}.png"
        if os.path.exists(c_path):
            img = Image.open(c_path).convert("RGB")
            img = make_square(img).resize((cell_size, cell_size), Image.Resampling.LANCZOS)
            x = 0 * (cell_size + pad) + pad
            canvas.paste(img, (x, y))
            draw.rectangle([x, y, x + cell_size, y + cell_size], outline=(203, 213, 225), width=1)

        # Style
        s_path = f"{WORKSPACE}/data/style/{s_id}.png"
        if os.path.exists(s_path):
            img = Image.open(s_path).convert("RGB")
            img = make_square(img).resize((cell_size, cell_size), Image.Resampling.LANCZOS)
            x = 1 * (cell_size + pad) + pad
            canvas.paste(img, (x, y))
            draw.rectangle([x, y, x + cell_size, y + cell_size], outline=(203, 213, 225), width=1)

        # Levels
        lvl_paths = [
            f"{WORKSPACE}/output/benchmark/level1_null/ortho_{c_id}_{s_id}.png",
            f"{WORKSPACE}/output/benchmark/level2_object/ortho_{c_id}_{s_id}.png",
            f"{WORKSPACE}/output/benchmark/level3_style_desc/ortho_{c_id}_{s_id}.png"
        ]
        for l_idx, p in enumerate(lvl_paths, start=2):
            if os.path.exists(p):
                img = Image.open(p).convert("RGB")
                img = make_square(img).resize((cell_size, cell_size), Image.Resampling.LANCZOS)
                x = l_idx * (cell_size + pad) + pad
                canvas.paste(img, (x, y))
                draw.rectangle([x, y, x + cell_size, y + cell_size], outline=(199, 210, 254), width=1)

    save_figure(canvas, "ablation_prompt_levels")

# ==============================================================================
# 4. ABLATION ARCHITECTURAL COMPONENTS (5 Pairs x 8 Columns, 1:1 Squares)
# ==============================================================================
def generate_ablation_components():
    print("\n--- 4/4: Generating Ablation Architectural Components ---")
    pairs = [
        ('01_backpack_dog', '06_historical_oil'),
        ('03_berry_bowl', '04_flat_illustration'),
        ('04_can', '13_3d_isometric'),
        ('07_vintage_clock', '08_pencil_sketch'),
        ('09_dog_sitting', '14_crayon_drawing')
    ]

    cols = [
        {"id": "content", "label": "Content", "bg": (241, 245, 249)},
        {"id": "style", "label": "Style", "bg": (241, 245, 249)},
        {"id": "full", "label": "(A) Full OrthoStyle", "bg": (224, 231, 255), "is_full": True},
        {"id": "pure_mean", "label": "(B) Pure Mean", "bg": (248, 250, 252)},
        {"id": "raw_style", "label": "(C) Raw Style", "bg": (248, 250, 252)},
        {"id": "no_guidance", "label": "(D) w/o Score Guidance", "bg": (248, 250, 252)},
        {"id": "no_pushforward", "label": "(E) w/o Pushforward", "bg": (248, 250, 252)},
        {"id": "no_gating", "label": "(F) w/o Gated Canny", "bg": (248, 250, 252)},
    ]

    cell_size = 256
    pad = 5
    top_header = 45

    total_w = len(cols) * (cell_size + pad) + pad
    total_h = top_header + len(pairs) * (cell_size + pad) + pad

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    header_font = get_font(14, bold=True)

    # Headers
    for c_idx, col in enumerate(cols):
        x = c_idx * (cell_size + pad) + pad
        is_full = col.get("is_full", False)
        banner_w = cell_size
        banner_h = top_header - 10
        draw.rounded_rectangle([x, 5, x + banner_w, 5 + banner_h], radius=4, fill=col["bg"], outline=(99, 102, 241) if is_full else (203, 213, 225), width=2 if is_full else 1)
        
        bbox = header_font.getbbox(col["label"])
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text((x + (banner_w - tw)//2, 5 + (banner_h - th)//2 - 2), col["label"], font=header_font, fill=(67, 56, 202) if is_full else (15, 23, 42))

    # Rows
    for r_idx, (c_id, s_id) in enumerate(pairs):
        y = top_header + r_idx * (cell_size + pad) + pad

        # Content
        c_path = f"{WORKSPACE}/data/content/{c_id}.png"
        if os.path.exists(c_path):
            img = Image.open(c_path).convert("RGB")
            img = make_square(img).resize((cell_size, cell_size), Image.Resampling.LANCZOS)
            x = 0 * (cell_size + pad) + pad
            canvas.paste(img, (x, y))
            draw.rectangle([x, y, x + cell_size, y + cell_size], outline=(203, 213, 225), width=1)

        # Style
        s_path = f"{WORKSPACE}/data/style/{s_id}.png"
        if os.path.exists(s_path):
            img = Image.open(s_path).convert("RGB")
            img = make_square(img).resize((cell_size, cell_size), Image.Resampling.LANCZOS)
            x = 1 * (cell_size + pad) + pad
            canvas.paste(img, (x, y))
            draw.rectangle([x, y, x + cell_size, y + cell_size], outline=(203, 213, 225), width=1)

        # Ablation outputs
        ablation_paths = {
            "full": f"{WORKSPACE}/output/benchmark/level1_null/ortho_{c_id}_{s_id}.png",
            "pure_mean": f"{WORKSPACE}/output/benchmark/ablation_B_pure_mean/level1_null/ortho_{c_id}_{s_id}.png",
            "raw_style": f"{WORKSPACE}/output/benchmark/ablation_C_raw_style/level1_null/ortho_{c_id}_{s_id}.png",
            "no_guidance": f"{WORKSPACE}/output/benchmark/ablation_D_no_guidance/level1_null/ortho_{c_id}_{s_id}.png",
            "no_pushforward": f"{WORKSPACE}/output/benchmark/ablation_E_no_pushforward/level1_null/ortho_{c_id}_{s_id}.png",
            "no_gating": f"{WORKSPACE}/output/benchmark/ablation_F_no_gating/level1_null/ortho_{c_id}_{s_id}.png",
        }

        for c_idx, col in enumerate(cols[2:], start=2):
            mid = col["id"]
            p = ablation_paths.get(mid)
            if p and os.path.exists(p):
                img = Image.open(p).convert("RGB")
                img = make_square(img).resize((cell_size, cell_size), Image.Resampling.LANCZOS)
                x = c_idx * (cell_size + pad) + pad
                canvas.paste(img, (x, y))
                if col.get("is_full"):
                    draw.rectangle([x, y, x + cell_size, y + cell_size], outline=(99, 102, 241), width=3)
                else:
                    draw.rectangle([x, y, x + cell_size, y + cell_size], outline=(226, 232, 240), width=1)

    save_figure(canvas, "ablation_components")

# ==============================================================================
# 5. BASELINES QUALITATIVE COMPARISON CUT (5 Pairs x 9 Columns, 1:1 Squares)
# ==============================================================================
def generate_baselines_comparison_cut():
    print("\n--- 5/6: Generating Baselines Qualitative Comparison Cut (5 Pairs x 9 Columns) ---")
    curated_cut = [
        ('11_fancy_boot', '01_antimonocromatismo'),
        ('06_cat_sitting', '10_starry_night_sketch'),
        ('10_rubber_duck', '11_abstract_3d_render'),
        ('09_dog_sitting', '04_flat_illustration'),
        ('02_bear_plushie', '03_cyberpunk')
    ]

    cols = [
        {"id": "content", "label": "Content", "bg": (241, 245, 249), "fg": (15, 23, 42)},
        {"id": "style", "label": "Style", "bg": (241, 245, 249), "fg": (15, 23, 42)},
        {"id": "styleid", "label": "StyleID", "bg": (248, 250, 252), "fg": (51, 65, 85)},
        {"id": "attenst", "label": "AttenST", "bg": (248, 250, 252), "fg": (51, 65, 85)},
        {"id": "diffuseIT", "label": "DiffuseIT", "bg": (248, 250, 252), "fg": (51, 65, 85)},
        {"id": "instantstyle", "label": "InstantStyle", "bg": (248, 250, 252), "fg": (51, 65, 85)},
        {"id": "itoc", "label": "ITOC", "bg": (248, 250, 252), "fg": (51, 65, 85)},
        {"id": "rb", "label": "RB-Modulation", "bg": (248, 250, 252), "fg": (51, 65, 85)},
        {"id": "ours", "label": "OrthoStyle (Ours)", "bg": (224, 231, 255), "fg": (67, 56, 202), "is_ours": True},
    ]

    cell_size = 256
    pad = 5
    top_header = 45

    total_w = len(cols) * (cell_size + pad) + pad
    total_h = top_header + len(curated_cut) * (cell_size + pad) + pad

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    header_font = get_font(16, bold=True)

    # Top Column Headers
    for c_idx, col in enumerate(cols):
        x = c_idx * (cell_size + pad) + pad
        is_ours = col.get("is_ours", False)
        
        banner_w = cell_size
        banner_h = top_header - 10
        draw.rounded_rectangle([x, 5, x + banner_w, 5 + banner_h], radius=4, fill=col["bg"], outline=(99, 102, 241) if is_ours else (203, 213, 225), width=2 if is_ours else 1)
        
        bbox = header_font.getbbox(col["label"])
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text((x + (banner_w - tw)//2, 5 + (banner_h - th)//2 - 2), col["label"], font=header_font, fill=col["fg"])

    # Rows (1:1 strict square cells)
    for r_idx, (c_id, s_id) in enumerate(curated_cut):
        y = top_header + r_idx * (cell_size + pad) + pad

        # 1. Content
        c_path = f"{WORKSPACE}/data/content/{c_id}.png"
        if os.path.exists(c_path):
            img = Image.open(c_path).convert("RGB")
            img = make_square(img).resize((cell_size, cell_size), Image.Resampling.LANCZOS)
            x = 0 * (cell_size + pad) + pad
            canvas.paste(img, (x, y))
            draw.rectangle([x, y, x + cell_size, y + cell_size], outline=(203, 213, 225), width=1)

        # 2. Style (strict 1:1 square)
        s_path = f"{WORKSPACE}/data/style/{s_id}.png"
        if os.path.exists(s_path):
            img = Image.open(s_path).convert("RGB")
            img = make_square(img).resize((cell_size, cell_size), Image.Resampling.LANCZOS)
            x = 1 * (cell_size + pad) + pad
            canvas.paste(img, (x, y))
            draw.rectangle([x, y, x + cell_size, y + cell_size], outline=(203, 213, 225), width=1)

        # Baselines & Ours
        method_paths = {
            "styleid": f"{WORKSPACE}/baseline_results/outputs_styleid_soictdata/styleid_{c_id}_{s_id}.png",
            "attenst": f"{WORKSPACE}/baseline_results/outputs_attenst_soictdata/attenst_{c_id}_{s_id}.png",
            "diffuseIT": f"{WORKSPACE}/baseline_results/outputs_diffuseIT_soictdata/DiffuseIT_{c_id}_{s_id}.png",
            "instantstyle": f"{WORKSPACE}/baseline_results/outputs_instantstyle_soictdata/instantstyle_{c_id}_{s_id}.png",
            "itoc": f"{WORKSPACE}/baseline_results/outputs_itoc_soictdata/itoc_{c_id}_{s_id}.png",
            "rb": f"{WORKSPACE}/baseline_results/outputs_rb_soictdata/outputs_rb_soictdata_{c_id}_{s_id}.png",
            "ours": f"{WORKSPACE}/output/benchmark/level1_null/ortho_{c_id}_{s_id}.png",
        }

        for c_idx, col in enumerate(cols[2:], start=2):
            mid = col["id"]
            p = method_paths.get(mid)
            if p and os.path.exists(p):
                img = Image.open(p).convert("RGB")
                img = make_square(img).resize((cell_size, cell_size), Image.Resampling.LANCZOS)
                x = c_idx * (cell_size + pad) + pad
                canvas.paste(img, (x, y))
                if col.get("is_ours"):
                    draw.rectangle([x, y, x + cell_size, y + cell_size], outline=(99, 102, 241), width=3)
                else:
                    draw.rectangle([x, y, x + cell_size, y + cell_size], outline=(226, 232, 240), width=1)

    save_figure(canvas, "qualitative_comparison_baselines_cut")

# ==============================================================================
# 6. ABLATION ARCHITECTURAL COMPONENTS CUT (3 Pairs x 10 Columns, 1:1 Squares)
# ==============================================================================
def generate_ablation_components_cut():
    print("\n--- 6/6: Generating Ablation Architectural Components Cut (3 Pairs x 10 Columns) ---")
    pairs = [
        ('01_backpack_dog', '06_historical_oil'),
        ('03_berry_bowl', '04_flat_illustration'),
        ('04_can', '13_3d_isometric')
    ]

    cols = [
        {"id": "content", "label": "Content", "bg": (241, 245, 249)},
        {"id": "style", "label": "Style", "bg": (241, 245, 249)},
        {"id": "full", "label": "(A) Full OrthoStyle", "bg": (224, 231, 255), "is_full": True},
        {"id": "pure_mean", "label": "(B) Pure Mean", "bg": (248, 250, 252)},
        {"id": "raw_style", "label": "(C) Raw Style", "bg": (248, 250, 252)},
        {"id": "no_guidance", "label": "(D) w/o Score Guidance", "bg": (248, 250, 252)},
        {"id": "no_pushforward", "label": "(E) w/o Pushforward", "bg": (248, 250, 252)},
        {"id": "no_gating", "label": "(F) w/o Gated Canny", "bg": (248, 250, 252)},
        {"id": "level2_object", "label": "Object Prompt (\"a <obj>\")", "bg": (238, 242, 255), "is_prompt": True},
        {"id": "level3_style_desc", "label": "Style Prompt (\"<obj> in <sty>\")", "bg": (238, 242, 255), "is_prompt": True},
    ]

    cell_size = 256
    pad = 5
    top_header = 45

    total_w = len(cols) * (cell_size + pad) + pad
    total_h = top_header + len(pairs) * (cell_size + pad) + pad

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    header_font = get_font(13, bold=True)

    # Headers
    for c_idx, col in enumerate(cols):
        x = c_idx * (cell_size + pad) + pad
        is_full = col.get("is_full", False)
        is_prompt = col.get("is_prompt", False)
        banner_w = cell_size
        banner_h = top_header - 10
        
        outline_color = (99, 102, 241) if is_full else ((129, 140, 248) if is_prompt else (203, 213, 225))
        outline_w = 2 if (is_full or is_prompt) else 1
        draw.rounded_rectangle([x, 5, x + banner_w, 5 + banner_h], radius=4, fill=col["bg"], outline=outline_color, width=outline_w)
        
        bbox = header_font.getbbox(col["label"])
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        text_color = (67, 56, 202) if (is_full or is_prompt) else (15, 23, 42)
        draw.text((x + (banner_w - tw)//2, 5 + (banner_h - th)//2 - 2), col["label"], font=header_font, fill=text_color)

    # Rows
    for r_idx, (c_id, s_id) in enumerate(pairs):
        y = top_header + r_idx * (cell_size + pad) + pad

        # Content
        c_path = f"{WORKSPACE}/data/content/{c_id}.png"
        if os.path.exists(c_path):
            img = Image.open(c_path).convert("RGB")
            img = make_square(img).resize((cell_size, cell_size), Image.Resampling.LANCZOS)
            x = 0 * (cell_size + pad) + pad
            canvas.paste(img, (x, y))
            draw.rectangle([x, y, x + cell_size, y + cell_size], outline=(203, 213, 225), width=1)

        # Style
        s_path = f"{WORKSPACE}/data/style/{s_id}.png"
        if os.path.exists(s_path):
            img = Image.open(s_path).convert("RGB")
            img = make_square(img).resize((cell_size, cell_size), Image.Resampling.LANCZOS)
            x = 1 * (cell_size + pad) + pad
            canvas.paste(img, (x, y))
            draw.rectangle([x, y, x + cell_size, y + cell_size], outline=(203, 213, 225), width=1)

        # Ablation outputs (Components A-F + Prompts Level 2 & 3)
        ablation_paths = {
            "full": f"{WORKSPACE}/output/benchmark/level1_null/ortho_{c_id}_{s_id}.png",
            "pure_mean": f"{WORKSPACE}/output/benchmark/ablation_B_pure_mean/level1_null/ortho_{c_id}_{s_id}.png",
            "raw_style": f"{WORKSPACE}/output/benchmark/ablation_C_raw_style/level1_null/ortho_{c_id}_{s_id}.png",
            "no_guidance": f"{WORKSPACE}/output/benchmark/ablation_D_no_guidance/level1_null/ortho_{c_id}_{s_id}.png",
            "no_pushforward": f"{WORKSPACE}/output/benchmark/ablation_E_no_pushforward/level1_null/ortho_{c_id}_{s_id}.png",
            "no_gating": f"{WORKSPACE}/output/benchmark/ablation_F_no_gating/level1_null/ortho_{c_id}_{s_id}.png",
            "level2_object": f"{WORKSPACE}/output/benchmark/level2_object/ortho_{c_id}_{s_id}.png",
            "level3_style_desc": f"{WORKSPACE}/output/benchmark/level3_style_desc/ortho_{c_id}_{s_id}.png",
        }

        for c_idx, col in enumerate(cols[2:], start=2):
            mid = col["id"]
            p = ablation_paths.get(mid)
            if p and os.path.exists(p):
                img = Image.open(p).convert("RGB")
                img = make_square(img).resize((cell_size, cell_size), Image.Resampling.LANCZOS)
                x = c_idx * (cell_size + pad) + pad
                canvas.paste(img, (x, y))
                if col.get("is_full"):
                    draw.rectangle([x, y, x + cell_size, y + cell_size], outline=(99, 102, 241), width=3)
                elif col.get("is_prompt"):
                    draw.rectangle([x, y, x + cell_size, y + cell_size], outline=(165, 180, 252), width=2)
                else:
                    draw.rectangle([x, y, x + cell_size, y + cell_size], outline=(226, 232, 240), width=1)

    save_figure(canvas, "ablation_components_cut")
    save_figure(canvas, "ablation_components_")

if __name__ == "__main__":
    t_start = time.time()
    generate_baselines_comparison_cut()
    generate_ablation_components_cut()
    print(f"\n=== Both Cut Figures Generated Successfully in {time.time()-t_start:.1f}s! ===")

