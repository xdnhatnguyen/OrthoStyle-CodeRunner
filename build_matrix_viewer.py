#!/usr/bin/env python3
"""
build_matrix_viewer.py
Generates an interactive, production-grade 15x15 Matrix Visualizer & Paper Curation Web Application.
"""

import os
import json

WORKSPACE_ROOT = os.path.abspath(os.path.dirname(__file__))

contents = [os.path.splitext(f)[0] for f in sorted(os.listdir(os.path.join(WORKSPACE_ROOT, 'data/content')))]
styles = [os.path.splitext(f)[0] for f in sorted(os.listdir(os.path.join(WORKSPACE_ROOT, 'data/style')))]

# Friendly human labels
def human_label(name):
    parts = name.split('_')[1:] # drop index like 01_
    return ' '.join(parts).title()

content_metadata = []
for c in contents:
    content_metadata.append({
        "id": c,
        "name": human_label(c),
        "file": f"data/content/{c}.png"
    })

style_metadata = []
for s in styles:
    style_metadata.append({
        "id": s,
        "name": human_label(s),
        "file": f"data/style/{s}.png"
    })

# Check ablation availability
ablations_def = [
    {"key": "ours", "label": "Full OrthoStyle (A)", "path_tpl": "output/benchmark/level1_null/ortho_{c}_{s}.png", "desc": "Default Proposed Framework (alpha_s=0.85, tau=1)"},
    {"key": "pure_mean", "label": "Pure Mean Token (B)", "path_tpl": "output/benchmark/ablation_B_pure_mean/level1_null/ortho_{c}_{s}.png", "desc": "alpha_s = 0.0 (Mean Pooling like RB-Modulation)"},
    {"key": "raw_style", "label": "Raw Style Token (C)", "path_tpl": "output/benchmark/ablation_C_raw_style/level1_null/ortho_{c}_{s}.png", "desc": "alpha_s = 1.0 (Unblended Raw Token)"},
    {"key": "no_guidance", "label": "w/o Score Guidance (D)", "path_tpl": "output/benchmark/ablation_D_no_guidance/level1_null/ortho_{c}_{s}.png", "desc": "Standard Sampling (No Guidance Gradients)"},
    {"key": "no_ortho", "label": "w/o Orthogonal Proj (D2)", "path_tpl": "output/benchmark/ablation_D2_no_ortho/level1_null/ortho_{c}_{s}.png", "desc": "Standard Gradient (No Score-Orthogonal Projection)"},
    {"key": "no_pushforward", "label": "w/o AdaIN Pushforward (E)", "path_tpl": "output/benchmark/ablation_E_no_pushforward/level1_null/ortho_{c}_{s}.png", "desc": "No Clean Latent AdaIN (tau=0)"},
    {"key": "no_gating", "label": "w/o Semantic Gating (F)", "path_tpl": "output/benchmark/ablation_F_no_gating/level1_null/ortho_{c}_{s}.png", "desc": "Raw Canny Edge Detection without rembg"},
    {"key": "sweep_tau_1", "label": "Sweep tau = 1", "path_tpl": "output/benchmark/sweep_tau_1/level1_null/ortho_{c}_{s}.png", "desc": "Early Pushforward tau = 1 (p_switch = 0.05)"},
    {"key": "sweep_tau_2", "label": "Sweep tau = 2", "path_tpl": "output/benchmark/sweep_tau_2/level1_null/ortho_{c}_{s}.png", "desc": "Early Pushforward tau = 2 (p_switch = 0.10)"},
    {"key": "sweep_tau_3", "label": "Sweep tau = 3", "path_tpl": "output/benchmark/sweep_tau_3/level1_null/ortho_{c}_{s}.png", "desc": "Early Pushforward tau = 3 (p_switch = 0.15)"},
    {"key": "sweep_tau_4", "label": "Sweep tau = 4", "path_tpl": "output/benchmark/sweep_tau_4/level1_null/ortho_{c}_{s}.png", "desc": "Early Pushforward tau = 4 (p_switch = 0.20)"},
]

baselines_def = [
    {"key": "ours", "label": "OrthoStyle (Ours)", "path_tpl": "output/benchmark/level1_null/ortho_{c}_{s}.png", "is_ours": True},
    {"key": "styleid", "label": "StyleID (CVPR'24)", "path_tpl": "baseline_results/outputs_styleid_soictdata/styleid_{c}_{s}.png", "is_ours": False},
    {"key": "attenst", "label": "AttenST (2024)", "path_tpl": "baseline_results/outputs_attenst_soictdata/attenst_{c}_{s}.png", "is_ours": False},
    {"key": "diffuseIT", "label": "DiffuseIT (CVPR'23)", "path_tpl": "baseline_results/outputs_diffuseIT_soictdata/DiffuseIT_{c}_{s}.png", "is_ours": False},
    {"key": "instantstyle", "label": "InstantStyle (2024)", "path_tpl": "baseline_results/outputs_instantstyle_soictdata/instantstyle_{c}_{s}.png", "is_ours": False},
    {"key": "itoc", "label": "ITOC (2024)", "path_tpl": "baseline_results/outputs_itoc_soictdata/itoc_{c}_{s}.png", "is_ours": False},
    {"key": "rb", "label": "RB-Modulation (2024)", "path_tpl": "baseline_results/outputs_rb_soictdata/outputs_rb_soictdata_{c}_{s}.png", "is_ours": False},
]

# Verify which files exist
grid_data = {}
for c in contents:
    grid_data[c] = {}
    for s in styles:
        grid_data[c][s] = {
            "ortho": f"output/benchmark/level1_null/ortho_{c}_{s}.png",
            "baselines": {},
            "ablations": {}
        }
        for b in baselines_def:
            p = b["path_tpl"].format(c=c, s=s)
            if os.path.exists(os.path.join(WORKSPACE_ROOT, p)):
                grid_data[c][s]["baselines"][b["key"]] = p

        for a in ablations_def:
            p = a["path_tpl"].format(c=c, s=s)
            if os.path.exists(os.path.join(WORKSPACE_ROOT, p)):
                grid_data[c][s]["ablations"][a["key"]] = p

print("Loaded grid data for 15x15 matrix.")

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>OrthoStyle: 15x15 Benchmark Matrix Visualizer & Paper Curation Tool</title>
  <style>
    :root {{
      --bg-primary: #0a0d14;
      --bg-secondary: #121826;
      --bg-card: rgba(22, 30, 46, 0.85);
      --bg-glass: rgba(18, 24, 38, 0.75);
      --border-color: rgba(255, 255, 255, 0.08);
      --border-accent: rgba(99, 102, 241, 0.4);
      --text-primary: #f1f5f9;
      --text-secondary: #94a3b8;
      --text-muted: #64748b;
      --accent-indigo: #6366f1;
      --accent-purple: #a855f7;
      --accent-emerald: #10b981;
      --accent-amber: #f59e0b;
      --accent-rose: #f43f5e;
      --gold-gradient: linear-gradient(135deg, #f59e0b, #ec4899);
      --cell-size: 110px;
    }}

    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}

    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Inter', sans-serif;
      background-color: var(--bg-primary);
      color: var(--text-primary);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      overflow-x: auto;
    }}

    /* Top Navigation Header */
    header {{
      position: sticky;
      top: 0;
      left: 0;
      right: 0;
      z-index: 1000;
      background: var(--bg-glass);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border-bottom: 1px solid var(--border-color);
      padding: 12px 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 20px;
    }}

    .brand-area {{
      display: flex;
      align-items: center;
      gap: 14px;
    }}

    .logo-badge {{
      background: linear-gradient(135deg, #4f46e5, #9333ea);
      color: white;
      font-weight: 800;
      font-size: 14px;
      padding: 6px 12px;
      border-radius: 8px;
      letter-spacing: 0.5px;
      box-shadow: 0 4px 12px rgba(79, 70, 229, 0.4);
    }}

    .brand-titles h1 {{
      font-size: 18px;
      font-weight: 700;
      color: var(--text-primary);
      display: flex;
      align-items: center;
      gap: 8px;
    }}

    .brand-titles p {{
      font-size: 12px;
      color: var(--text-secondary);
    }}

    .controls-bar {{
      display: flex;
      align-items: center;
      gap: 16px;
      flex-wrap: wrap;
    }}

    .control-group {{
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 13px;
      color: var(--text-secondary);
    }}

    .control-group label {{
      font-weight: 500;
    }}

    input[type="range"] {{
      accent-color: var(--accent-indigo);
      cursor: pointer;
    }}

    .btn {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 7px 14px;
      border-radius: 8px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s ease;
      border: 1px solid transparent;
    }}

    .btn-primary {{
      background: linear-gradient(135deg, #4f46e5, #7c3aed);
      color: white;
      box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
    }}
    .btn-primary:hover {{
      opacity: 0.92;
      transform: translateY(-1px);
    }}

    .btn-secondary {{
      background: rgba(255, 255, 255, 0.06);
      color: var(--text-primary);
      border-color: var(--border-color);
    }}
    .btn-secondary:hover {{
      background: rgba(255, 255, 255, 0.12);
    }}

    .btn-success {{
      background: linear-gradient(135deg, #059669, #10b981);
      color: white;
    }}

    .curated-badge {{
      background: rgba(99, 102, 241, 0.15);
      border: 1px solid var(--accent-indigo);
      color: #c7d2fe;
      padding: 4px 10px;
      border-radius: 20px;
      font-size: 12px;
      font-weight: 600;
      display: flex;
      align-items: center;
      gap: 6px;
      cursor: pointer;
    }}

    /* Main Container with 15x15 Matrix */
    main {{
      flex: 1;
      padding: 20px;
      display: flex;
      justify-content: center;
    }}

    .matrix-wrapper {{
      display: inline-block;
      background: var(--bg-secondary);
      border: 1px solid var(--border-color);
      border-radius: 12px;
      box-shadow: 0 20px 40px rgba(0, 0, 0, 0.6);
      overflow: auto;
      max-width: 98vw;
      max-height: calc(100vh - 100px);
      position: relative;
    }}

    table.matrix-table {{
      border-collapse: separate;
      border-spacing: 0;
      width: max-content;
    }}

    /* Top Left Corner Diagonal Header */
    .corner-header {{
      position: sticky;
      top: 0;
      left: 0;
      z-index: 50;
      width: var(--cell-size);
      height: var(--cell-size);
      background: #1e293b;
      border-right: 2px solid rgba(255, 255, 255, 0.15);
      border-bottom: 2px solid rgba(255, 255, 255, 0.15);
      padding: 4px;
      box-sizing: border-box;
    }}

    .diagonal-box {{
      width: 100%;
      height: 100%;
      position: relative;
      background: linear-gradient(to top right, #1e293b 49.5%, rgba(255,255,255,0.2) 50%, #0f172a 50.5%);
      border-radius: 6px;
      overflow: hidden;
    }}

    .diagonal-style {{
      position: absolute;
      top: 6px;
      right: 8px;
      font-size: 11px;
      font-weight: 700;
      color: #38bdf8;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }}

    .diagonal-content {{
      position: absolute;
      bottom: 6px;
      left: 8px;
      font-size: 11px;
      font-weight: 700;
      color: #fbbf24;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }}

    /* Column Headers (Styles) */
    th.style-header {{
      position: sticky;
      top: 0;
      z-index: 40;
      background: #1e293b;
      border-bottom: 2px solid rgba(255, 255, 255, 0.15);
      border-right: 1px solid var(--border-color);
      width: var(--cell-size);
      min-width: var(--cell-size);
      padding: 6px;
      text-align: center;
      vertical-align: top;
    }}

    .header-thumb-card {{
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 4px;
    }}

    .header-thumb-card img {{
      width: calc(var(--cell-size) - 16px);
      height: calc(var(--cell-size) - 34px);
      object-fit: cover;
      border-radius: 6px;
      border: 1px solid rgba(255, 255, 255, 0.2);
      box-shadow: 0 2px 6px rgba(0, 0, 0, 0.4);
    }}

    .header-thumb-card .label {{
      font-size: 10px;
      font-weight: 600;
      color: #93c5fd;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      max-width: calc(var(--cell-size) - 10px);
    }}

    /* Row Headers (Contents) */
    th.content-header {{
      position: sticky;
      left: 0;
      z-index: 40;
      background: #1e293b;
      border-right: 2px solid rgba(255, 255, 255, 0.15);
      border-bottom: 1px solid var(--border-color);
      width: var(--cell-size);
      min-width: var(--cell-size);
      padding: 6px;
      text-align: center;
      vertical-align: middle;
    }}

    .content-thumb-card {{
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 4px;
    }}

    .content-thumb-card img {{
      width: calc(var(--cell-size) - 16px);
      height: calc(var(--cell-size) - 34px);
      object-fit: cover;
      border-radius: 6px;
      border: 1px solid rgba(255, 255, 255, 0.2);
      box-shadow: 0 2px 6px rgba(0, 0, 0, 0.4);
    }}

    .content-thumb-card .label {{
      font-size: 10px;
      font-weight: 600;
      color: #fde68a;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      max-width: calc(var(--cell-size) - 10px);
    }}

    /* Grid Output Cells */
    td.matrix-cell {{
      width: var(--cell-size);
      height: var(--cell-size);
      min-width: var(--cell-size);
      min-height: var(--cell-size);
      padding: 2px;
      border-right: 1px solid var(--border-color);
      border-bottom: 1px solid var(--border-color);
      position: relative;
      cursor: pointer;
      background: #0d131f;
      transition: all 0.15s ease;
    }}

    td.matrix-cell:hover {{
      background: #1a2336;
      transform: scale(1.04);
      z-index: 30;
      box-shadow: 0 8px 20px rgba(0, 0, 0, 0.8), 0 0 0 2px var(--accent-indigo);
      border-radius: 4px;
    }}

    td.matrix-cell.selected-comp {{
      box-shadow: inset 0 0 0 2px #38bdf8, 0 0 12px rgba(56, 189, 248, 0.5);
    }}

    td.matrix-cell.selected-abla {{
      box-shadow: inset 0 0 0 2px #f43f5e, 0 0 12px rgba(244, 63, 94, 0.5);
    }}

    td.matrix-cell.selected-both {{
      box-shadow: inset 0 0 0 2px #a855f7, 0 0 12px rgba(168, 85, 247, 0.6);
    }}

    .cell-container {{
      width: 100%;
      height: 100%;
      position: relative;
      overflow: hidden;
      border-radius: 4px;
    }}

    .cell-container img {{
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
      transition: transform 0.2s ease;
    }}

    .cell-hover-overlay {{
      position: absolute;
      inset: 0;
      background: rgba(10, 13, 20, 0.7);
      opacity: 0;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 6px;
      transition: opacity 0.15s ease;
      padding: 4px;
    }}

    td.matrix-cell:hover .cell-hover-overlay {{
      opacity: 1;
    }}

    .overlay-btn {{
      background: rgba(255, 255, 255, 0.15);
      backdrop-filter: blur(4px);
      border: 1px solid rgba(255, 255, 255, 0.25);
      color: white;
      font-size: 10px;
      font-weight: 600;
      padding: 3px 8px;
      border-radius: 4px;
      display: flex;
      align-items: center;
      gap: 4px;
    }}

    .tag-ours {{
      position: absolute;
      top: 3px;
      left: 3px;
      background: rgba(99, 102, 241, 0.85);
      color: white;
      font-size: 8px;
      font-weight: 700;
      padding: 1px 4px;
      border-radius: 3px;
      letter-spacing: 0.3px;
      pointer-events: none;
    }}

    .badge-star {{
      position: absolute;
      top: 3px;
      right: 3px;
      font-size: 12px;
      color: #fbbf24;
      display: none;
      filter: drop-shadow(0 1px 3px rgba(0,0,0,0.8));
    }}

    td.matrix-cell.selected-comp .badge-star,
    td.matrix-cell.selected-abla .badge-star,
    td.matrix-cell.selected-both .badge-star {{
      display: block;
    }}

    /* Modal Styles */
    .modal-backdrop {{
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.85);
      backdrop-filter: blur(8px);
      z-index: 2000;
      display: none;
      align-items: center;
      justify-content: center;
      padding: 20px;
      overflow-y: auto;
    }}

    .modal-backdrop.active {{
      display: flex;
    }}

    .modal-window {{
      background: var(--bg-secondary);
      border: 1px solid var(--border-color);
      border-radius: 16px;
      width: 100%;
      max-width: 1240px;
      max-height: 92vh;
      display: flex;
      flex-direction: column;
      box-shadow: 0 25px 60px rgba(0, 0, 0, 0.9);
      overflow: hidden;
      animation: modalIn 0.2s ease-out;
    }}

    @keyframes modalIn {{
      from {{ transform: scale(0.96); opacity: 0; }}
      to {{ transform: scale(1); opacity: 1; }}
    }}

    .modal-header {{
      padding: 16px 24px;
      background: rgba(255, 255, 255, 0.03);
      border-bottom: 1px solid var(--border-color);
      display: flex;
      align-items: center;
      justify-content: space-between;
    }}

    .modal-title-area {{
      display: flex;
      align-items: center;
      gap: 16px;
    }}

    .modal-title-area h2 {{
      font-size: 18px;
      font-weight: 700;
    }}

    .modal-title-area .sub {{
      font-size: 12px;
      color: var(--text-secondary);
    }}

    .modal-tabs {{
      display: flex;
      gap: 4px;
      background: rgba(0, 0, 0, 0.4);
      padding: 4px;
      border-radius: 8px;
      border: 1px solid var(--border-color);
    }}

    .modal-tab-btn {{
      padding: 6px 14px;
      font-size: 12px;
      font-weight: 600;
      color: var(--text-secondary);
      background: transparent;
      border: none;
      border-radius: 6px;
      cursor: pointer;
      transition: all 0.15s ease;
    }}

    .modal-tab-btn.active {{
      background: var(--accent-indigo);
      color: white;
    }}

    .btn-close {{
      background: rgba(255, 255, 255, 0.08);
      border: none;
      color: var(--text-secondary);
      font-size: 20px;
      width: 32px;
      height: 32px;
      border-radius: 8px;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
    }}
    .btn-close:hover {{
      background: rgba(255, 255, 255, 0.2);
      color: white;
    }}

    .modal-body {{
      padding: 20px;
      overflow-y: auto;
      flex: 1;
      display: flex;
      flex-direction: column;
      gap: 20px;
    }}

    /* Reference Card Strip */
    .reference-banner {{
      display: flex;
      align-items: center;
      gap: 16px;
      background: rgba(0, 0, 0, 0.3);
      padding: 12px 16px;
      border-radius: 10px;
      border: 1px solid var(--border-color);
    }}

    .ref-item {{
      display: flex;
      align-items: center;
      gap: 10px;
    }}

    .ref-item img {{
      width: 54px;
      height: 54px;
      object-fit: cover;
      border-radius: 6px;
      border: 1px solid rgba(255, 255, 255, 0.2);
    }}

    .ref-info {{
      display: flex;
      flex-direction: column;
    }}

    .ref-info .type {{
      font-size: 11px;
      text-transform: uppercase;
      font-weight: 700;
      color: var(--text-muted);
    }}

    .ref-info .name {{
      font-size: 13px;
      font-weight: 600;
      color: var(--text-primary);
    }}

    /* Comparison Grid in Modal */
    .compare-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(210px, 1fr));
      gap: 16px;
    }}

    .compare-card {{
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 10px;
      padding: 10px;
      display: flex;
      flex-direction: column;
      gap: 8px;
      position: relative;
      transition: all 0.15s ease;
    }}

    .compare-card:hover {{
      border-color: rgba(255, 255, 255, 0.25);
    }}

    .compare-card.is-ours {{
      border-color: var(--accent-indigo);
      box-shadow: 0 0 16px rgba(99, 102, 241, 0.25);
    }}

    .compare-card-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
    }}

    .compare-card-header h4 {{
      font-size: 13px;
      font-weight: 700;
      color: var(--text-primary);
    }}

    .compare-card-header .badge {{
      font-size: 9px;
      padding: 2px 6px;
      border-radius: 4px;
      font-weight: 700;
      text-transform: uppercase;
    }}

    .badge-ours {{
      background: var(--accent-indigo);
      color: white;
    }}

    .badge-baseline {{
      background: rgba(255, 255, 255, 0.1);
      color: var(--text-secondary);
    }}

    .compare-card img {{
      width: 100%;
      aspect-ratio: 1;
      object-fit: cover;
      border-radius: 6px;
      border: 1px solid rgba(255, 255, 255, 0.1);
    }}

    .card-checkbox-row {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-size: 12px;
      color: var(--text-secondary);
      padding-top: 4px;
    }}

    .card-checkbox-row label {{
      display: flex;
      align-items: center;
      gap: 6px;
      cursor: pointer;
    }}

    .modal-footer {{
      padding: 16px 24px;
      background: rgba(0, 0, 0, 0.4);
      border-top: 1px solid var(--border-color);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }}

    .footer-left {{
      display: flex;
      align-items: center;
      gap: 12px;
    }}

    /* Drawer / Side panel for Curation */
    .drawer {{
      position: fixed;
      right: -420px;
      top: 60px;
      bottom: 0;
      width: 400px;
      background: var(--bg-secondary);
      border-left: 1px solid var(--border-color);
      z-index: 1500;
      box-shadow: -10px 0 30px rgba(0, 0, 0, 0.8);
      transition: right 0.3s cubic-bezier(0.16, 1, 0.3, 1);
      display: flex;
      flex-direction: column;
    }}

    .drawer.active {{
      right: 0;
    }}

    .drawer-header {{
      padding: 16px 20px;
      border-bottom: 1px solid var(--border-color);
      display: flex;
      align-items: center;
      justify-content: space-between;
    }}

    .drawer-header h3 {{
      font-size: 16px;
      font-weight: 700;
    }}

    .drawer-body {{
      padding: 16px;
      overflow-y: auto;
      flex: 1;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }}

    .curated-item-card {{
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 10px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }}

    .curated-item-thumbs {{
      display: flex;
      align-items: center;
      gap: 6px;
    }}

    .curated-item-thumbs img {{
      width: 42px;
      height: 42px;
      object-fit: cover;
      border-radius: 4px;
    }}

    .curated-item-info {{
      flex: 1;
    }}

    .curated-item-info h5 {{
      font-size: 12px;
      font-weight: 600;
    }}

    .curated-item-info p {{
      font-size: 11px;
      color: var(--text-muted);
    }}

    .drawer-footer {{
      padding: 16px 20px;
      border-top: 1px solid var(--border-color);
      display: flex;
      flex-direction: column;
      gap: 10px;
    }}

    /* Toast Notification */
    .toast {{
      position: fixed;
      bottom: 24px;
      left: 50%;
      transform: translateX(-50%) translateY(100px);
      background: #1e293b;
      border: 1px solid var(--accent-indigo);
      color: white;
      padding: 10px 20px;
      border-radius: 30px;
      font-size: 13px;
      font-weight: 600;
      box-shadow: 0 10px 25px rgba(0, 0, 0, 0.6);
      z-index: 3000;
      transition: transform 0.25s ease, opacity 0.25s ease;
      opacity: 0;
      pointer-events: none;
    }}

    .toast.show {{
      transform: translateX(-50%) translateY(0);
      opacity: 1;
    }}
  </style>
</head>
<body>

  <!-- Top Navigation Bar -->
  <header>
    <div class="brand-area">
      <div class="logo-badge">ORTHOSTYLE</div>
      <div class="brand-titles">
        <h1>15&times;15 Benchmark Matrix Visualizer</h1>
        <p>SOICT-Data (225 Content-Style Pairs) &bull; Baseline & Ablation Curation Suite</p>
      </div>
    </div>

    <div class="controls-bar">
      <div class="control-group">
        <label for="cellZoom">Zoom:</label>
        <input type="range" id="cellZoom" min="70" max="180" value="110" oninput="updateCellSize(this.value)">
      </div>

      <button class="btn btn-secondary" onclick="exportFull15x15Teaser()">
        &#128396; Export Full 15x15 Matrix (PNG)
      </button>

      <div class="curated-badge" onclick="toggleDrawer()">
        &#9733; Curated for Paper: <span id="curatedCount">0</span>
      </div>
    </div>
  </header>

  <!-- Matrix Table View -->
  <main>
    <div class="matrix-wrapper">
      <table class="matrix-table" id="matrixTable">
        <thead>
          <tr id="tableHeaderRow">
            <!-- Top-Left Corner Header -->
            <th class="corner-header">
              <div class="diagonal-box">
                <span class="diagonal-style">Style &rarr;</span>
                <span class="diagonal-content">&darr; Content</span>
              </div>
            </th>
            <!-- Style Column Headers (15 items) will be generated by JS -->
          </tr>
        </thead>
        <tbody id="tableBody">
          <!-- Content Rows (15 items) and Output Cells (225 items) will be generated by JS -->
        </tbody>
      </table>
    </div>
  </main>

  <!-- Inspection & Comparison Modal -->
  <div class="modal-backdrop" id="modalBackdrop" onclick="closeModalOnBackdrop(event)">
    <div class="modal-window" onclick="event.stopPropagation()">
      <div class="modal-header">
        <div class="modal-title-area">
          <h2 id="modalTitle">Pair Inspection</h2>
          <span class="sub" id="modalSubTitle"></span>
        </div>
        <div class="modal-tabs">
          <button class="modal-tab-btn active" id="tabBtnBaselines" onclick="switchModalTab('baselines')">
            Baselines Comparison (7)
          </button>
          <button class="modal-tab-btn" id="tabBtnAblations" onclick="switchModalTab('ablations')">
            Ablation Settings (11)
          </button>
        </div>
        <button class="btn-close" onclick="closeModal()">&times;</button>
      </div>

      <div class="modal-body">
        <!-- Content and Style Exemplar Banner -->
        <div class="reference-banner">
          <div class="ref-item">
            <img id="modalContentImg" src="" alt="Content Reference">
            <div class="ref-info">
              <span class="type">Content Input</span>
              <span class="name" id="modalContentName"></span>
            </div>
          </div>
          <div style="color: var(--text-muted); font-size: 20px;">&times;</div>
          <div class="ref-item">
            <img id="modalStyleImg" src="" alt="Style Reference">
            <div class="ref-info">
              <span class="type">Style Reference</span>
              <span class="name" id="modalStyleName"></span>
            </div>
          </div>
          <div style="margin-left: auto;">
            <button class="btn btn-secondary" id="btnToggleCurate" onclick="toggleCurrentPairCuration()">
              &#9733; Mark for Paper
            </button>
          </div>
        </div>

        <!-- Tab 1: Baselines View -->
        <div id="tabContentBaselines">
          <div class="compare-grid" id="baselinesGrid"></div>
        </div>

        <!-- Tab 2: Ablations View -->
        <div id="tabContentAblations" style="display: none;">
          <div class="compare-grid" id="ablationsGrid"></div>
        </div>
      </div>

      <div class="modal-footer">
        <div class="footer-left">
          <span style="font-size: 12px; color: var(--text-secondary);">Select cards above to include in comparison image</span>
        </div>
        <div style="display: flex; gap: 10px;">
          <button class="btn btn-primary" onclick="saveComparisonComposite()">
            &#128444; Save Comparison Strip (PNG)
          </button>
          <button class="btn btn-secondary" onclick="saveAblationComposite()">
            &#128300; Save Ablation Strip (PNG)
          </button>
        </div>
      </div>
    </div>
  </div>

  <!-- Curation Drawer (Sidebar) -->
  <div class="drawer" id="curationDrawer">
    <div class="drawer-header">
      <h3>&#9733; Curated Paper Candidates</h3>
      <button class="btn-close" onclick="toggleDrawer()">&times;</button>
    </div>
    <div class="drawer-body" id="drawerBody">
      <!-- Items will be populated by JS -->
    </div>
    <div class="drawer-footer">
      <button class="btn btn-primary" onclick="exportCurationJSON()">
        Export Selection JSON
      </button>
      <button class="btn btn-secondary" onclick="exportCurationCSV()">
        Export Selection CSV
      </button>
      <button class="btn btn-secondary" style="color: #f87171;" onclick="clearAllCuration()">
        Clear Selection
      </button>
    </div>
  </div>

  <!-- Toast Element -->
  <div class="toast" id="toastMessage">Action completed</div>

  <!-- Hidden Canvas for Composite Image Export -->
  <canvas id="exportCanvas" style="display: none;"></canvas>

  <script>
    // Embedded Data from Server / Workspace
    const contents = {json.dumps(content_metadata)};
    const styles = {json.dumps(style_metadata)};
    const baselinesDef = {json.dumps(baselines_def)};
    const ablationsDef = {json.dumps(ablations_def)};
    const gridData = {json.dumps(grid_data)};

    let currentActivePair = {{ contentId: null, styleId: null }};
    let curatedPairs = JSON.parse(localStorage.getItem('orthostyle_curated_pairs') || '[]');

    // Initialize DOM
    document.addEventListener('DOMContentLoaded', () => {{
      renderMatrix();
      updateCuratedCount();
    }});

    function updateCellSize(val) {{
      document.documentElement.style.setProperty('--cell-size', val + 'px');
    }}

    function showToast(msg) {{
      const toast = document.getElementById('toastMessage');
      toast.innerText = msg;
      toast.classList.add('show');
      setTimeout(() => toast.classList.remove('show'), 2500);
    }}

    // Render 15x15 Matrix
    function renderMatrix() {{
      const headerRow = document.getElementById('tableHeaderRow');
      const tableBody = document.getElementById('tableBody');

      // Clear existing except corner
      while (headerRow.children.length > 1) {{
        headerRow.removeChild(headerRow.lastChild);
      }}
      tableBody.innerHTML = '';

      // Render 15 Style Headers
      styles.forEach(s => {{
        const th = document.createElement('th');
        th.className = 'style-header';
        th.title = s.name;
        th.innerHTML = `
          <div class="header-thumb-card">
            <img src="${{s.file}}" alt="${{s.name}}" loading="lazy">
            <span class="label">${{s.name}}</span>
          </div>
        `;
        headerRow.appendChild(th);
      }});

      // Render 15 Content Rows and 225 Cells
      contents.forEach(c => {{
        const tr = document.createElement('tr');

        // Content Row Header
        const th = document.createElement('th');
        th.className = 'content-header';
        th.title = c.name;
        th.innerHTML = `
          <div class="content-thumb-card">
            <img src="${{c.file}}" alt="${{c.name}}" loading="lazy">
            <span class="label">${{c.name}}</span>
          </div>
        `;
        tr.appendChild(th);

        // 15 Output Cells
        styles.forEach(s => {{
          const td = document.createElement('td');
          td.className = 'matrix-cell';
          td.id = `cell_${{c.id}}_${{s.id}}`;
          
          const pairKey = `${{c.id}}_${{s.id}}`;
          if (curatedPairs.includes(pairKey)) {{
            td.classList.add('selected-comp');
          }}

          const cellData = gridData[c.id][s.id];
          const imgSrc = cellData.ortho;

          td.innerHTML = `
            <div class="cell-container">
              <span class="tag-ours">OURS</span>
              <span class="badge-star">&#9733;</span>
              <img src="${{imgSrc}}" alt="${{c.name}} - ${{s.name}}" loading="lazy">
              <div class="cell-hover-overlay">
                <span class="overlay-btn">&#128065; Inspect</span>
              </div>
            </div>
          `;

          td.onclick = () => openModal(c.id, s.id);
          tr.appendChild(td);
        }});

        tableBody.appendChild(tr);
      }});
    }}

    // Open Inspection Modal
    function openModal(contentId, styleId) {{
      currentActivePair = {{ contentId, styleId }};
      const content = contents.find(c => c.id === contentId);
      const style = styles.find(s => s.id === styleId);

      document.getElementById('modalTitle').innerText = `${{content.name}} &times; ${{style.name}}`;
      document.getElementById('modalSubTitle').innerText = `ID: ${{contentId}} / ${{styleId}}`;
      document.getElementById('modalContentImg').src = content.file;
      document.getElementById('modalContentName').innerText = content.name;
      document.getElementById('modalStyleImg').src = style.file;
      document.getElementById('modalStyleName').innerText = style.name;

      updateModalCurateBtn();
      renderBaselinesGrid(contentId, styleId);
      renderAblationsGrid(contentId, styleId);

      document.getElementById('modalBackdrop').classList.add('active');
    }}

    function closeModal() {{
      document.getElementById('modalBackdrop').classList.remove('active');
    }}

    function closeModalOnBackdrop(e) {{
      if (e.target === document.getElementById('modalBackdrop')) {{
        closeModal();
      }}
    }}

    function switchModalTab(tabName) {{
      if (tabName === 'baselines') {{
        document.getElementById('tabBtnBaselines').classList.add('active');
        document.getElementById('tabBtnAblations').classList.remove('active');
        document.getElementById('tabContentBaselines').style.display = 'block';
        document.getElementById('tabContentAblations').style.display = 'none';
      }} else {{
        document.getElementById('tabBtnBaselines').classList.remove('active');
        document.getElementById('tabBtnAblations').classList.add('active');
        document.getElementById('tabContentBaselines').style.display = 'none';
        document.getElementById('tabContentAblations').style.display = 'block';
      }}
    }}

    // Render Baselines inside Modal
    function renderBaselinesGrid(contentId, styleId) {{
      const container = document.getElementById('baselinesGrid');
      container.innerHTML = '';
      const pair = gridData[contentId][styleId];

      baselinesDef.forEach((b, idx) => {{
        const card = document.createElement('div');
        card.className = `compare-card ${{b.is_ours ? 'is-ours' : ''}}`;
        const imgPath = pair.baselines[b.key];

        if (imgPath) {{
          card.innerHTML = `
            <div class="compare-card-header">
              <h4>${{b.label}}</h4>
              <span class="badge ${{b.is_ours ? 'badge-ours' : 'badge-baseline'}}">
                ${{b.is_ours ? 'Proposed' : 'Baseline'}}
              </span>
            </div>
            <img src="${{imgPath}}" alt="${{b.label}}" loading="lazy">
            <div class="card-checkbox-row">
              <label>
                <input type="checkbox" checked class="comp-check" data-name="${{b.label}}" data-src="${{imgPath}}">
                Include in Strip
              </label>
            </div>
          `;
          container.appendChild(card);
        }}
      }});
    }}

    // Render Ablations inside Modal
    function renderAblationsGrid(contentId, styleId) {{
      const container = document.getElementById('ablationsGrid');
      container.innerHTML = '';
      const pair = gridData[contentId][styleId];

      ablationsDef.forEach((a, idx) => {{
        const card = document.createElement('div');
        card.className = `compare-card ${{a.key === 'ours' ? 'is-ours' : ''}}`;
        const imgPath = pair.ablations[a.key];

        if (imgPath) {{
          card.innerHTML = `
            <div class="compare-card-header">
              <h4>${{a.label}}</h4>
              <span class="badge ${{a.key === 'ours' ? 'badge-ours' : 'badge-baseline'}}">
                ${{a.key === 'ours' ? 'Full' : 'Setting'}}
              </span>
            </div>
            <img src="${{imgPath}}" alt="${{a.label}}" loading="lazy">
            <p style="font-size: 11px; color: var(--text-muted);">${{a.desc}}</p>
            <div class="card-checkbox-row">
              <label>
                <input type="checkbox" checked class="abla-check" data-name="${{a.label}}" data-src="${{imgPath}}">
                Include in Strip
              </label>
            </div>
          `;
          container.appendChild(card);
        }}
      }});
    }}

    // Curation Management
    function toggleCurrentPairCuration() {{
      const key = `${{currentActivePair.contentId}}_${{currentActivePair.styleId}}`;
      const idx = curatedPairs.indexOf(key);
      const cell = document.getElementById(`cell_${{key}}`);

      if (idx > -1) {{
        curatedPairs.splice(idx, 1);
        if (cell) cell.classList.remove('selected-comp');
        showToast('Removed from Paper Curation');
      }} else {{
        curatedPairs.push(key);
        if (cell) cell.classList.add('selected-comp');
        showToast('Saved to Paper Curation list!');
      }}

      localStorage.setItem('orthostyle_curated_pairs', JSON.stringify(curatedPairs));
      updateModalCurateBtn();
      updateCuratedCount();
      renderDrawer();
    }}

    function updateModalCurateBtn() {{
      const key = `${{currentActivePair.contentId}}_${{currentActivePair.styleId}}`;
      const btn = document.getElementById('btnToggleCurate');
      if (curatedPairs.includes(key)) {{
        btn.innerHTML = '&#9733; In Paper Curation (Click to Remove)';
        btn.classList.remove('btn-secondary');
        btn.classList.add('btn-primary');
      }} else {{
        btn.innerHTML = '&#9734; Add to Paper Curation';
        btn.classList.remove('btn-primary');
        btn.classList.add('btn-secondary');
      }}
    }}

    function updateCuratedCount() {{
      document.getElementById('curatedCount').innerText = curatedPairs.length;
    }}

    function toggleDrawer() {{
      const drawer = document.getElementById('curationDrawer');
      drawer.classList.toggle('active');
      if (drawer.classList.contains('active')) {{
        renderDrawer();
      }}
    }}

    function renderDrawer() {{
      const body = document.getElementById('drawerBody');
      body.innerHTML = '';

      if (curatedPairs.length === 0) {{
        body.innerHTML = '<p style="font-size: 13px; color: var(--text-muted); text-align: center; margin-top: 40px;">No pairs curated yet. Click on any cell in the 15x15 matrix to inspect and mark candidates.</p>';
        return;
      }}

      curatedPairs.forEach(key => {{
        const [cId, sId] = key.split('_').length === 4 ? 
          [`${{key.split('_')[0]}}_${{key.split('_')[1]}}`, `${{key.split('_')[2]}}_${{key.split('_')[3]}}`] :
          [key.split('_')[0], key.split('_')[1]]; // fallback
        
        // Find exact matching content and style
        const c = contents.find(item => key.startsWith(item.id));
        const s = styles.find(item => key.endsWith(item.id));

        if (!c || !s) return;

        const card = document.createElement('div');
        card.className = 'curated-item-card';
        card.innerHTML = `
          <div class="curated-item-thumbs">
            <img src="${{c.file}}" title="Content: ${{c.name}}">
            <img src="${{s.file}}" title="Style: ${{s.name}}">
            <img src="output/benchmark/level1_null/ortho_${{c.id}}_${{s.id}}.png" title="OrthoStyle Output" style="border: 1px solid var(--accent-indigo);">
          </div>
          <div class="curated-item-info">
            <h5>${{c.name}} &times; ${{s.name}}</h5>
            <p>${{c.id}} / ${{s.id}}</p>
          </div>
          <button class="btn btn-secondary" style="padding: 4px 8px; font-size: 11px;" onclick="openModal('${{c.id}}', '${{s.id}}')">
            Inspect
          </button>
        `;
        body.appendChild(card);
      }});
    }}

    function clearAllCuration() {{
      if (confirm('Clear all curated candidates?')) {{
        curatedPairs = [];
        localStorage.removeItem('orthostyle_curated_pairs');
        updateCuratedCount();
        renderDrawer();
        document.querySelectorAll('.matrix-cell').forEach(c => c.classList.remove('selected-comp'));
        showToast('Curated list cleared');
      }}
    }}

    function exportCurationJSON() {{
      const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(curatedPairs, null, 2));
      const a = document.createElement('a');
      a.href = dataStr;
      a.download = 'curated_paper_candidates.json';
      a.click();
      showToast('Exported curated_paper_candidates.json');
    }}

    function exportCurationCSV() {{
      let csv = 'Content_ID,Content_Name,Style_ID,Style_Name,Output_Path\\n';
      curatedPairs.forEach(key => {{
        const c = contents.find(item => key.startsWith(item.id));
        const s = styles.find(item => key.endsWith(item.id));
        if (c && s) {{
          csv += `${{c.id}},"${{c.name}}",${{s.id}},"${{s.name}}",output/benchmark/level1_null/ortho_${{c.id}}_${{s.id}}.png\\n`;
        }}
      }});
      const dataStr = 'data:text/csv;charset=utf-8,' + encodeURIComponent(csv);
      const a = document.createElement('a');
      a.href = dataStr;
      a.download = 'curated_paper_candidates.csv';
      a.click();
      showToast('Exported curated_paper_candidates.csv');
    }}

    // ==========================================
    // Canvas Image Compositing & Saving Engine
    // ==========================================
    async function loadImage(src) {{
      return new Promise((resolve, reject) => {{
        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload = () => resolve(img);
        img.onerror = () => resolve(null); // safely skip broken
        img.src = src;
      }});
    }}

    // Save Comparison Strip (Content | Style | OrthoStyle | Baselines)
    async function saveComparisonComposite() {{
      showToast('Generating comparison strip...');
      const checks = Array.from(document.querySelectorAll('.comp-check:checked'));
      if (checks.length === 0) {{
        alert('Please check at least one model card.');
        return;
      }}

      const content = contents.find(c => c.id === currentActivePair.contentId);
      const style = styles.find(s => s.id === currentActivePair.styleId);

      const items = [
        {{ label: 'Content', src: content.file }},
        {{ label: 'Style Ref', src: style.file }},
        ...checks.map(c => ({{ label: c.dataset.name, src: c.dataset.src }}))
      ];

      const loadedImages = await Promise.all(items.map(async item => ({{
        label: item.label,
        img: await loadImage(item.src)
      }})));

      const validItems = loadedImages.filter(i => i.img !== null);
      if (validItems.length === 0) {{
        alert('Error loading images for canvas.');
        return;
      }}

      const itemW = 512;
      const itemH = 512;
      const headerH = 48;
      const totalW = validItems.length * itemW;
      const totalH = itemH + headerH;

      const canvas = document.getElementById('exportCanvas');
      canvas.width = totalW;
      canvas.height = totalH;
      const ctx = canvas.getContext('2d');

      // Background
      ctx.fillStyle = '#0f172a';
      ctx.fillRect(0, 0, totalW, totalH);

      // Draw columns
      validItems.forEach((item, idx) => {{
        const x = idx * itemW;
        
        // Header Text
        ctx.fillStyle = item.label.includes('Ours') ? '#6366f1' : '#f1f5f9';
        ctx.font = 'bold 22px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(item.label, x + itemW / 2, 32);

        // Image
        ctx.drawImage(item.img, x, headerH, itemW, itemH);

        // Highlight border for ours
        if (item.label.includes('Ours')) {{
          ctx.strokeStyle = '#6366f1';
          ctx.lineWidth = 6;
          ctx.strokeRect(x + 3, headerH + 3, itemW - 6, itemH - 6);
        }}
      }});

      // Export
      const a = document.createElement('a');
      a.download = `comparison_${{currentActivePair.contentId}}_${{currentActivePair.styleId}}.png`;
      a.href = canvas.toDataURL('image/png');
      a.click();
      showToast('Comparison Strip saved as PNG!');
    }}

    // Save Ablation Strip
    async function saveAblationComposite() {{
      showToast('Generating ablation strip...');
      const checks = Array.from(document.querySelectorAll('.abla-check:checked'));
      if (checks.length === 0) {{
        alert('Please check at least one ablation card.');
        return;
      }}

      const content = contents.find(c => c.id === currentActivePair.contentId);
      const style = styles.find(s => s.id === currentActivePair.styleId);

      const items = [
        {{ label: 'Content', src: content.file }},
        {{ label: 'Style Ref', src: style.file }},
        ...checks.map(c => ({{ label: c.dataset.name, src: c.dataset.src }}))
      ];

      const loadedImages = await Promise.all(items.map(async item => ({{
        label: item.label,
        img: await loadImage(item.src)
      }})));

      const validItems = loadedImages.filter(i => i.img !== null);
      if (validItems.length === 0) return;

      const itemW = 512;
      const itemH = 512;
      const headerH = 48;
      const totalW = validItems.length * itemW;
      const totalH = itemH + headerH;

      const canvas = document.getElementById('exportCanvas');
      canvas.width = totalW;
      canvas.height = totalH;
      const ctx = canvas.getContext('2d');

      ctx.fillStyle = '#0f172a';
      ctx.fillRect(0, 0, totalW, totalH);

      validItems.forEach((item, idx) => {{
        const x = idx * itemW;
        ctx.fillStyle = item.label.includes('Full') ? '#10b981' : '#f1f5f9';
        ctx.font = 'bold 20px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(item.label, x + itemW / 2, 32);

        ctx.drawImage(item.img, x, headerH, itemW, itemH);

        if (item.label.includes('Full')) {{
          ctx.strokeStyle = '#10b981';
          ctx.lineWidth = 6;
          ctx.strokeRect(x + 3, headerH + 3, itemW - 6, itemH - 6);
        }}
      }});

      const a = document.createElement('a');
      a.download = `ablation_${{currentActivePair.contentId}}_${{currentActivePair.styleId}}.png`;
      a.href = canvas.toDataURL('image/png');
      a.click();
      showToast('Ablation Strip saved as PNG!');
    }}

    // Export Full 15x15 Matrix Teaser Image (Like Paper Teaser Figure)
    async function exportFull15x15Teaser() {{
      showToast('Stitching full 15x15 matrix (this may take a few seconds)...');
      const cellPx = 256; // High resolution 256x256 per cell
      const headerPx = 256;
      const matrixSize = 15;
      const totalW = headerPx + (matrixSize * cellPx);
      const totalH = headerPx + (matrixSize * cellPx);

      const canvas = document.getElementById('exportCanvas');
      canvas.width = totalW;
      canvas.height = totalH;
      const ctx = canvas.getContext('2d');

      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, totalW, totalH);

      // 1. Draw Corner Dual Box
      ctx.fillStyle = '#e2e8f0';
      ctx.fillRect(0, 0, headerPx, headerPx);
      ctx.strokeStyle = '#94a3b8';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.lineTo(headerPx, headerPx);
      ctx.stroke();

      ctx.fillStyle = '#1e293b';
      ctx.font = 'bold 28px sans-serif';
      ctx.fillText('Style \\u2192', headerPx * 0.45, headerPx * 0.35);
      ctx.fillText('\\u2193 Content', headerPx * 0.08, headerPx * 0.85);

      // 2. Draw Top Style Headers
      for (let sIdx = 0; sIdx < styles.length; sIdx++) {{
        const s = styles[sIdx];
        const x = headerPx + (sIdx * cellPx);
        const img = await loadImage(s.file);
        if (img) {{
          ctx.drawImage(img, x, 0, cellPx, headerPx);
        }}
      }}

      // 3. Draw Left Content Headers
      for (let cIdx = 0; cIdx < contents.length; cIdx++) {{
        const c = contents[cIdx];
        const y = headerPx + (cIdx * cellPx);
        const img = await loadImage(c.file);
        if (img) {{
          ctx.drawImage(img, 0, y, headerPx, cellPx);
        }}
      }}

      // 4. Draw 225 Cells
      for (let cIdx = 0; cIdx < contents.length; cIdx++) {{
        const c = contents[cIdx];
        const y = headerPx + (cIdx * cellPx);
        for (let sIdx = 0; sIdx < styles.length; sIdx++) {{
          const s = styles[sIdx];
          const x = headerPx + (sIdx * cellPx);
          const imgSrc = gridData[c.id][s.id].ortho;
          const img = await loadImage(imgSrc);
          if (img) {{
            ctx.drawImage(img, x, y, cellPx, cellPx);
          }}
        }}
      }}

      // Download
      const a = document.createElement('a');
      a.download = 'qualitative_matrix_teaser_full15x15.png';
      a.href = canvas.toDataURL('image/png');
      a.click();
      showToast('Full 15x15 Matrix exported successfully!');
    }}
  </script>
</body>
</html>
"""

output_path = os.path.join(WORKSPACE_ROOT, "visualize_matrix.html")
with open(output_path, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"Generated {output_path} successfully!")
