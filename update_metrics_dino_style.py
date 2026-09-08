#!/usr/bin/env python3
"""
update_metrics_dino_style.py
Recomputes / aggregates DINO Style Alignment metric (cosine similarity between generated image and style reference image)
across all tasks (Table 1 ablations, sweeps, prompt levels, and 6 baselines).

Preserves all existing metrics (CSD, CLIP-I, LPIPS, CFSD, DCL), updates master files,
and exports the two tables formatted with:
  - Metric values multiplied by 100
  - Decimal separator as comma ','
  - 3 decimal places
"""

import os
import sys
import time
from pathlib import Path
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from PIL import Image

from evaluation.evaluate_style_transfer_metrics import (
    DINOImageEncoder,
    stem_to_path,
    resolve_generated_path,
    cosine,
)
from evaluate_all_benchmarks import TASK_REGISTRY

def format_comma(val, decimals=3):
    if pd.isna(val) or val is None:
        return ""
    try:
        fval = float(val) * 100.0
        s = f"{fval:.{decimals}f}"
        return s.replace(".", ",")
    except Exception:
        return str(val)

def clean_and_summarize(output_dir="metrics"):
    task_summaries = {}

    for task_name, task_info in TASK_REGISTRY.items():
        full_csv = os.path.join(output_dir, f"metrics_full_{task_name}.csv")
        summary_csv = os.path.join(output_dir, f"metrics_summary_{task_name}.csv")

        if not os.path.exists(full_csv):
            continue

        df = pd.read_csv(full_csv)

        # Drop any Average / AVERAGE / nan rows
        df = df[~df["content_name"].astype(str).str.strip().str.lower().isin(["average", "nan", ""])].copy()

        metric_cols = [c for c in [
            "csd_style_similarity",
            "style_retrieval_correct",
            "style_retrieval_accuracy",
            "clip_i_content_similarity",
            "dino_content_similarity",
            "dino_style_similarity",
            "dino",
            "lpips",
            "cfsd",
            "dcl",
        ] if c in df.columns]

        avg_series = df[metric_cols].mean(numeric_only=True)
        avg_row = {col: "" for col in df.columns}
        avg_row["content_name"] = "Average"
        for col in metric_cols:
            avg_row[col] = avg_series[col]

        # Re-save cleaned full CSV with a single Average row
        df_with_avg = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
        df_with_avg.to_csv(full_csv, index=False)

        # Save single summary row
        summary_row = {
            "task_name": task_name,
            "task_desc": task_info.get("desc", task_name),
            "category": task_info.get("category", ""),
            "num_evaluated": len(df),
        }
        for col in metric_cols:
            summary_row[col] = avg_series[col]

        if "style_retrieval_correct" in summary_row and "style_retrieval_accuracy" not in summary_row:
            summary_row["style_retrieval_accuracy"] = summary_row["style_retrieval_correct"]

        summary_df = pd.DataFrame([summary_row])
        summary_df.to_csv(summary_csv, index=False)
        task_summaries[task_name] = summary_row

    # Update master_all_metrics.csv
    master_rows = list(task_summaries.values())
    master_df = pd.DataFrame(master_rows)
    master_csv = os.path.join(output_dir, "master_all_metrics.csv")
    master_df.to_csv(master_csv, index=False)
    print(f"[✔] Master metrics updated: {master_csv} ({len(master_df)} tasks)")

    # 4. Generate Table 1: Ablations, Sweeps, and Prompting (Image 1)
    table1_rows = [
        {
            "row_id": "(A)",
            "Model": "(A) Full method (OrthoStyle)",
            "task_name": "level1_null",
            "num_eval": 225,
            "blended_tokens": "x",
            "adain": "x",
            "semantic_gated": "x",
            "score_ortho": "x",
            "alpha_s": "0,85",
            "tau": "1",
            "prompting": "Null",
            "status": "FALSE",
        },
        {
            "row_id": "(B)",
            "Model": "(B) Pure Mean Token (α_s = 0)",
            "task_name": "table2_B_pure_mean",
            "num_eval": 225,
            "blended_tokens": "x",
            "adain": "x",
            "semantic_gated": "x",
            "score_ortho": "x",
            "alpha_s": "0",
            "tau": "1",
            "prompting": "Null",
            "status": "FALSE",
        },
        {
            "row_id": "(C)",
            "Model": "(C) Raw Style Token (α_s = 1.0)",
            "task_name": "table2_C_raw_style",
            "num_eval": 225,
            "blended_tokens": "x",
            "adain": "x",
            "semantic_gated": "x",
            "score_ortho": "x",
            "alpha_s": "1",
            "tau": "1",
            "prompting": "Null",
            "status": "FALSE",
        },
        {
            "row_id": "(D)",
            "Model": "(D) w/o Score-Orthogonal Guidance",
            "task_name": "table2_D_no_guidance",
            "num_eval": 225,
            "blended_tokens": "x",
            "adain": "x",
            "semantic_gated": "x",
            "score_ortho": "",
            "alpha_s": "0,85",
            "tau": "1",
            "prompting": "Null",
            "status": "FALSE",
        },
        {
            "row_id": "(E)",
            "Model": "(E) w/o AdaIN Pushforward",
            "task_name": "table2_E_no_pushforward",
            "num_eval": 225,
            "blended_tokens": "x",
            "adain": "",
            "semantic_gated": "x",
            "score_ortho": "x",
            "alpha_s": "0,85",
            "tau": "1",
            "prompting": "Null",
            "status": "FALSE",
        },
        {
            "row_id": "(F)",
            "Model": "(F) w/o Semantic Gated Canny",
            "task_name": "table2_F_no_gating",
            "num_eval": 225,
            "blended_tokens": "x",
            "adain": "x",
            "semantic_gated": "",
            "score_ortho": "x",
            "alpha_s": "0,85",
            "tau": "1",
            "prompting": "Null",
            "status": "FALSE",
        },
        {
            "row_id": "(E_sweep_1)",
            "Model": "(E) Sweep",
            "task_name": "sweep_tau_1",
            "num_eval": 49,
            "blended_tokens": "x",
            "adain": "x",
            "semantic_gated": "x",
            "score_ortho": "x",
            "alpha_s": "0,85",
            "tau": "1",
            "prompting": "Null",
            "status": "FALSE",
        },
        {
            "row_id": "(E_sweep_2)",
            "Model": "(E) Sweep",
            "task_name": "sweep_tau_2",
            "num_eval": 49,
            "blended_tokens": "x",
            "adain": "x",
            "semantic_gated": "x",
            "score_ortho": "x",
            "alpha_s": "0,85",
            "tau": "2",
            "prompting": "Null",
            "status": "FALSE",
        },
        {
            "row_id": "(E_sweep_3)",
            "Model": "(E) Sweep",
            "task_name": "sweep_tau_3",
            "num_eval": 49,
            "blended_tokens": "x",
            "adain": "x",
            "semantic_gated": "x",
            "score_ortho": "x",
            "alpha_s": "0,85",
            "tau": "3",
            "prompting": "Null",
            "status": "FALSE",
        },
        {
            "row_id": "(E_sweep_4)",
            "Model": "(E) Sweep",
            "task_name": "sweep_tau_4",
            "num_eval": 49,
            "blended_tokens": "x",
            "adain": "x",
            "semantic_gated": "x",
            "score_ortho": "x",
            "alpha_s": "0,85",
            "tau": "4",
            "prompting": "Null",
            "status": "FALSE",
        },
        {
            "row_id": "(H_obj)",
            "Model": "(H) Prompting level",
            "task_name": "level2_object",
            "num_eval": 225,
            "blended_tokens": "x",
            "adain": "x",
            "semantic_gated": "x",
            "score_ortho": "x",
            "alpha_s": "0,85",
            "tau": "1",
            "prompting": "Obj",
            "status": "FALSE",
        },
        {
            "row_id": "(H_sty)",
            "Model": "(H) Prompting level",
            "task_name": "level3_style_desc",
            "num_eval": 225,
            "blended_tokens": "x",
            "adain": "x",
            "semantic_gated": "x",
            "score_ortho": "x",
            "alpha_s": "0,85",
            "tau": "1",
            "prompting": "Sty",
            "status": "FALSE",
        },
    ]

    t1_out = []
    for r in table1_rows:
        t_name = r["task_name"]
        sum_data = task_summaries.get(t_name, {})
        entry = {
            "Model": r["Model"],
            "Num eval": r["num_eval"],
            "Blended Style Tokens & Early step controller": r["blended_tokens"],
            "AdaIN": r["adain"],
            "Semantic Gated Preprocessing": r["semantic_gated"],
            "Score-Orthogonal Guidance": r["score_ortho"],
            "Style token Blended ($\\alpha_s$)": r["alpha_s"],
            "Sweep ($\\tau$)": r["tau"],
            "Prompting": r["prompting"],
            "CSD (H)": format_comma(sum_data.get("csd_style_similarity")),
            "DINO (H)": format_comma(sum_data.get("dino_style_similarity")),
            "CLIP-I (H)": format_comma(sum_data.get("clip_i_content_similarity")),
            "LPIPS (L)": format_comma(sum_data.get("lpips")),
            "DCL (D)": format_comma(sum_data.get("dcl")),
            "Status": r["status"],
        }
        t1_out.append(entry)

    df_t1 = pd.DataFrame(t1_out)
    p_t1 = os.path.join(output_dir, "table1_ablation_sweeps_formatted.csv")
    df_t1.to_csv(p_t1, index=False)
    print(f"[✔] Formatted Table 1 saved: {p_t1}")

    # 5. Generate Table 2: Comparison with Baselines (Image 2)
    table2_models = [
        ("OrthoStyle", "level1_null"),
        ("StyleID", "baseline_styleid"),
        ("AttenST", "baseline_attenst"),
        ("DiffuseIT", "baseline_diffuseIT"),
        ("InstantStyle", "baseline_instantstyle"),
        ("ITOC", "baseline_itoc"),
        ("RB-Modulation", "baseline_rb_modulation"),
    ]

    t2_out = []
    for display_name, t_name in table2_models:
        sum_data = task_summaries.get(t_name, {})
        entry = {
            "Model": display_name,
            "CSD (H)": format_comma(sum_data.get("csd_style_similarity")),
            "DINO (H)": format_comma(sum_data.get("dino_style_similarity")),
            "CLIP-I (H)": format_comma(sum_data.get("clip_i_content_similarity")),
            "LPIPS (L)": format_comma(sum_data.get("lpips")),
            "DCL (D)": format_comma(sum_data.get("dcl")),
            "Status": "TRUE",
        }
        t2_out.append(entry)

    df_t2 = pd.DataFrame(t2_out)
    p_t2 = os.path.join(output_dir, "table2_baselines_formatted.csv")
    df_t2.to_csv(p_t2, index=False)
    print(f"[✔] Formatted Table 2 saved: {p_t2}")

    return df_t1, df_t2

if __name__ == "__main__":
    clean_and_summarize()
