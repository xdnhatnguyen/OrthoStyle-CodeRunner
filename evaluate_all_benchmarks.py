#!/usr/bin/env python3
"""
Master Evaluation Script for OrthoStyle Benchmarks & Ablations.
Evaluates 7 core metrics:
  1. csd_style_similarity: CSD ViT-Large style embedding cosine similarity
  2. style_retrieval_accuracy: Top-1 style reference retrieval among candidate styles
  3. clip_i_content_similarity: CLIP ViT-B/32 content embedding cosine similarity
  4. dino_content_similarity: DINO ViT-S/16 content embedding cosine similarity
  5. lpips: Perceptual patch distance to content reference (AlexNet)
  6. cfsd: Content Feature Structural Distance (VGG19 relu3_1)
  7. dcl: Directional Content Leakage (DINO grayscale)

Aggregates all results into structured Master CSV summary tables.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
from tqdm import tqdm

# Add local path for evaluation module
sys.path.append("evaluation")
from evaluate_style_transfer_metrics import (
    IMAGE_EXTS,
    CFSDMetric,
    CLIPImageEncoder,
    DINOImageEncoder,
    LPIPSMetric,
    add_average_row,
    build_csd_preprocess,
    cosine,
    csd_style_embedding_from_image,
    directional_content_leakage,
    list_images,
    resolve_generated_path,
    safe_metric,
    setup_csd,
    stem_to_path,
)

# Benchmark Task Registry
TASK_REGISTRY = {
    # 1. Main Benchmark (Table 1 & Prompt Robustness)
    "level1_null": {
        "desc": "Main Benchmark: Level 1 Null Prompt (Full 225 pairs)",
        "generated_dir": "output/benchmark/level1_null",
        "category": "table1_benchmark",
        "prefix": "ortho",
    },
    "level2_object": {
        "desc": "Main Benchmark: Level 2 Object Prompt (Full 225 pairs)",
        "generated_dir": "output/benchmark/level2_object",
        "category": "table1_benchmark",
        "prefix": "ortho",
    },
    "level3_style_desc": {
        "desc": "Main Benchmark: Level 3 Style Description (Full 225 pairs)",
        "generated_dir": "output/benchmark/level3_style_desc",
        "category": "table1_benchmark",
        "prefix": "ortho",
    },
    # 2. Table 2 Component Ablation Study (Evaluated on Null Prompt)
    "table2_A_full_ours": {
        "desc": "Table 2 (A): Full Proposed OrthoStyle",
        "generated_dir": "output/benchmark/level1_null",
        "category": "table2_ablation",
        "prefix": "ortho",
    },
    "table2_B_pure_mean": {
        "desc": "Table 2 (B): Pure Mean Token (alpha_s=0.0)",
        "generated_dir": "output/benchmark/ablation_B_pure_mean/level1_null",
        "category": "table2_ablation",
        "prefix": "ortho",
    },
    "table2_C_raw_style": {
        "desc": "Table 2 (C): Raw Style Token (alpha_s=1.0)",
        "generated_dir": "output/benchmark/ablation_C_raw_style/level1_null",
        "category": "table2_ablation",
        "prefix": "ortho",
    },
    "table2_D_no_guidance": {
        "desc": "Table 2 (D): No Guidance (No DINO, No Loss, No Grad)",
        "generated_dir": "output/benchmark/ablation_D_no_guidance/level1_null",
        "category": "table2_ablation",
        "prefix": "ortho",
    },
    "table2_E_no_pushforward": {
        "desc": "Table 2 (E): No AdaIN Pushforward (tau=0)",
        "generated_dir": "output/benchmark/ablation_E_no_pushforward/level1_null",
        "category": "table2_ablation",
        "prefix": "ortho",
    },
    "table2_F_no_gating": {
        "desc": "Table 2 (F): No ControlNet & No Gating (No Canny, No rembg)",
        "generated_dir": "output/benchmark/ablation_F_no_gating/level1_null",
        "category": "table2_ablation",
        "prefix": "ortho",
    },
    # Bonus intermediate runs (preserved)
    "table2_D2_no_ortho": {
        "desc": "Table 2 (D2): Standard Guidance (No Score-Orthogonal Projection)",
        "generated_dir": "output/benchmark/ablation_D2_no_ortho/level1_null",
        "category": "table2_bonus",
        "prefix": "ortho",
    },
    "table2_F2_no_semantic_gating": {
        "desc": "Table 2 (F2): Standard Canny ControlNet (No Semantic rembg Gating)",
        "generated_dir": "output/benchmark/ablation_F2_no_semantic_gating/level1_null",
        "category": "table2_bonus",
        "prefix": "ortho",
    },
    # 3. Tau Pushforward Sweeps (7x7 = 49 pairs)
    "sweep_tau_1": {
        "desc": "Tau Sweep: tau=1 (p_switch=0.05, 49 pairs)",
        "generated_dir": "output/benchmark/sweep_tau_1/level1_null",
        "category": "tau_sweeps",
        "prefix": "ortho",
    },
    "sweep_tau_2": {
        "desc": "Tau Sweep: tau=2 (Default p_switch=0.10, 49 pairs)",
        "generated_dir": "output/benchmark/sweep_tau_2/level1_null",
        "category": "tau_sweeps",
        "prefix": "ortho",
    },
    "sweep_tau_3": {
        "desc": "Tau Sweep: tau=3 (p_switch=0.15, 49 pairs)",
        "generated_dir": "output/benchmark/sweep_tau_3/level1_null",
        "category": "tau_sweeps",
        "prefix": "ortho",
    },
    "sweep_tau_4": {
        "desc": "Tau Sweep: tau=4 (p_switch=0.20, 49 pairs)",
        "generated_dir": "output/benchmark/sweep_tau_4/level1_null",
        "category": "tau_sweeps",
        "prefix": "ortho",
    },
    # 4. Baselines Comparison (Table 1: 225 pairs per baseline)
    "baseline_attenst": {
        "desc": "Baseline: AttnST (Attention-based Style Transfer)",
        "generated_dir": "baseline_results/outputs_attenst_soictdata",
        "category": "baselines",
        "prefix": "attenst",
    },
    "baseline_diffuseIT": {
        "desc": "Baseline: DiffuseIT",
        "generated_dir": "baseline_results/outputs_diffuseIT_soictdata",
        "category": "baselines",
        "prefix": "DiffuseIT",
    },
    "baseline_instantstyle": {
        "desc": "Baseline: InstantStyle",
        "generated_dir": "baseline_results/outputs_instantstyle_soictdata",
        "category": "baselines",
        "prefix": "instantstyle",
    },
    "baseline_itoc": {
        "desc": "Baseline: ITOC",
        "generated_dir": "baseline_results/outputs_itoc_soictdata",
        "category": "baselines",
        "prefix": "itoc",
    },
    "baseline_rb_modulation": {
        "desc": "Baseline: RB-Modulation",
        "generated_dir": "baseline_results/outputs_rb_soictdata",
        "category": "baselines",
        "prefix": "outputs_rb_soictdata",
    },
    "baseline_styleid": {
        "desc": "Baseline: StyleID",
        "generated_dir": "baseline_results/outputs_styleid_soictdata",
        "category": "baselines",
        "prefix": "styleid",
    },
}

SUITES = {
    "main": ["level1_null", "level2_object", "level3_style_desc"],
    "table2": [
        "table2_A_full_ours",
        "table2_B_pure_mean",
        "table2_C_raw_style",
        "table2_D_no_guidance",
        "table2_E_no_pushforward",
        "table2_F_no_gating",
    ],
    "sweeps": ["sweep_tau_1", "sweep_tau_2", "sweep_tau_3", "sweep_tau_4"],
    "baselines": [
        "table2_A_full_ours",
        "baseline_attenst",
        "baseline_diffuseIT",
        "baseline_instantstyle",
        "baseline_itoc",
        "baseline_rb_modulation",
        "baseline_styleid",
    ],
    "all": list(TASK_REGISTRY.keys()),
}


class ModelEvaluator:
    """Preloads all metric models once onto GPU/CPU to avoid repeated reloading overhead."""

    def __init__(self, device: str = "cuda:0", csd_ckpt: str = "third_party/CSD/checkpoint.pth"):
        self.device = device
        print(f"[*] Initializing Evaluation Suite on device: {device}...")
        t0 = time.time()

        # 1. CSD
        print("  [1/5] Loading CSD ViT-Large model...", flush=True)
        self.csd_model = setup_csd(device, csd_checkpoint=csd_ckpt, csd_third_party="third_party")
        self.csd_preprocess = build_csd_preprocess()

        # 2. CLIP-I
        print("  [2/5] Loading CLIP-I ViT-B/32 vision encoder...", flush=True)
        self.clip_encoder = CLIPImageEncoder("openai/clip-vit-base-patch32", device)

        # 3. DINO
        print("  [3/5] Loading DINO ViT-S/16 model...", flush=True)
        self.dino_encoder = DINOImageEncoder("facebook/dino-vits16", device)

        # 4. LPIPS
        print("  [4/5] Loading LPIPS (AlexNet)...", flush=True)
        self.lpips_metric = LPIPSMetric(device, net="alex")

        # 5. CFSD
        print("  [5/5] Loading CFSD (VGG19 relu3_1)...", flush=True)
        self.cfsd_metric = CFSDMetric(device, image_size=224, vgg_until_layer=12)

        print(f"[✔] All evaluation models loaded successfully in {time.time() - t0:.1f}s!\n")

    def encode_references(self, content_dir: str, style_dir: str):
        """Precomputes embeddings for all content and style references."""
        print("[*] Pre-encoding reference content & style images...")
        content_paths = stem_to_path(content_dir)
        style_paths = stem_to_path(style_dir)

        # Style CSD
        style_csd_embs = {}
        style_dcl_embs = {}
        for name, path in tqdm(style_paths.items(), desc="Encoding style refs", leave=False):
            style_csd_embs[name] = csd_style_embedding_from_image(
                self.csd_model, self.csd_preprocess, path, self.device, is_generated=False
            )
            style_dcl_embs[name] = self.dino_encoder.encode_path(path, is_generated=False, grayscale=True)

        # Content CLIP & DINO
        content_clip_embs = {}
        content_dino_embs = {}
        content_dcl_embs = {}
        for name, path in tqdm(content_paths.items(), desc="Encoding content refs", leave=False):
            content_clip_embs[name] = self.clip_encoder.encode_path(path, is_generated=False)
            content_dino_embs[name] = self.dino_encoder.encode_path(path, is_generated=False, grayscale=False)
            content_dcl_embs[name] = self.dino_encoder.encode_path(path, is_generated=False, grayscale=True)

        return {
            "content_paths": content_paths,
            "style_paths": style_paths,
            "style_csd_embs": style_csd_embs,
            "style_dcl_embs": style_dcl_embs,
            "content_clip_embs": content_clip_embs,
            "content_dino_embs": content_dino_embs,
            "content_dcl_embs": content_dcl_embs,
        }

    def evaluate_task(
        self,
        task_name: str,
        task_info: dict,
        ref_data: dict,
        output_dir: str = "metrics",
        overwrite: bool = False,
    ) -> Optional[dict]:
        generated_dir = task_info["generated_dir"]
        prefix = task_info.get("prefix", "ortho")
        desc = task_info.get("desc", task_name)

        summary_csv = os.path.join(output_dir, f"metrics_summary_{task_name}.csv")
        full_csv = os.path.join(output_dir, f"metrics_full_{task_name}.csv")

        # Cache check
        if not overwrite and os.path.exists(summary_csv):
            print(f"[{task_name}] Summary CSV exists -> Loading from cache: {summary_csv}")
            try:
                cached_df = pd.read_csv(summary_csv)
                res_dict = cached_df.to_dict(orient="records")[0]
                res_dict["task_name"] = task_name
                res_dict["task_desc"] = desc
                res_dict["category"] = task_info.get("category", "")
                return res_dict
            except Exception:
                pass

        if not os.path.exists(generated_dir):
            print(f"[{task_name}] [SKIP - NO DIR] Directory not found: {generated_dir}")
            return None

        # Check existing images
        gen_images = [f for f in os.listdir(generated_dir) if any(f.endswith(ext) for ext in IMAGE_EXTS)]
        if len(gen_images) == 0:
            print(f"[{task_name}] [SKIP - EMPTY] No generated images in: {generated_dir}")
            return None

        print(f"\n{'='*70}")
        print(f"[*] Evaluating Task: {task_name} ({desc})")
        print(f"    Source folder: {generated_dir} ({len(gen_images)} images found)")
        print(f"{'='*70}")

        content_paths = ref_data["content_paths"]
        style_paths = ref_data["style_paths"]
        style_csd_embs = ref_data["style_csd_embs"]
        style_dcl_embs = ref_data["style_dcl_embs"]
        content_clip_embs = ref_data["content_clip_embs"]
        content_dino_embs = ref_data["content_dino_embs"]
        content_dcl_embs = ref_data["content_dcl_embs"]

        rows = []
        t_start = time.time()

        for content_name, content_path in content_paths.items():
            for style_name, style_path in style_paths.items():
                gen_path = resolve_generated_path(
                    generated_dir=generated_dir,
                    prefix=prefix,
                    content_name=content_name,
                    style_name=style_name,
                    generated_pattern="{prefix}_{content}_{style}.png",
                    allow_glob_fallback=True,
                )

                if gen_path is None or not gen_path.exists():
                    continue

                row = {
                    "content_name": content_name,
                    "style_name": style_name,
                    "generated_file": gen_path.name,
                }

                # 1. CSD Style Similarity & Retrieval
                gen_csd_emb = safe_metric(
                    lambda: csd_style_embedding_from_image(
                        self.csd_model, self.csd_preprocess, gen_path, self.device, is_generated=True
                    )
                )
                if gen_csd_emb is not None and style_name in style_csd_embs:
                    row["csd_style_similarity"] = cosine(gen_csd_emb, style_csd_embs[style_name])
                    sims = {s: cosine(gen_csd_emb, emb) for s, emb in style_csd_embs.items()}
                    top1 = max(sims, key=sims.get)
                    row["style_retrieval_correct"] = 1.0 if top1 == style_name else 0.0

                # 2. CLIP-I Content Similarity
                gen_clip_emb = safe_metric(lambda: self.clip_encoder.encode_path(gen_path, is_generated=True))
                if gen_clip_emb is not None and content_name in content_clip_embs:
                    row["clip_i_content_similarity"] = cosine(gen_clip_emb, content_clip_embs[content_name])

                # 3. DINO Content Similarity
                gen_dino_emb = safe_metric(
                    lambda: self.dino_encoder.encode_path(gen_path, is_generated=True, grayscale=False)
                )
                if gen_dino_emb is not None and content_name in content_dino_embs:
                    row["dino_content_similarity"] = cosine(gen_dino_emb, content_dino_embs[content_name])

                # 4. DCL
                gen_dcl_emb = safe_metric(
                    lambda: self.dino_encoder.encode_path(gen_path, is_generated=True, grayscale=True)
                )
                if gen_dcl_emb is not None and content_name in content_dcl_embs and style_name in style_dcl_embs:
                    row["dcl"] = directional_content_leakage(
                        content_dcl_embs[content_name], style_dcl_embs[style_name], gen_dcl_emb, clip=True
                    )

                # 5. LPIPS
                row["lpips"] = safe_metric(lambda: self.lpips_metric.distance(content_path, gen_path))

                # 6. CFSD
                row["cfsd"] = safe_metric(lambda: self.cfsd_metric.distance(content_path, gen_path))

                rows.append(row)

        elapsed = time.time() - t_start
        if len(rows) == 0:
            print(f"[{task_name}] No matching pairs could be evaluated.")
            return None

        df = pd.DataFrame(rows)
        df_with_avg = add_average_row(df)
        df_with_avg.to_csv(full_csv, index=False)

        metric_cols = [
            "csd_style_similarity",
            "style_retrieval_correct",
            "clip_i_content_similarity",
            "dino_content_similarity",
            "lpips",
            "cfsd",
            "dcl",
        ]
        summary_df = df[metric_cols].mean(numeric_only=True).to_frame("mean").T
        summary_df = summary_df.rename(columns={"style_retrieval_correct": "style_retrieval_accuracy"})
        summary_df.insert(0, "num_evaluated", len(df))
        summary_df.insert(0, "category", task_info.get("category", ""))
        summary_df.insert(0, "task_desc", desc)
        summary_df.insert(0, "task_name", task_name)

        summary_df.to_csv(summary_csv, index=False)
        print(f"[✔] Evaluated {len(df)} pairs in {elapsed:.1f}s ({elapsed/max(1, len(df)):.2f}s/pair)")
        print(summary_df.to_string(index=False))

        return summary_df.to_dict(orient="records")[0]


def main():
    parser = argparse.ArgumentParser(description="Master OrthoStyle Benchmarks & Ablations Evaluator")
    parser.add_argument("--suite", type=str, default="all", choices=list(SUITES.keys()) + list(TASK_REGISTRY.keys()))
    parser.add_argument("--content_dir", type=str, default="data/content")
    parser.add_argument("--style_dir", type=str, default="data/style")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--csd_checkpoint", type=str, default="third_party/CSD/checkpoint.pth")
    parser.add_argument("--output_dir", type=str, default="metrics")
    parser.add_argument("--overwrite", action="store_true", help="Force re-evaluation even if summary CSV exists")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Determine tasks to evaluate
    if args.suite in SUITES:
        target_task_keys = SUITES[args.suite]
    else:
        target_task_keys = [args.suite]

    print(f"[*] Starting Evaluation Pipeline for Suite: '{args.suite}' ({len(target_task_keys)} tasks)")
    print(f"[*] Results will be saved to: '{args.output_dir}/'")

    # Preload models
    evaluator = ModelEvaluator(device=args.device, csd_ckpt=args.csd_checkpoint)
    ref_data = evaluator.encode_references(args.content_dir, args.style_dir)

    all_summaries = []

    for task_key in target_task_keys:
        task_info = TASK_REGISTRY[task_key]
        res = evaluator.evaluate_task(
            task_name=task_key,
            task_info=task_info,
            ref_data=ref_data,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
        if res is not None:
            all_summaries.append(res)

    if len(all_summaries) > 0:
        master_df = pd.DataFrame(all_summaries)

        # Reorder columns nicely
        col_order = [
            "task_name",
            "category",
            "task_desc",
            "num_evaluated",
            "csd_style_similarity",
            "style_retrieval_accuracy",
            "clip_i_content_similarity",
            "dino_content_similarity",
            "lpips",
            "cfsd",
            "dcl",
        ]
        # Merge with existing master if present
        master_path = os.path.join(args.output_dir, "master_all_metrics.csv")
        if os.path.exists(master_path):
            try:
                old_master = pd.read_csv(master_path)
                combined = pd.concat([old_master, master_df], ignore_index=True)
                master_df = combined.drop_duplicates(subset=["task_name"], keep="last")
            except Exception:
                pass

        cols = [c for c in col_order if c in master_df.columns] + [c for c in master_df.columns if c not in col_order]
        master_df = master_df[cols]
        master_df.to_csv(master_path, index=False)

        # Categorized master reports
        # Table 1: Main Benchmark
        df_table1 = master_df[master_df["category"] == "table1_benchmark"]
        if len(df_table1) > 0:
            p1 = os.path.join(args.output_dir, "master_table1_benchmark.csv")
            df_table1.to_csv(p1, index=False)
            print(f"\n[✔] Table 1 Summary saved to: {p1}")

        # Table 2: Component Ablations
        df_table2 = master_df[master_df["category"].str.startswith("table2")]
        if len(df_table2) > 0:
            p2 = os.path.join(args.output_dir, "master_table2_ablations.csv")
            df_table2.to_csv(p2, index=False)
            print(f"[✔] Table 2 Summary saved to: {p2}")

        # Tau Sweeps
        df_sweeps = master_df[master_df["category"] == "tau_sweeps"]
        if len(df_sweeps) > 0:
            p3 = os.path.join(args.output_dir, "master_sweeps_tau.csv")
            df_sweeps.to_csv(p3, index=False)
            print(f"[✔] Tau Sweeps Summary saved to: {p3}")

        # Baselines Comparison (Ours vs Baselines)
        df_baselines = master_df[
            (master_df["category"] == "baselines") | (master_df["task_name"].isin(["table2_A_full_ours", "level1_null"]))
        ].drop_duplicates(subset=["task_name"], keep="last")
        if len(df_baselines) > 0:
            p_base = os.path.join(args.output_dir, "master_baselines_comparison.csv")
            df_baselines.to_csv(p_base, index=False)
            print(f"[✔] Baselines Comparison Summary saved to: {p_base}")

        print(f"\n{'='*80}")
        print("MASTER BENCHMARK METRICS SUMMARY:")
        print(f"{'='*80}")
        print(master_df.to_string(index=False))
        print(f"{'='*80}\n")
    else:
        print("\n[!] No completed tasks found to summarize. Check generated_dir paths or wait for generation.")


if __name__ == "__main__":
    main()
