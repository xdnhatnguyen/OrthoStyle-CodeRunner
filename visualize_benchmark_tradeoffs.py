#!/usr/bin/env python3
"""
visualize_benchmark_tradeoffs.py
Generates publication-quality figures for OrthoStyle style transfer paper:
  1. Figure 1: 2D Pareto Frontier (Content Preservation vs Style Fidelity) with Sweep curve
  2. Figure 2: 5-Axis Radar Chart showing multi-dimensional balance
  3. Figure 3: Head-to-Head percentage improvement over Backbone (RB-Modulation)
  4. Figure 4: Overall Harmonic Balance Score (Style vs Content)
  5. Master Combined Figure: 2x2 publication-ready layout
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.patheffects as pe

# Set publication style
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["axes.edgecolor"] = "#2D3748"
plt.rcParams["axes.labelcolor"] = "#1A202C"
plt.rcParams["xtick.color"] = "#2D3748"
plt.rcParams["ytick.color"] = "#2D3748"
plt.rcParams["grid.color"] = "#E2E8F0"
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.alpha"] = 0.7

OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Data Loading & Preparation
# -----------------------------------------------------------------------------
df_master = pd.read_csv("metrics/master_all_metrics.csv")

# Extract 7 baselines
baseline_keys = [
    "level1_null",
    "baseline_styleid",
    "baseline_attenst",
    "baseline_diffuseIT",
    "baseline_instantstyle",
    "baseline_itoc",
    "baseline_rb_modulation",
]
name_map = {
    "level1_null": "OrthoStyle (Ours)",
    "baseline_styleid": "StyleID",
    "baseline_attenst": "AttenST",
    "baseline_diffuseIT": "DiffuseIT",
    "baseline_instantstyle": "InstantStyle",
    "baseline_itoc": "ITOC",
    "baseline_rb_modulation": "RB-Modulation",
}
color_map = {
    "OrthoStyle (Ours)": "#D92638",  # Bold Crimson
    "StyleID": "#3182CE",           # Blue
    "AttenST": "#38A169",           # Green
    "AttnST": "#38A169",            # Green fallback
    "DiffuseIT": "#805AD5",         # Purple
    "InstantStyle": "#DD6B20",      # Orange
    "ITOC": "#718096",              # Slate Gray
    "RB-Modulation": "#4A5568",     # Dark Gray
}

df_b = df_master[df_master["task_name"].isin(baseline_keys)].copy()
df_b["display_name"] = df_b["task_name"].map(name_map)
df_b["csd"] = df_b["csd_style_similarity"] * 100.0
df_b["dino"] = df_b["dino_style_similarity"] * 100.0
df_b["clip"] = df_b["clip_i_content_similarity"] * 100.0
df_b["lpips"] = df_b["lpips"] * 100.0
df_b["dcl"] = df_b["dcl"] * 100.0

# Extract tau sweeps
sweep_keys = ["sweep_tau_1", "sweep_tau_2", "sweep_tau_3", "sweep_tau_4"]
df_sweeps = df_master[df_master["task_name"].isin(sweep_keys)].copy()
df_sweeps["csd"] = df_sweeps["csd_style_similarity"] * 100.0
df_sweeps["dino"] = df_sweeps["dino_style_similarity"] * 100.0
df_sweeps["clip"] = df_sweeps["clip_i_content_similarity"] * 100.0
df_sweeps["lpips"] = df_sweeps["lpips"] * 100.0
df_sweeps["dcl"] = df_sweeps["dcl"] * 100.0
df_sweeps = df_sweeps.sort_values(by="task_name")


# -----------------------------------------------------------------------------
# Figure 1: 2D Pareto Frontier (Content vs Style)
# -----------------------------------------------------------------------------
def plot_figure1_pareto():
    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    # Background quadrant shading
    ax.axvspan(64, 73.5, ymin=0.55, ymax=1.0, color="#FED7D7", alpha=0.25, zorder=1)
    ax.text(65, 49.5, "Content Collapse Zone\n(High Style, Destroyed Content)", fontsize=9, color="#9B2C2C", style="italic", weight="bold")

    ax.axvspan(83, 89, ymin=0, ymax=0.35, color="#EDF2F7", alpha=0.35, zorder=1)
    ax.text(83.2, 17, "Style Failure Zone\n(Reconstruction Only)", fontsize=9, color="#4A5568", style="italic", weight="bold")

    # Plot Tau Sweep curve (Pareto trajectory for OrthoStyle)
    sweep_x = df_sweeps["clip"].tolist()
    sweep_y = df_sweeps["csd"].tolist()
    ours_row = df_b[df_b["display_name"] == "OrthoStyle (Ours)"].iloc[0]
    sweep_x.append(ours_row["clip"])
    sweep_y.append(ours_row["csd"])
    sort_idx = np.argsort(sweep_x)
    sweep_x_sorted = np.array(sweep_x)[sort_idx]
    sweep_y_sorted = np.array(sweep_y)[sort_idx]

    ax.plot(sweep_x_sorted, sweep_y_sorted, color="#E53E3E", linestyle="--", linewidth=2.2, alpha=0.85, zorder=3, label="OrthoStyle Frontier (varying $\\tau$)")

    # Plot baseline points
    for _, row in df_b.iterrows():
        name = row["display_name"]
        x = row["clip"]
        y = row["csd"]
        color = color_map.get(name, "#4A5568")

        if name == "OrthoStyle (Ours)":
            ax.scatter(x, y, color=color, s=280, marker="*", edgecolor="#FFFFFF", linewidth=1.5, zorder=6, label="OrthoStyle (Ours)")
            ax.annotate(
                f"  {name}\n  (Pareto Sweet Spot)",
                xy=(x, y),
                xytext=(x - 6.5, y + 2.8),
                fontsize=10.5,
                fontweight="bold",
                color="#9B2C2C",
                arrowprops=dict(arrowstyle="->", color="#9B2C2C", lw=1.5),
                zorder=7,
            )
        else:
            ax.scatter(x, y, color=color, s=120, marker="o", edgecolor="#2D3748", linewidth=1.0, zorder=4)
            offset = (1.2, -1.2)
            if name == "InstantStyle":
                offset = (-7.0, 1.2)
            elif name == "DiffuseIT":
                offset = (1.2, 1.0)
            elif name == "ITOC":
                offset = (-3.5, 1.5)
            elif name == "RB-Modulation":
                offset = (-8.0, -2.5)
            elif name == "AttenST":
                offset = (-5.5, -2.2)
            elif name == "StyleID":
                offset = (1.2, -0.2)

            ax.annotate(
                name,
                xy=(x, y),
                xytext=(x + offset[0], y + offset[1]),
                fontsize=9.5,
                fontweight="medium",
                color="#2D3748",
                zorder=5,
            )

    # Annotate Tau sweep points
    for idx, row in df_sweeps.iterrows():
        t_label = row["task_name"].replace("sweep_tau_", "$\\tau$=")
        ax.scatter(row["clip"], row["csd"], color="#FC8181", s=45, marker="^", edgecolor="#C53030", linewidth=0.8, zorder=4)
        ax.annotate(t_label, (row["clip"], row["csd"]), xytext=(row["clip"] + 0.4, row["csd"] + 0.5), fontsize=7.5, color="#C53030")

    ax.set_xlabel("Content Preservation (CLIP-I $\\uparrow$)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Style Fidelity (CSD $\\uparrow$)", fontsize=12, fontweight="bold")
    ax.set_title("Style-Content Pareto Frontier: OrthoStyle vs. Baselines", fontsize=14, fontweight="bold", pad=12)

    ax.set_xlim(64, 90)
    ax.set_ylim(10, 52)
    ax.grid(True)
    ax.legend(loc="lower left", frameon=True, framealpha=0.92, fontsize=9.5)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "figure1_pareto_frontier.png"), dpi=300)
    plt.savefig(os.path.join(OUTPUT_DIR, "figure1_pareto_frontier.pdf"))
    plt.close()
    print("[✔] Saved Figure 1: Pareto Frontier")


# -----------------------------------------------------------------------------
# Figure 2: 5-Axis Radar Chart
# -----------------------------------------------------------------------------
def plot_figure2_radar():
    categories = [
        "Style Sim.\n(CSD $\\uparrow$)",
        "Style Align.\n(DINO $\\uparrow$)",
        "Content Pres.\n(CLIP-I $\\uparrow$)",
        "Struct. Fid.\n(100 - LPIPS $\\uparrow$)",
        "Disentangle.\n(100 - DCL $\\uparrow$)",
    ]
    N = len(categories)

    plot_models = ["OrthoStyle (Ours)", "StyleID", "AttenST", "InstantStyle", "RB-Modulation"]

    raw_matrix = []
    for m in plot_models:
        r = df_b[df_b["display_name"] == m].iloc[0]
        raw_matrix.append([
            r["csd"],
            r["dino"],
            r["clip"],
            100.0 - r["lpips"],
            100.0 - r["dcl"],
        ])
    raw_matrix = np.array(raw_matrix)

    norm_matrix = np.zeros_like(raw_matrix)
    for col in range(N):
        c_min = raw_matrix[:, col].min()
        c_max = raw_matrix[:, col].max()
        norm_matrix[:, col] = 0.25 + 0.75 * (raw_matrix[:, col] - c_min) / (c_max - c_min + 1e-6)

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7.5, 7.0), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], categories, size=10, weight="bold")
    ax.set_rlabel_position(0)
    plt.yticks([0.4, 0.6, 0.8, 1.0], ["Low", "Med", "High", "Max"], color="#A0AEC0", size=8)
    plt.ylim(0, 1.1)

    for idx, m in enumerate(plot_models):
        values = norm_matrix[idx].tolist()
        values += values[:1]
        c = color_map.get(m, "#4A5568")

        if m == "OrthoStyle (Ours)":
            ax.plot(angles, values, linewidth=2.8, linestyle="solid", color=c, label=m, zorder=6)
            ax.fill(angles, values, color=c, alpha=0.25, zorder=5)
        elif m == "RB-Modulation":
            ax.plot(angles, values, linewidth=1.8, linestyle="--", color=c, label=m, zorder=4)
        else:
            ax.plot(angles, values, linewidth=1.5, linestyle="-", color=c, label=m, alpha=0.8, zorder=3)

    plt.title("Multi-Metric Evaluation Profile\n(Holistic Performance Comparison)", size=13, weight="bold", y=1.08)
    plt.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "figure2_radar_chart.png"), dpi=300)
    plt.savefig(os.path.join(OUTPUT_DIR, "figure2_radar_chart.pdf"))
    plt.close()
    print("[✔] Saved Figure 2: Radar Chart")


# -----------------------------------------------------------------------------
# Figure 3: Head-to-Head Improvement over RB-Modulation
# -----------------------------------------------------------------------------
def plot_figure3_head_to_head():
    ours = df_b[df_b["display_name"] == "OrthoStyle (Ours)"].iloc[0]
    rbm = df_b[df_b["display_name"] == "RB-Modulation"].iloc[0]

    pct_csd = ((ours["csd"] - rbm["csd"]) / rbm["csd"]) * 100.0
    pct_dino = ((ours["dino"] - rbm["dino"]) / rbm["dino"]) * 100.0
    pct_lpips = ((rbm["lpips"] - ours["lpips"]) / rbm["lpips"]) * 100.0
    pct_dcl = ((rbm["dcl"] - ours["dcl"]) / rbm["dcl"]) * 100.0
    pct_clip = ((ours["clip"] - rbm["clip"]) / rbm["clip"]) * 100.0

    metrics_labels = [
        "Style Fidelity\n(CSD Similarity)",
        "Style Alignment\n(DINO ViT)",
        "Structural Fidelity\n(LPIPS Error Reduction)",
        "Leakage Suppression\n(DCL Content Leakage Reduction)",
        "Content Preservation\n(CLIP-I Similarity)",
    ]
    pct_values = [pct_csd, pct_dino, pct_lpips, pct_dcl, pct_clip]

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    bars = ax.barh(metrics_labels, pct_values, color=["#38A169", "#38A169", "#38A169", "#38A169", "#4A5568"], height=0.55, edgecolor="#2D3748")

    ax.axvline(0, color="#2D3748", linewidth=1.2)
    ax.set_xlabel("Relative Improvement over Backbone RB-Modulation (%)", fontsize=11, fontweight="bold")
    ax.set_title("OrthoStyle Gains over Baseline Backbone (RB-Modulation)", fontsize=13, fontweight="bold", pad=12)

    for bar, val in zip(bars, pct_values):
        sign = "+" if val >= 0 else ""
        x_pos = val + 0.8 if val >= 0 else val - 0.8
        ha_align = "left" if val >= 0 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2, f"{sign}{val:.1f}%", va="center", ha=ha_align, fontsize=10, fontweight="bold", color="#1A202C")

    ax.set_xlim(-8, 32)
    ax.grid(axis="x", linestyle="--", alpha=0.7)
    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "figure3_head_to_head_vs_rbm.png"), dpi=300)
    plt.savefig(os.path.join(OUTPUT_DIR, "figure3_head_to_head_vs_rbm.pdf"))
    plt.close()
    print("[✔] Saved Figure 3: Head-to-Head vs RB-Modulation")


# -----------------------------------------------------------------------------
# Figure 4: Overall Harmonic Balance Score (F-Score)
# -----------------------------------------------------------------------------
def plot_figure4_balance_score():
    def norm_s(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-6)

    csd_n = norm_s(df_b["csd"].values)
    dino_n = norm_s(df_b["dino"].values)
    clip_n = norm_s(df_b["clip"].values)
    lpips_n = norm_s(100.0 - df_b["lpips"].values)
    dcl_n = norm_s(100.0 - df_b["dcl"].values)

    style_score = (csd_n + dino_n) / 2.0
    content_score = (clip_n + lpips_n + dcl_n) / 3.0

    h_score = 2.0 * (style_score * content_score) / (style_score + content_score + 1e-6) * 100.0

    df_score = pd.DataFrame({
        "Model": df_b["display_name"].values,
        "Harmonic_Score": h_score,
        "Style_Score": style_score * 100.0,
        "Content_Score": content_score * 100.0,
    }).sort_values(by="Harmonic_Score", ascending=True)

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    colors = ["#D92638" if m == "OrthoStyle (Ours)" else "#CBD5E0" for m in df_score["Model"]]

    bars = ax.barh(df_score["Model"], df_score["Harmonic_Score"], color=colors, height=0.55, edgecolor="#2D3748")

    ax.set_xlabel("Overall Style-Content Balance Score ($F_{balance}$, Higher is Better)", fontsize=11, fontweight="bold")
    ax.set_title("Overall Style-Content Balance Score (Harmonic Mean)", fontsize=13, fontweight="bold", pad=12)

    for bar, val, m in zip(bars, df_score["Harmonic_Score"], df_score["Model"]):
        fw = "bold" if m == "OrthoStyle (Ours)" else "normal"
        c = "#9B2C2C" if m == "OrthoStyle (Ours)" else "#2D3748"
        ax.text(val + 1.0, bar.get_y() + bar.get_height() / 2, f"{val:.1f}", va="center", ha="left", fontsize=9.5, fontweight=fw, color=c)

    ax.set_xlim(0, 80)
    ax.grid(axis="x", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "figure4_overall_balance_score.png"), dpi=300)
    plt.savefig(os.path.join(OUTPUT_DIR, "figure4_overall_balance_score.pdf"))
    plt.close()
    print("[✔] Saved Figure 4: Overall Harmonic Balance Score")


# -----------------------------------------------------------------------------
# Figure 5: Master Combined 2x2 Publication Figure
# -----------------------------------------------------------------------------
def plot_master_combined_figure():
    fig = plt.figure(figsize=(16, 13))

    # Panel A: Pareto Plot
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.axvspan(64, 73.5, ymin=0.55, ymax=1.0, color="#FED7D7", alpha=0.25, zorder=1)
    ax1.text(65, 49.5, "Content Collapse", fontsize=8.5, color="#9B2C2C", style="italic", weight="bold")
    ax1.axvspan(83, 89, ymin=0, ymax=0.35, color="#EDF2F7", alpha=0.35, zorder=1)
    ax1.text(83.2, 17, "Style Failure", fontsize=8.5, color="#4A5568", style="italic", weight="bold")

    sweep_x = df_sweeps["clip"].tolist()
    sweep_y = df_sweeps["csd"].tolist()
    ours_row = df_b[df_b["display_name"] == "OrthoStyle (Ours)"].iloc[0]
    sweep_x.append(ours_row["clip"])
    sweep_y.append(ours_row["csd"])
    sort_idx = np.argsort(sweep_x)
    ax1.plot(np.array(sweep_x)[sort_idx], np.array(sweep_y)[sort_idx], color="#E53E3E", linestyle="--", linewidth=2.0, alpha=0.85, zorder=3, label="OrthoStyle (varying $\\tau$)")

    for _, row in df_b.iterrows():
        name = row["display_name"]
        x, y = row["clip"], row["csd"]
        c = color_map.get(name, "#4A5568")
        if name == "OrthoStyle (Ours)":
            ax1.scatter(x, y, color=c, s=240, marker="*", edgecolor="#FFFFFF", linewidth=1.5, zorder=6, label=name)
            ax1.annotate(f" {name}\n (Pareto Sweet Spot)", (x, y), xytext=(x - 6.5, y + 2.5), fontsize=9.5, fontweight="bold", color="#9B2C2C", arrowprops=dict(arrowstyle="->", color="#9B2C2C", lw=1.3), zorder=7)
        else:
            ax1.scatter(x, y, color=c, s=100, marker="o", edgecolor="#2D3748", linewidth=0.8, zorder=4)
            offset = (1.0, -1.2)
            if name == "InstantStyle": offset = (-7.0, 1.2)
            elif name == "DiffuseIT": offset = (1.0, 1.0)
            elif name == "ITOC": offset = (-3.5, 1.2)
            elif name == "RB-Modulation": offset = (-8.0, -2.2)
            elif name == "AttenST": offset = (-5.5, -2.0)
            elif name == "StyleID": offset = (1.2, -0.2)
            ax1.annotate(name, (x, y), xytext=(x + offset[0], y + offset[1]), fontsize=8.5, color="#2D3748", zorder=5)

    ax1.set_xlabel("Content Preservation (CLIP-I $\\uparrow$)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Style Fidelity (CSD $\\uparrow$)", fontsize=11, fontweight="bold")
    ax1.set_title("(a) Style-Content Pareto Frontier", fontsize=12.5, fontweight="bold", pad=8)
    ax1.set_xlim(64, 90)
    ax1.set_ylim(10, 52)
    ax1.grid(True)
    ax1.legend(loc="lower left", fontsize=8.5)

    # Panel B: Head to head vs RBM
    ax2 = fig.add_subplot(2, 2, 2)
    rbm = df_b[df_b["display_name"] == "RB-Modulation"].iloc[0]
    pct_csd = ((ours_row["csd"] - rbm["csd"]) / rbm["csd"]) * 100.0
    pct_dino = ((ours_row["dino"] - rbm["dino"]) / rbm["dino"]) * 100.0
    pct_lpips = ((rbm["lpips"] - ours_row["lpips"]) / rbm["lpips"]) * 100.0
    pct_dcl = ((rbm["dcl"] - ours_row["dcl"]) / rbm["dcl"]) * 100.0
    pct_clip = ((ours_row["clip"] - rbm["clip"]) / rbm["clip"]) * 100.0

    metrics_labels = [
        "CSD Style Sim. $\\uparrow$",
        "DINO Style Align. $\\uparrow$",
        "LPIPS Error Reduc. $\\downarrow$",
        "DCL Leakage Reduc. $\\downarrow$",
        "CLIP-I Content $\\uparrow$",
    ]
    pct_values = [pct_csd, pct_dino, pct_lpips, pct_dcl, pct_clip]
    bars2 = ax2.barh(metrics_labels, pct_values, color=["#38A169", "#38A169", "#38A169", "#38A169", "#4A5568"], height=0.55, edgecolor="#2D3748")
    ax2.axvline(0, color="#2D3748", linewidth=1.0)
    ax2.set_xlabel("Relative Gain over RB-Modulation (%)", fontsize=11, fontweight="bold")
    ax2.set_title("(b) Relative Improvement over Backbone", fontsize=12.5, fontweight="bold", pad=8)
    for bar, val in zip(bars2, pct_values):
        sign = "+" if val >= 0 else ""
        x_pos = val + 0.8 if val >= 0 else val - 0.8
        ha_align = "left" if val >= 0 else "right"
        ax2.text(x_pos, bar.get_y() + bar.get_height() / 2, f"{sign}{val:.1f}%", va="center", ha=ha_align, fontsize=9.5, fontweight="bold")
    ax2.set_xlim(-8, 32)
    ax2.grid(axis="x", linestyle="--", alpha=0.7)
    ax2.invert_yaxis()

    # Panel C: Radar Chart
    ax3 = fig.add_subplot(2, 2, 3, polar=True)
    categories = ["CSD (Style)", "DINO (Align)", "CLIP-I (Cont)", "100 - LPIPS", "100 - DCL"]
    N = len(categories)
    plot_models = ["OrthoStyle (Ours)", "StyleID", "AttenST", "InstantStyle", "RB-Modulation"]
    raw_matrix = []
    for m in plot_models:
        r = df_b[df_b["display_name"] == m].iloc[0]
        raw_matrix.append([r["csd"], r["dino"], r["clip"], 100.0 - r["lpips"], 100.0 - r["dcl"]])
    raw_matrix = np.array(raw_matrix)
    norm_matrix = np.zeros_like(raw_matrix)
    for col in range(N):
        c_min, c_max = raw_matrix[:, col].min(), raw_matrix[:, col].max()
        norm_matrix[:, col] = 0.25 + 0.75 * (raw_matrix[:, col] - c_min) / (c_max - c_min + 1e-6)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    ax3.set_theta_offset(np.pi / 2)
    ax3.set_theta_direction(-1)
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(categories, size=9.5, weight="bold")
    ax3.set_yticklabels([])
    for idx, m in enumerate(plot_models):
        values = norm_matrix[idx].tolist()
        values += values[:1]
        c = color_map.get(m, "#4A5568")
        if m == "OrthoStyle (Ours)":
            ax3.plot(angles, values, linewidth=2.6, color=c, label=m, zorder=6)
            ax3.fill(angles, values, color=c, alpha=0.22, zorder=5)
        elif m == "RB-Modulation":
            ax3.plot(angles, values, linewidth=1.5, linestyle="--", color=c, label=m, zorder=4)
        else:
            ax3.plot(angles, values, linewidth=1.3, linestyle="-", color=c, label=m, alpha=0.75, zorder=3)
    ax3.set_title("(c) Multi-Metric Holistic Profile", size=12.5, weight="bold", y=1.08)
    ax3.legend(loc="upper right", bbox_to_anchor=(1.35, 1.05), fontsize=8.5)

    # Panel D: Harmonic Balance Score
    ax4 = fig.add_subplot(2, 2, 4)
    def norm_s(x): return (x - x.min()) / (x.max() - x.min() + 1e-6)
    csd_n = norm_s(df_b["csd"].values)
    dino_n = norm_s(df_b["dino"].values)
    clip_n = norm_s(df_b["clip"].values)
    lpips_n = norm_s(100.0 - df_b["lpips"].values)
    dcl_n = norm_s(100.0 - df_b["dcl"].values)
    style_score = (csd_n + dino_n) / 2.0
    content_score = (clip_n + lpips_n + dcl_n) / 3.0
    h_score = 2.0 * (style_score * content_score) / (style_score + content_score + 1e-6) * 100.0
    df_score = pd.DataFrame({
        "Model": df_b["display_name"].values,
        "Harmonic_Score": h_score,
    }).sort_values(by="Harmonic_Score", ascending=True)
    colors4 = ["#D92638" if m == "OrthoStyle (Ours)" else "#CBD5E0" for m in df_score["Model"]]
    bars4 = ax4.barh(df_score["Model"], df_score["Harmonic_Score"], color=colors4, height=0.55, edgecolor="#2D3748")
    ax4.set_xlabel("Overall Style-Content Balance ($F_{balance}$)", fontsize=11, fontweight="bold")
    ax4.set_title("(d) Overall Harmonic Trade-off Score", fontsize=12.5, fontweight="bold", pad=8)
    for bar, val, m in zip(bars4, df_score["Harmonic_Score"], df_score["Model"]):
        fw = "bold" if m == "OrthoStyle (Ours)" else "normal"
        c = "#9B2C2C" if m == "OrthoStyle (Ours)" else "#2D3748"
        ax4.text(val + 1.0, bar.get_y() + bar.get_height() / 2, f"{val:.1f}", va="center", ha="left", fontsize=9.5, fontweight=fw, color=c)
    ax4.set_xlim(0, 80)
    ax4.grid(axis="x", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "figure_master_paper_comparison.png"), dpi=300)
    plt.savefig(os.path.join(OUTPUT_DIR, "figure_master_paper_comparison.pdf"))
    plt.close()
    print("[✔] Saved Master Combined Figure: figure_master_paper_comparison.png png & .pdf")


def main():
    print("[*] Generating publication-quality visualization figures...")
    plot_figure1_pareto()
    plot_figure2_radar()
    plot_figure3_head_to_head()
    plot_figure4_balance_score()
    plot_master_combined_figure()
    print(f"\n[✔] All figures successfully generated and saved to '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()
