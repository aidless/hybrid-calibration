"""
Visualization module for Hybrid Tree-Ensemble + LLM Calibration Study.

Produces publication-quality figures:
  1. Reliability diagrams (multiple models overlaid)
  2. ECE bar chart with error bars
  3. ECE vs Accuracy scatter
  4. Pairwise significance heatmap
  5. Calibration degradation radar
"""

import os
import json
import numpy as np
from typing import Dict, List, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from config import FIGURES_DIR

# ============================================================
# Style
# ============================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# Color palette for model families
PALETTE = {
    "RF-Tabular": "#27ae60",
    "XGB-Tabular": "#2ecc71",
    "LGB-Tabular": "#a8e6cf",
    "RF-Embed": "#3498db",
    "XGB-Embed": "#5dade2",
    "LGB-Embed": "#aed6f1",
    "RF-Hybrid": "#8e44ad",
    "XGB-Hybrid": "#a569bd",
    "LGB-Hybrid": "#d2b4de",
    "MLP-Embed": "#e74c3c",
    "RF-Hybrid-Platt": "#f39c12",
    "RF-Hybrid-Isotonic": "#f1c40f",
    "MLP-Embed-Platt": "#e67e22",
    "MLP-Embed-Isotonic": "#f5b041",
}

FAMILY_COLORS = {
    "Tabular": "#27ae60",
    "Embedding": "#3498db",
    "Hybrid": "#8e44ad",
    "MLP": "#e74c3c",
    "Calibrated": "#f39c12",
}


def to_color(name: str) -> str:
    """Get color for a model name."""
    return PALETTE.get(name, "#7f8c8d")


def to_family(name: str) -> str:
    """Infer model family from name."""
    if "Calibrated" in name or "Platt" in name or "Isotonic" in name:
        return "Calibrated"
    if "MLP" in name:
        return "MLP"
    if "Hybrid" in name:
        return "Hybrid"
    if "Embed" in name:
        return "Embedding"
    return "Tabular"


# ============================================================
# Figure 1: Reliability Diagrams
# ============================================================

def plot_reliability_diagrams(
    results: Dict,
    dataset_name: str = "",
    max_models: int = 6,
    save: bool = True,
):
    """
    Reliability diagrams overlaid for top models.

    Each subplot: predicted probability vs observed accuracy.
    Diagonal = perfect calibration.
    """
    # Collect reliability data from last seed
    seed_results = results.get("seed_results", [])
    if not seed_results:
        print("[Viz] No seed results found")
        return

    last_seed = seed_results[-1]
    reliability_data = {}

    for r in last_seed["metrics"]:
        name = r["model_name"]
        rel = r["metrics"].get("reliability", {})
        if rel.get("bin_conf") is not None:
            reliability_data[name] = rel

    # Select models to show (limit for clarity)
    # Show: best Tabular, best Embed, best Hybrid, MLP-Embed, best Calibrated
    show_models = _select_display_models(list(reliability_data.keys()), max_models)

    n_cols = min(3, len(show_models))
    n_rows = int(np.ceil(len(show_models) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
    if n_rows * n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, name in enumerate(show_models):
        ax = axes[idx]
        rel = reliability_data.get(name, {})
        if not rel:
            continue

        bin_conf = np.array(rel["bin_conf"])
        bin_acc = np.array(rel["bin_acc"])
        bin_count = np.array(rel["bin_count"])
        bin_edges = np.array(rel["bin_edges"])

        # Bar width proportional to sample count
        bar_width = 0.08
        valid = ~np.isnan(bin_acc)

        # Diagonal
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5, label="Perfect")

        # Reliability bars
        for i in range(len(bin_conf)):
            if valid[i] and bin_count[i] > 0:
                alpha = 0.3 + 0.7 * (bin_count[i] / max(bin_count))
                ax.bar(
                    bin_conf[i], bin_acc[i],
                    width=bar_width, alpha=alpha,
                    color=to_color(name), edgecolor="white", linewidth=0.3,
                )

        # Gap line (miscalibration area)
        if np.any(valid):
            gap_x = []
            gap_y = []
            for i in range(len(bin_conf)):
                if valid[i] and bin_count[i] > 0:
                    gap_x.extend([bin_conf[i], bin_conf[i], None])
                    gap_y.extend([bin_conf[i], bin_acc[i], None])
            ax.plot(gap_x, gap_y, color=to_color(name), linewidth=1.2, alpha=0.7)

        # ECE annotation
        ece_val = _get_ece_for_model(results, name)
        ax.text(
            0.95, 0.05,
            f"ECE={ece_val:.4f}" if ece_val is not None else "",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
        )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Observed Accuracy")
        ax.set_title(name, fontweight="bold")
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for idx in range(len(show_models), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f"Reliability Diagrams — {dataset_name.upper()}", fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()

    if save:
        path = os.path.join(FIGURES_DIR, f"reliability_{dataset_name}.png")
        fig.savefig(path, facecolor="white")
        print(f"[Viz] Saved reliability diagram: {path}")
    plt.close(fig)


# ============================================================
# Figure 2: ECE Bar Chart
# ============================================================

def plot_ece_bars(
    results: Dict,
    dataset_name: str = "",
    save: bool = True,
):
    """Bar chart of ECE (mean ± std) across models."""
    aggregated = results.get("aggregated", {})
    if not aggregated:
        print("[Viz] No aggregated results")
        return

    # Sort by ECE (ascending = better calibration)
    sorted_models = sorted(aggregated.items(), key=lambda x: x[1]["ece_mean"])

    names = [m[0] for m in sorted_models]
    ece_means = [m[1]["ece_mean"] for m in sorted_models]
    ece_stds = [m[1]["ece_std"] for m in sorted_models]
    colors = [to_color(n) for n in names]

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(range(len(names)), ece_means, yerr=ece_stds,
                  color=colors, edgecolor="white", linewidth=0.5,
                  capsize=3, error_kw={"linewidth": 1})

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("ECE (↓ better)")
    ax.set_title(f"Expected Calibration Error by Model — {dataset_name.upper()}")
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.5)

    # Annotate values
    for bar, mean in zip(bars, ece_means):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
            f"{mean:.4f}", ha="center", va="bottom", fontsize=8,
        )

    fig.tight_layout()

    if save:
        path = os.path.join(FIGURES_DIR, f"ece_bars_{dataset_name}.png")
        fig.savefig(path, facecolor="white")
        print(f"[Viz] Saved ECE bar chart: {path}")
    plt.close(fig)


# ============================================================
# Figure 3: ECE vs Accuracy Scatter
# ============================================================

def plot_ece_vs_accuracy(
    results: Dict,
    dataset_name: str = "",
    save: bool = True,
):
    """Scatter plot: ECE (x) vs Accuracy (y) for each model."""
    aggregated = results.get("aggregated", {})
    if not aggregated:
        return

    fig, ax = plt.subplots(figsize=(9, 7))

    for name, metrics in aggregated.items():
        family = to_family(name)
        ax.errorbar(
            metrics["ece_mean"], metrics["accuracy_mean"],
            xerr=metrics["ece_std"], yerr=metrics["accuracy_std"],
            marker="o", markersize=10,
            color=FAMILY_COLORS.get(family, "#7f8c8d"),
            label=name,
            capsize=2, linewidth=1,
        )

    ax.set_xlabel("ECE (↓ better)")
    ax.set_ylabel("Accuracy (↑ better)")
    ax.set_title(f"Calibration vs Accuracy Trade-off — {dataset_name.upper()}")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if save:
        path = os.path.join(FIGURES_DIR, f"ece_vs_acc_{dataset_name}.png")
        fig.savefig(path, facecolor="white")
        print(f"[Viz] Saved ECE vs Accuracy scatter: {path}")
    plt.close(fig)


# ============================================================
# Figure 4: Pairwise Significance Heatmap
# ============================================================

def plot_significance_heatmap(
    results: Dict,
    dataset_name: str = "",
    metric: str = "ece",
    save: bool = True,
):
    """Heatmap of pairwise statistical significance."""
    pairwise = results.get("pairwise_tests_ece", [])
    if not pairwise:
        return

    # Build matrix
    all_models = sorted(set(
        [pt["model_a"] for pt in pairwise] + [pt["model_b"] for pt in pairwise]
    ))
    n = len(all_models)
    model_to_idx = {m: i for i, m in enumerate(all_models)}

    # Matrix: -log10(p) signed by direction (A better than B = negative diff = blue)
    matrix = np.zeros((n, n))
    annot = np.empty((n, n), dtype=object)

    for pt in pairwise:
        i = model_to_idx[pt["model_a"]]
        j = model_to_idx[pt["model_b"]]
        # mean_diff = A - B. Negative = A has lower ECE = better.
        signed_logp = -np.sign(pt["mean_diff"]) * (-np.log10(max(pt["p_value_corrected"], 1e-10)))
        matrix[i, j] = signed_logp
        matrix[j, i] = -signed_logp
        sig_marker = "*" if pt["significant_corrected"] else ""
        annot[i, j] = f"{pt['mean_diff']:.3f}{sig_marker}"
        annot[j, i] = f"{-pt['mean_diff']:.3f}{sig_marker}"

    # Mask lower triangle
    mask = np.tril(np.ones_like(matrix, dtype=bool), k=-1)

    fig, ax = plt.subplots(figsize=(max(10, n * 0.8), max(8, n * 0.7)))
    sns.heatmap(
        matrix, mask=mask,
        annot=annot, fmt="",
        cmap="RdBu_r", center=0,
        xticklabels=all_models, yticklabels=all_models,
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Signed -log10(p) (blue = row better)"},
        ax=ax,
    )
    ax.set_title(f"Pairwise ECE Significance — {dataset_name.upper()}\n"
                 f"(* = significant after Bonferroni)")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    fig.tight_layout()

    if save:
        path = os.path.join(FIGURES_DIR, f"significance_heatmap_{dataset_name}.png")
        fig.savefig(path, facecolor="white")
        print(f"[Viz] Saved significance heatmap: {path}")
    plt.close(fig)


# ============================================================
# Figure 5: Calibration Degradation Summary
# ============================================================

def plot_summary(
    results: Dict,
    dataset_name: str = "",
    save: bool = True,
):
    """Combined summary figure: ECE bars + accuracy scatter subplots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    aggregated = results.get("aggregated", {})
    if not aggregated:
        return

    # --- Left: ECE bars ---
    sorted_models_ece = sorted(aggregated.items(), key=lambda x: x[1]["ece_mean"])
    names_ece = [m[0] for m in sorted_models_ece]
    ece_means = [m[1]["ece_mean"] for m in sorted_models_ece]
    ece_stds = [m[1]["ece_std"] for m in sorted_models_ece]

    # Group bars by family color
    bar_colors = []
    for n in names_ece:
        family = to_family(n)
        bar_colors.append(FAMILY_COLORS.get(family, "#7f8c8d"))

    bars = ax1.bar(range(len(names_ece)), ece_means, yerr=ece_stds,
                   color=bar_colors, edgecolor="white", linewidth=0.5,
                   capsize=3)
    ax1.set_xticks(range(len(names_ece)))
    ax1.set_xticklabels(names_ece, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("ECE (↓ better)")
    ax1.set_title("Expected Calibration Error")
    ax1.grid(axis="y", alpha=0.3)

    # Legend for families
    from matplotlib.patches import Patch
    legend_patches = [Patch(color=c, label=f) for f, c in FAMILY_COLORS.items()]
    ax1.legend(handles=legend_patches, fontsize=8, loc="upper left")

    # --- Right: ECE vs Accuracy ---
    for name, metrics in aggregated.items():
        family = to_family(name)
        ax2.errorbar(
            metrics["ece_mean"], metrics["accuracy_mean"],
            xerr=metrics["ece_std"], yerr=metrics["accuracy_std"],
            marker="o", markersize=8,
            color=FAMILY_COLORS.get(family, "#7f8c8d"),
            label=name,
            capsize=2, linewidth=0.8,
        )
        ax2.annotate(
            name, (metrics["ece_mean"], metrics["accuracy_mean"]),
            textcoords="offset points", xytext=(5, 5),
            fontsize=7, alpha=0.8,
        )

    ax2.set_xlabel("ECE (↓ better)")
    ax2.set_ylabel("Accuracy (↑ better)")
    ax2.set_title("Calibration vs Accuracy")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"Hybrid Calibration Study — {dataset_name.upper()}",
                 fontsize=16, fontweight="bold")
    fig.tight_layout()

    if save:
        path = os.path.join(FIGURES_DIR, f"summary_{dataset_name}.png")
        fig.savefig(path, facecolor="white")
        print(f"[Viz] Saved summary figure: {path}")
    plt.close(fig)


# ============================================================
# Generate All Figures
# ============================================================

def generate_all_figures(results_file: str):
    """Generate all figures from a saved results JSON file."""
    with open(results_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    dataset_name = results.get("dataset", "unknown")

    print(f"\n[Viz] Generating figures for {dataset_name} ...")
    plot_reliability_diagrams(results, dataset_name)
    plot_ece_bars(results, dataset_name)
    plot_ece_vs_accuracy(results, dataset_name)
    plot_significance_heatmap(results, dataset_name)
    plot_summary(results, dataset_name)
    print(f"[Viz] All figures saved to {FIGURES_DIR}/")


# ============================================================
# Helpers
# ============================================================

def _get_ece_for_model(results: Dict, model_name: str) -> Optional[float]:
    """Extract mean ECE for a model from aggregated results."""
    aggregated = results.get("aggregated", {})
    if model_name in aggregated:
        return aggregated[model_name]["ece_mean"]
    return None


def _select_display_models(all_models: List[str], max_models: int) -> List[str]:
    """Select representative models for display."""
    priority = []
    # Pick one per family
    families = {}
    for name in all_models:
        fam = to_family(name)
        if fam not in families:
            families[fam] = name
    priority.extend(families.values())
    # Add remaining
    for name in all_models:
        if name not in priority:
            priority.append(name)
    return priority[:max_models]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("results_file", help="Path to results JSON file")
    args = parser.parse_args()
    generate_all_figures(args.results_file)
