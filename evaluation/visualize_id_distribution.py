#!/usr/bin/env python3
"""
Visualize videos-per-identity distributions for DogReID-1553.

Generates:
1. Overall dataset video-per-identity distribution histogram + boxplot.
2. Breakdowns across splits (Closed-Set: Train, Query, Gallery; Open-Set: Train, Query, Gallery).
3. Boxplot comparison across all subsets.
4. Summary statistics table (CSV and console print).
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Plot videos per identity distribution.")
    parser.add_argument(
        "--splits_csv",
        type=str,
        default="splits.csv",
        help="Path to splits.csv (default: splits.csv in project root)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation/figs",
        help="Directory to save figures (default: evaluation/figs)"
    )
    return parser.parse_args()


def compute_stats(series, label=""):
    return {
        "Subset": label,
        "Unique IDs": len(series),
        "Total Videos": int(series.sum()),
        "Mean": f"{series.mean():.2f}",
        "Std": f"{series.std():.2f}",
        "Min": int(series.min()),
        "25%": f"{series.quantile(0.25):.1f}",
        "Median": f"{series.median():.1f}",
        "75%": f"{series.quantile(0.75):.1f}",
        "Max": int(series.max()),
    }


def main():
    args = parse_args()
    csv_path = Path(args.splits_csv)
    if not csv_path.exists():
        project_root = Path(__file__).resolve().parent.parent
        csv_path = project_root / args.splits_csv

    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find splits file at {csv_path}")

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        project_root = Path(__file__).resolve().parent.parent
        out_dir = project_root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} video records across {df['DOG_ID'].nunique()} dogs from {csv_path}")

    # Pure matplotlib styling
    plt.rcParams.update({
        "font.size": 11,
        "axes.edgecolor": "#333333",
        "axes.grid": True,
        "grid.color": "#e0e0e0",
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
        "figure.autolayout": True
    })

    stats_list = []

    # 1. Overall videos per ID
    overall_counts = df.groupby("DOG_ID")["VIDEO_ID"].count()
    stats_list.append(compute_stats(overall_counts, "Overall Dataset"))

    # 2. Closed-Set splits
    closed_train = df[df["SPLIT_CLOSED_SET"] == "train"].groupby("DOG_ID")["VIDEO_ID"].count()
    closed_query = df[df["SPLIT_CLOSED_SET"] == "query"].groupby("DOG_ID")["VIDEO_ID"].count()
    closed_gallery = df[df["SPLIT_CLOSED_SET"] == "gallery"].groupby("DOG_ID")["VIDEO_ID"].count()

    stats_list.append(compute_stats(closed_train, "Closed-Set Train"))
    stats_list.append(compute_stats(closed_query, "Closed-Set Query"))
    stats_list.append(compute_stats(closed_gallery, "Closed-Set Gallery"))

    # 3. Open-Set splits
    open_train = df[df["SPLIT_OPEN_SET"] == "train"].groupby("DOG_ID")["VIDEO_ID"].count()
    open_query = df[df["SPLIT_OPEN_SET"] == "query"].groupby("DOG_ID")["VIDEO_ID"].count()
    open_gallery = df[df["SPLIT_OPEN_SET"] == "gallery"].groupby("DOG_ID")["VIDEO_ID"].count()

    stats_list.append(compute_stats(open_train, "Open-Set Train"))
    stats_list.append(compute_stats(open_query, "Open-Set Query"))
    stats_list.append(compute_stats(open_gallery, "Open-Set Gallery"))

    stats_df = pd.DataFrame(stats_list)
    print("\n" + "=" * 80)
    print("VIDEOS PER IDENTITY DISTRIBUTION SUMMARY")
    print("=" * 80)
    print(stats_df.to_string(index=False))
    print("=" * 80 + "\n")

    # Save summary stats to CSV
    stats_csv_path = out_dir / "videos_per_identity_summary.csv"
    stats_df.to_csv(stats_csv_path, index=False)
    print(f"Saved stats table to: {stats_csv_path}")

    # -------------------------------------------------------------
    # Plot 1: Overall Dataset Distribution (Histogram + Boxplot)
    # -------------------------------------------------------------
    fig, (ax_hist, ax_box) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [0.75, 0.25]}
    )

    max_val = int(overall_counts.max())
    bins = np.arange(0.5, max_val + 1.5, 1)

    ax_hist.hist(
        overall_counts,
        bins=bins,
        color="#3470a3",
        edgecolor="black",
        linewidth=1.0,
        rwidth=0.85
    )
    mean_v = float(overall_counts.mean())
    med_v = float(overall_counts.median())
    ax_hist.axvline(mean_v, color="#e74c3c", linestyle="--", linewidth=2, label=f"Mean: {mean_v:.2f}")
    ax_hist.axvline(med_v, color="#2ecc71", linestyle="-", linewidth=2, label=f"Median: {med_v:.1f}")
    ax_hist.set_title("Overall Videos per Dog Identity (DogReID-1553)", fontsize=14, fontweight="bold")
    ax_hist.set_ylabel("Number of Identities")
    ax_hist.legend(fontsize=11)

    ax_box.boxplot(
        overall_counts,
        vert=False,
        patch_artist=True,
        boxprops=dict(facecolor="#729fcf", color="black"),
        medianprops=dict(color="red", linewidth=2),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        flierprops=dict(marker="o", color="black", alpha=0.5, markersize=4)
    )
    ax_box.set_xlabel("Number of Videos per Identity")
    ax_box.set_yticks([])

    out_overall = out_dir / "videos_per_id_overall.png"
    plt.savefig(out_overall, dpi=300)
    plt.close()
    print(f"Saved plot: {out_overall}")

    # -------------------------------------------------------------
    # Plot 2: Closed-Set vs Open-Set Comparison (Subplots)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=True)

    split_configs = [
        # Row 1: Closed-Set
        (axes[0, 0], closed_train, "Closed-Set: Train", "#2b5c8f"),
        (axes[0, 1], closed_query, "Closed-Set: Query", "#d95f02"),
        (axes[0, 2], closed_gallery, "Closed-Set: Gallery", "#7570b3"),
        # Row 2: Open-Set
        (axes[1, 0], open_train, "Open-Set: Train", "#1b9e77"),
        (axes[1, 1], open_query, "Open-Set: Query", "#e7298a"),
        (axes[1, 2], open_gallery, "Open-Set: Gallery", "#66a61e"),
    ]

    for ax, counts, title, color in split_configs:
        curr_max = int(counts.max())
        curr_bins = np.arange(0.5, curr_max + 1.5, 1)
        ax.hist(
            counts,
            bins=curr_bins,
            color=color,
            edgecolor="black",
            linewidth=1.0,
            rwidth=0.85,
            alpha=0.85
        )
        m_val = float(counts.mean())
        md_val = float(counts.median())
        ax.axvline(m_val, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {m_val:.2f}")
        ax.axvline(md_val, color="darkgreen", linestyle="-", linewidth=1.5, label=f"Med: {md_val:.1f}")
        ax.set_title(f"{title} (n={len(counts)} dogs)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Videos per Dog")
        ax.set_ylabel("Count of Dogs")
        ax.legend(fontsize=10, loc="upper right")

    fig.suptitle("Videos per Identity Distribution Across Closed-Set and Open-Set Splits", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_splits = out_dir / "videos_per_id_splits_breakdown.png"
    plt.savefig(out_splits, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out_splits}")

    # -------------------------------------------------------------
    # Plot 3: Boxplot Comparison across all subsets
    # -------------------------------------------------------------
    plt.figure(figsize=(11, 6))
    subsets_data = [
        ("Closed Train", closed_train),
        ("Closed Query", closed_query),
        ("Closed Gallery", closed_gallery),
        ("Open Train", open_train),
        ("Open Query", open_query),
        ("Open Gallery", open_gallery),
        ("Overall", overall_counts),
    ]
    labels = [s[0] for s in subsets_data]
    data_to_plot = [s[1] for s in subsets_data]
    colors = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77", "#e7298a", "#66a61e", "#3470a3"]

    box = plt.boxplot(
        data_to_plot,
        tick_labels=labels,
        patch_artist=True,
        showmeans=True,
        meanprops={"marker": "^", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 8},
        medianprops={"color": "red", "linewidth": 2},
        whiskerprops={"color": "#333333"},
        capprops={"color": "#333333"},
        flierprops={"marker": "o", "color": "gray", "alpha": 0.4, "markersize": 4}
    )

    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    plt.title("Videos per Identity Comparison Across Subsets (▲ = Mean, Red Bar = Median)", fontsize=13, fontweight="bold")
    plt.xlabel("Dataset Split / Subset")
    plt.ylabel("Number of Videos")
    plt.xticks(rotation=15)
    plt.tight_layout()

    out_box = out_dir / "videos_per_id_subsets_boxplot.png"
    plt.savefig(out_box, dpi=300)
    plt.close()
    print(f"Saved plot: {out_box}")


if __name__ == "__main__":
    main()
