#!/usr/bin/env python3
"""
Visualize the overall videos-per-identity distribution for DogReID-1553.

Generates:
1. Overall dataset histogram + boxplot.
2. Summary statistics table (CSV and console print).
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot videos per identity distribution."
    )
    parser.add_argument(
        "--splits_csv",
        type=str,
        default="splits.csv",
        help="Path to splits.csv (default: splits.csv in project root)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation/figures",
        help="Directory to save figures (default: evaluation/figures)"
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
    project_root = Path(__file__).resolve().parent.parent

    if not csv_path.exists():
        csv_path = project_root / args.splits_csv

    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find splits file at {csv_path}")

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    print(
        f"Loaded {len(df)} video records across "
        f"{df['DOG_ID'].nunique()} dogs from {csv_path}"
    )

    # -------------------------------------------------------------
    # Plot styling
    # -------------------------------------------------------------
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 32,
        "axes.labelsize": 32,
        "axes.titlesize": 32,
        "xtick.labelsize": 32,
        "ytick.labelsize": 32,
        "legend.fontsize": 32,
        "axes.linewidth": 1.5,
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.25,
    })

    # -------------------------------------------------------------
    # Overall videos per identity and summary statistics
    # -------------------------------------------------------------
    overall_counts = df.groupby("DOG_ID")["VIDEO_ID"].count()

    stats_df = pd.DataFrame([
        compute_stats(overall_counts, "Overall Dataset")
    ])

    print("\n" + "=" * 90)
    print("VIDEOS PER IDENTITY DISTRIBUTION SUMMARY")
    print("=" * 90)
    print(stats_df.to_string(index=False))
    print("=" * 90 + "\n")

    stats_csv_path = out_dir / "videos_per_identity_summary.csv"
    stats_df.to_csv(stats_csv_path, index=False)
    print(f"Saved statistics: {stats_csv_path}")

    # -------------------------------------------------------------
    # Overall Dataset Distribution (Histogram + Boxplot)
    # -------------------------------------------------------------
    fig, (ax_hist, ax_box) = plt.subplots(
        2,
        1,
        figsize=(13, 10),
        sharex=True,
        gridspec_kw={
            "height_ratios": [3.2, 0.8],
            "hspace": 0.08
        }
    )

    max_val = int(overall_counts.max())
    bins = np.arange(0.5, max_val + 1.5, 1)

    # Histogram
    ax_hist.hist(
        overall_counts,
        bins=bins,
        color="#4778A8",
        edgecolor="white",
        linewidth=1.0,
        rwidth=0.85,
        alpha=0.95
    )

    mean_v = float(overall_counts.mean())
    med_v = float(overall_counts.median())

    ax_hist.axvline(
        mean_v,
        color="#D65A4A",
        linestyle="--",
        linewidth=3.0,
        label=f"Mean: {mean_v:.2f}",
        zorder=3
    )

    ax_hist.axvline(
        med_v,
        color="#39845E",
        linestyle="-",
        linewidth=3.0,
        label=f"Median: {med_v:.1f}",
        zorder=3
    )

    ax_hist.set_ylabel("Number of Identities", labelpad=15)

    ax_hist.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        edgecolor="#DDDDDD",
        borderpad=0.8
    )

    # Clean histogram axes
    ax_hist.set_axisbelow(True)
    ax_hist.grid(
        axis="y",
        color="#D9DDE3",
        linestyle="--",
        linewidth=1.0,
        alpha=0.8
    )
    ax_hist.grid(axis="x", visible=False)

    ax_hist.spines["top"].set_visible(False)
    ax_hist.spines["right"].set_visible(False)

    # Boxplot
    ax_box.boxplot(
        overall_counts,
        vert=False,
        patch_artist=True,
        widths=0.55,
        boxprops=dict(
            facecolor="#91B0CD",
            edgecolor="#333333",
            linewidth=1.5
        ),
        medianprops=dict(
            color="#D65A4A",
            linewidth=3.0
        ),
        whiskerprops=dict(
            color="#333333",
            linewidth=1.6
        ),
        capprops=dict(
            color="#333333",
            linewidth=1.6
        ),
        flierprops=dict(
            marker="o",
            markerfacecolor="#555555",
            markeredgecolor="white",
            alpha=0.55,
            markersize=6
        )
    )

    ax_box.set_xlabel("Number of Videos per Identity", labelpad=15)
    ax_box.set_yticks([])

    ax_box.grid(
        axis="x",
        color="#D9DDE3",
        linestyle="--",
        linewidth=1.0,
        alpha=0.8
    )
    ax_box.grid(axis="y", visible=False)

    for spine in ["top", "right", "left"]:
        ax_box.spines[spine].set_visible(False)


    # Save
    out_overall = out_dir / "videos_per_id_overall.pdf"
    fig.savefig(out_overall, format="pdf")
    plt.close(fig)

    print(f"Saved plot: {out_overall}")


if __name__ == "__main__":
    main()