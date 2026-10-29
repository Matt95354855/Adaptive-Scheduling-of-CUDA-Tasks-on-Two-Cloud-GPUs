#!/usr/bin/env python3
"""Render the paired worker-pool control after real GPU measurements."""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with Path(f"{args.prefix}-seeds.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    summary = json.loads(Path(f"{args.prefix}-summary.json").read_text())
    if len(rows) != 8 or summary["seed_count"] != 8:
        raise ValueError("Expected eight paired seeds before plotting")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), constrained_layout=True)
    fig.patch.set_facecolor("#f6f8fb")
    for ax in axes:
        ax.set_facecolor("white")
        ax.grid(axis="y", color="#dce3eb", linewidth=.7)
        ax.set_axisbelow(True)

    gains = [float(row["warm_wall_gain_pct"]) for row in rows]
    colors = ["#137c74" if gain > 0 else "#c45c4b" for gain in gains]
    axes[0].bar(range(8), gains, color=colors, width=.66)
    axes[0].axhline(0, color="#32445b", linewidth=1)
    axes[0].set_xticks(range(8), [row["seed"] for row in rows], rotation=45)
    axes[0].set_ylabel("Gain de temps mural du pool chaud (%)")
    axes[0].set_title("Une paire par seed", loc="left", fontweight="bold")
    axes[0].text(.02, .98,
                 f"Médiane : {summary['wall_gain_median_pct']:+.1f} %\n"
                 f"Seeds gagnées : {summary['warm_faster_seeds']}/8",
                 transform=axes[0].transAxes, va="top", fontsize=9,
                 bbox={"facecolor": "white", "edgecolor": "#dce3eb", "alpha": .9})

    arms = ("cold", "warm")
    labels = ["Froid\nGPU 0", "Froid\nGPU 1", "Chaud\nGPU 0", "Chaud\nGPU 1"]
    values = [summary["occupancy_descriptive"][arm][str(gpu)]["mean_sm_pct"]
              for arm in arms for gpu in (0, 1)]
    axes[1].bar(range(4), values, color=["#7389a3", "#95a8bd", "#137c74", "#58a99d"], width=.65)
    axes[1].axhline(70, color="#bd7832", linestyle="--", linewidth=1.2, label="Objectif pratique 70 %")
    axes[1].set_xticks(range(4), labels)
    axes[1].set_ylim(0, 105)
    axes[1].set_ylabel("SM moyen des relevés à 1 s (%)")
    axes[1].set_title("Occupation pendant les blocs", loc="left", fontweight="bold")
    axes[1].legend(frameon=False, loc="upper right", fontsize=8)
    for index, value in enumerate(values):
        axes[1].text(index, value + 2, f"{value:.1f}", ha="center", fontsize=9)

    fig.suptitle("Workers conservés entre quatre lots : coût complet et utilisation",
                 x=.03, ha="left", fontsize=14, fontweight="bold", color="#172c47")
    fig.text(.03, .01, "Contrôle secondaire descriptif · 8 seeds appariées · deux RTX PRO 4000 Blackwell",
             color="#556b82", fontsize=8)
    for suffix in ("png", "svg"):
        fig.savefig(f"{args.prefix}-figure.{suffix}", dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
