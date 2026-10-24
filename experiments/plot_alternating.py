#!/usr/bin/env python3
"""Render the prospective paired H0/H40 feedback-gain comparison."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


ROOT = Path(__file__).resolve().parents[1]
data = json.loads((ROOT / "results/alternating-2026-10-24-summary.json").read_text())
rows = data["per_seed"]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "svg.fonttype": "none"})
fig, ax = plt.subplots(figsize=(9.4, 5.8), dpi=160)
fig.patch.set_facecolor("#f7f9fc")
ax.set_facecolor("#ffffff")

for row in rows:
    y0, y1 = row["h0_gain_pct"], row["h40_gain_pct"]
    color = "#0a7380" if y1 >= y0 else "#c56647"
    ax.plot([0, 1], [y0, y1], color=color, alpha=0.42, linewidth=1.6, zorder=2)
    ax.scatter([0, 1], [y0, y1], color=color, s=33, edgecolor="white",
               linewidth=0.6, zorder=3)

ax.axhline(0, color="#25364a", linewidth=1.0, linestyle=(0, (4, 4)), alpha=0.68)
ax.set_xticks([0, 1], ["H0 · sans charge CPU", "H40 · 40 processus CPU"])
ax.set_xlim(-0.14, 1.14)
ax.set_ylim(-30, 30)
ax.yaxis.set_major_locator(MultipleLocator(10))
ax.grid(axis="y", color="#e2e8ef", linewidth=0.8)
ax.set_axisbelow(True)
ax.set_ylabel("Gain de feedback vs round-robin (%)")
ax.set_title("La contention semble réduire le gain, sans preuve concluante",
             loc="left", fontsize=15, weight="bold", color="#152b42", pad=18)
ax.text(0, 1.018, "12 seeds appariées · deux répétitions par cellule · nouveau pod, 24 octobre 2026",
        transform=ax.transAxes, fontsize=9.5, color="#5b6b7d")
ax.text(0.5, 0.055,
        "Δ médian H40 − H0 : −13,08 points  ·  IC bootstrap 95 % : [−21,81 ; +5,19]",
        transform=ax.transAxes, ha="center", va="center", color="#152b42", fontsize=9,
        bbox={"boxstyle": "round,pad=0.55", "facecolor": "#eef3f7", "edgecolor": "none"})
fig.text(0.12, 0.03, "Chaque segment relie la même seed ; couleur turquoise = gain amélioré sous H40.",
         color="#5b6b7d", fontsize=8.5)
fig.subplots_adjust(left=0.12, right=0.96, top=0.84, bottom=0.19)

out = ROOT / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "05_contention_interaction.svg", facecolor=fig.get_facecolor())
fig.savefig(out / "05_contention_interaction.png", facecolor=fig.get_facecolor())
print(out / "05_contention_interaction.svg")
