#!/usr/bin/env python3
"""Render fixed online-probe results and the cost of the shared prefix."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
rows = list(csv.DictReader((ROOT / "results/online-probe-seeds-2026-10-24.csv").open()))
summary = json.loads((ROOT / "results/online-probe-summary-2026-10-24.json").read_text())
rows.sort(key=lambda row: float(row["gain_vs_probe_feedback_pct"]))

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "svg.fonttype": "none"})
fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5.5), dpi=160,
                                 gridspec_kw={"width_ratios": [1.45, 1]})
fig.patch.set_facecolor("#f7f9fc")
for ax in (left, right):
    ax.set_facecolor("white")

values = [float(row["gain_vs_probe_feedback_pct"]) for row in rows]
colors = ["#0a7380" if value > 0 else "#c56647" for value in values]
left.bar(range(len(rows)), values, color=colors, width=0.78)
left.axhline(0, color="#25364a", linewidth=1)
left.set_xticks(range(len(rows)), [row["seed"] for row in rows], rotation=70, fontsize=8)
left.set_xlabel("Seed, triée par gain")
left.set_ylabel("Gain de probe_switch vs probe_feedback (%)")
left.grid(axis="y", color="#e2e8ef", linewidth=0.7)
left.set_axisbelow(True)
left.set_title("Choisir la route n'améliore pas\nle feedback avec la même sonde",
               loc="left", weight="bold", color="#152b42", fontsize=12)
left.text(0.97, 0.95, "Médiane −0,12 %\nIC 95 % [−0,59 ; +0,23]",
          transform=left.transAxes, ha="right", va="top", color="#152b42", fontsize=9,
          bbox={"boxstyle": "round,pad=0.5", "facecolor": "#eef3f7", "edgecolor": "none"})

route = summary["switch_routes"]
right.barh(["Feedback", "Round-robin"], [route["feedback"], route["round_robin"]],
           color=["#0a7380", "#c56647"], height=0.52)
right.set_xlim(0, 24)
right.set_xlabel("Seeds sélectionnées, sur 24")
right.set_title("La sonde choisit souvent feedback", loc="left",
                weight="bold", color="#152b42", fontsize=12)
right.grid(axis="x", color="#e2e8ef", linewidth=0.7)
right.set_axisbelow(True)
for y, value in enumerate([route["feedback"], route["round_robin"]]):
    right.text(value + 0.5, y, str(value), va="center", color="#152b42", weight="bold")
right.text(0, -0.84, "Le contrôle probe_feedback est plus rapide\nque probe_rr sur les 24 seeds.",
           fontsize=9, color="#5b6b7d")

fig.suptitle("Décision en ligne sur deux RTX PRO 4000 Blackwell", x=0.075, ha="left",
             fontsize=16, weight="bold", color="#152b42")
fig.text(0.075, 0.89, "240 runs · 2 880 tâches · seeds 1400–1423 · deux répétitions par politique",
         color="#5b6b7d", fontsize=9.5)
fig.subplots_adjust(left=0.075, right=0.96, top=0.77, bottom=0.2, wspace=0.3)
out = ROOT / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "06_online_probe.svg", facecolor=fig.get_facecolor())
fig.savefig(out / "06_online_probe.png", facecolor=fig.get_facecolor())
print(out / "06_online_probe.svg")
