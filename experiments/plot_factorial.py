#!/usr/bin/env python3
"""Draw the preregistered interaction and secondary input-preparation diagnostics."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.summary.read_text())
    rows = data["per_seed"]
    if len(rows) != 12:
        raise ValueError("Expected twelve paired seeds")
    primary = data["primary_difference_of_interactions_pct_points"]
    diagnostics = data["diagnostics_descriptive"]

    navy, teal, coral, muted = "#172f45", "#087986", "#cd654c", "#64788a"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"})
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.6, 6.2), dpi=170,
                                     gridspec_kw={"width_ratios": [1.15, 1]})
    fig.patch.set_facecolor("#f6f8fb")
    for ax in (left, right):
        ax.set_facecolor("white")

    for row in rows:
        fresh, reused = row["fresh_delta_gain_pct"], row["reused_delta_gain_pct"]
        color = teal if reused >= fresh else coral
        left.plot([0, 1], [fresh, reused], color=color, alpha=0.48, linewidth=1.7)
        left.scatter([0, 1], [fresh, reused], color=color, s=34, edgecolor="white",
                     linewidth=0.7, zorder=3)
    left.axhline(0, color=navy, linewidth=1, linestyle=(0, (4, 4)), alpha=0.7)
    left.set_xticks([0, 1], ["Entrées fraîches", "Entrées réutilisées"])
    left.set_xlim(-0.17, 1.17)
    left.set_ylabel("Changement du gain feedback sous H40 (points)")
    left.set_title("Interaction CPU × entrées, par seed", loc="left",
                   fontsize=12, weight="bold", color=navy, pad=14)
    left.grid(axis="y", color="#e2e8ef", linewidth=0.8)
    left.set_axisbelow(True)
    ci = primary["bootstrap_95pct_ci"]
    left.text(0.5, 0.05,
              f"Contraste médian : {primary['median']:+.2f} points\n"
              f"IC bootstrap 95 % : [{ci[0]:+.2f} ; {ci[1]:+.2f}]",
              transform=left.transAxes, ha="center", va="bottom", fontsize=9,
              color=navy, bbox={"boxstyle": "round,pad=0.55",
                                "facecolor": "#eef3f7", "edgecolor": "none"})

    labels = ["Fraîches · H0", "Fraîches · H40", "Réutilisées · H0", "Réutilisées · H40"]
    keys = [("host_to_device", 0), ("host_to_device", 40),
            ("host_to_device_reused_pinned", 0),
            ("host_to_device_reused_pinned", 40)]
    preparation = []
    for mode, workers in keys:
        values = [diagnostics[f"{mode}_h{workers}_{policy}"]["input_prepare_ms_median"]
                  for policy in ("round_robin", "feedback")]
        preparation.append(sum(values) / len(values))
    right.barh(range(4), preparation, color=[teal, coral, teal, coral], height=0.57)
    right.set_yticks(range(4), labels)
    right.invert_yaxis()
    right.set_xlabel("Préparation médiane d'une tâche (ms)")
    right.set_title("Diagnostic secondaire côté hôte", loc="left",
                    fontsize=12, weight="bold", color=navy, pad=14)
    right.grid(axis="x", color="#e2e8ef", linewidth=0.8)
    right.set_axisbelow(True)
    for y, value in enumerate(preparation):
        right.text(value, y, f"  {value:.1f}", va="center", color=navy, fontsize=9)
    right.set_xlim(0, max(preparation) * 1.24 if max(preparation) else 1)

    fig.suptitle("Contrôle prospectif H0/H40 × durée de vie des entrées",
                 x=0.08, y=0.95, ha="left", fontsize=16, weight="bold", color=navy)
    fig.text(0.08, 0.89, "12 seeds appariées · 192 runs · mêmes valeurs numériques dans les quatre cellules",
             color=muted, fontsize=9.5)
    fig.text(0.08, 0.045,
             "Gauche : statistique préenregistrée. Droite : moyenne descriptive des deux politiques, sans test causal séparé.",
             color=muted, fontsize=8.5)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.78, bottom=0.16, wspace=0.45)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, facecolor=fig.get_facecolor())
    print(args.out)


if __name__ == "__main__":
    main()
