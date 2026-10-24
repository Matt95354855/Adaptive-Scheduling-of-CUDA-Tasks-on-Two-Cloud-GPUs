#!/usr/bin/env python3
"""Publication figures for post-hoc probe choices and whole-campaign GPU duty cycle."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

NAVY = "#172f45"
MUTED = "#64788a"
TEAL = "#087986"
CORAL = "#cd654c"
PALE = "#d9e2e9"
MID = "#81bdc1"
BG = "#f6f8fb"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "svg.fonttype": "none", "savefig.facecolor": BG})


def read_json(name):
    return json.loads((RESULTS / name).read_text())


def read_csv(name):
    with (RESULTS / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def save(fig, stem):
    fig.savefig(FIGURES / f"{stem}.svg")
    fig.savefig(FIGURES / f"{stem}.png", dpi=180)
    plt.close(fig)
    print(FIGURES / f"{stem}.svg")


def utilization_figure():
    short = read_json("online-probe-utilization-summary-2026-10-24.json")
    dense = read_json("sustained-utilization-summary-2026-10-24.json")
    dense_runs = read_csv("sustained-2026-10-24.csv")
    measured_s = sum(float(row["makespan_ms"]) for row in dense_runs) / 1000
    sampled_s = dense["samples"]["0"]["count"] * dense["sample_interval_s"]
    outside_s = sampled_s - measured_s
    if outside_s < 0:
        raise ValueError("Measured batches exceed sampled campaign duration")

    fig = plt.figure(figsize=(13.2, 6.8))
    fig.patch.set_facecolor(BG)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.22, 1], height_ratios=[1, 1],
                           left=0.15, right=0.96, top=0.76, bottom=0.18,
                           hspace=0.54, wspace=0.32)
    bars = fig.add_subplot(grid[:, 0])
    time_ax = fig.add_subplot(grid[0, 1])
    event_ax = fig.add_subplot(grid[1, 1])
    for ax in (bars, time_ax, event_ax):
        ax.set_facecolor("white")
    fig.text(0.08, 0.94, "07 / OÙ PASSE LE TEMPS GPU ?", fontsize=10,
             weight="bold", color=TEAL)
    fig.text(0.08, 0.885, "Des lots presque saturés, un pod souvent au repos",
             fontsize=17, weight="bold", color=NAVY)
    fig.text(0.08, 0.83,
             "Mesure SM chaque seconde sur deux charges distinctes · deux RTX PRO 4000 Blackwell",
             fontsize=10, color=MUTED)

    configs = [("Sonde · GPU 0", short, "0"), ("Sonde · GPU 1", short, "1"),
               ("Calcul dense · GPU 0", dense, "0"), ("Calcul dense · GPU 1", dense, "1")]
    for index, (label, dataset, gpu) in enumerate(configs):
        row = dataset["samples"][gpu]
        zero = row["pct_samples_zero"]
        high = row["pct_samples_ge_70"]
        middle = 100 - zero - high
        y = 3 - index
        bars.barh(y, zero, color=PALE, height=0.58)
        bars.barh(y, middle, left=zero, color=MID, height=0.58)
        bars.barh(y, high, left=zero + middle, color=TEAL, height=0.58)
        bars.text(104, y, f"moy. {row['mean_sm_pct']:.1f} %", va="center",
                  color=NAVY, weight="bold", fontsize=9)
    bars.set_yticks(range(4), [item[0] for item in reversed(configs)])
    bars.set_xlim(0, 128)
    bars.set_xticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100 %"])
    bars.set_title("Occupation SM sur toute la campagne", loc="left",
                   fontsize=12, weight="bold", color=NAVY, pad=14)
    fig.legend(handles=[Patch(color=PALE, label="0 %"),
                        Patch(color=MID, label="1–69 %"),
                        Patch(color=TEAL, label="≥ 70 %")],
               loc="lower left", bbox_to_anchor=(0.15, 0.095), ncol=3,
               frameon=False, fontsize=9)
    bars.spines["left"].set_visible(False)
    bars.tick_params(axis="y", length=0)

    time_ax.barh(0, measured_s, color=TEAL, height=0.45)
    time_ax.barh(0, outside_s, left=measured_s, color=PALE, height=0.45)
    time_ax.set_xlim(0, sampled_s * 1.02)
    time_ax.set_yticks([])
    time_ax.set_xticks([0, sampled_s / 2, sampled_s], ["0", "≈ 76 s", f"{sampled_s:.0f} s"])
    time_ax.set_title("Calcul dense : temps cumulé observé",
                      loc="left", fontsize=12, weight="bold", color=NAVY, pad=12)
    time_ax.text(measured_s / 2, 0, f"lots mesurés\n{measured_s:.1f} s",
                 ha="center", va="center", color="white", weight="bold", fontsize=10)
    time_ax.text(measured_s + outside_s / 2, 0,
                 f"hors lots\n{outside_s:.1f} s", ha="center", va="center",
                 color=NAVY, fontsize=10)
    time_ax.set_xlabel("153 relevés à une seconde ; parts cumulées, pas une chronologie")
    time_ax.spines["left"].set_visible(False)

    event_ax.set_title("À l'intérieur des lots denses", loc="left",
                       fontsize=12, weight="bold", color=NAVY, pad=12)
    for y, gpu in [(1, "0"), (0, "1")]:
        median = dense["cuda_event_fraction"][gpu]["median_pct"]
        event_ax.hlines(y, 0, median, linewidth=8, color=TEAL, alpha=0.35)
        event_ax.scatter(median, y, s=100, color=TEAL, zorder=3)
        event_ax.text(median - 3, y, f"{median:.1f} %", ha="right", va="center",
                      color=NAVY, weight="bold", fontsize=10)
    event_ax.set_xlim(0, 108)
    event_ax.set_yticks([0, 1], ["GPU 1", "GPU 0"])
    event_ax.set_xticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100 %"])
    event_ax.set_xlabel("Médiane temps événements CUDA / makespan")
    event_ax.grid(axis="x", color="#e3eaf0", linewidth=0.8)
    event_ax.set_axisbelow(True)

    fig.text(0.15, 0.045,
             "SM et temps d'événements CUDA sont deux mesures différentes. "
             "La sonde couvre 643 s ; le calcul dense, 153 s.",
             color=MUTED, fontsize=9)
    save(fig, "07_occupation_gpu_deux_echelles")


def route_figure():
    rows = read_csv("online-probe-route-diagnostics-2026-10-24.csv")
    matrix = [[0, 0], [0, 0]]
    for row in rows:
        matrix[int(row["rr_imbalanced"])][0 if row["chosen_route"] == "feedback" else 1] += 1
    ranked = sorted(rows, key=lambda row: 100 *
                    (1 - float(row["probe_feedback_median_ms"]) /
                     float(row["probe_rr_median_ms"])))
    fig, (left, right) = plt.subplots(1, 2, figsize=(13.2, 7.2),
                                     gridspec_kw={"width_ratios": [0.95, 1.15]})
    fig.patch.set_facecolor(BG)
    for ax in (left, right):
        ax.set_facecolor("white")
    fig.text(0.07, 0.955, "08 / DIAGNOSTIC EXPLORATOIRE", fontsize=10,
             weight="bold", color=TEAL)
    fig.text(0.07, 0.9, "La sonde choisit parfois la mauvaise continuation",
             fontsize=17, weight="bold", color=NAVY)
    fig.text(0.07, 0.85,
             "24 nouvelles seeds · même préfixe de quatre tâches dans les deux contrôles",
             fontsize=10, color=MUTED)

    left.set_title("Choix selon l'équilibre final", loc="left", fontsize=12,
                   weight="bold", color=NAVY, pad=16)
    left.imshow(matrix, cmap="BuGn", vmin=0, vmax=14, aspect="auto")
    for y in range(2):
        for x in range(2):
            value = matrix[y][x]
            left.text(x, y, str(value), ha="center", va="center", fontsize=25,
                      weight="bold", color="white" if value >= 9 else NAVY)
    left.set_xticks([0, 1], ["Feedback", "Round-robin"])
    left.set_yticks([0, 1], ["Équilibré 2–2\n10 seeds", "Déséquilibré\n14 seeds"])
    left.set_xlabel("Continuation choisie après la sonde")
    left.tick_params(length=0)
    for edge in ("left", "bottom"):
        left.spines[edge].set_visible(False)

    gains = [100 * (1 - float(row["probe_feedback_median_ms"]) /
                    float(row["probe_rr_median_ms"])) for row in ranked]
    colors = [CORAL if row["chosen_route"] == "round_robin" else TEAL for row in ranked]
    for y, (gain, color) in enumerate(zip(gains, colors)):
        right.hlines(y, 0, gain, color=color, linewidth=1.8, alpha=0.7)
        right.scatter(gain, y, color=color, s=35, zorder=3)
    right.axvline(0, color=NAVY, linewidth=1)
    right.set_yticks(range(len(ranked)), [row["seed"] for row in ranked], fontsize=8)
    right.set_xlim(-0.5, 23)
    right.set_ylim(-1, len(ranked))
    right.set_xlabel("Gain de probe_feedback vs probe_rr (%)")
    right.set_title("Feedback gagne sur les 24 seeds", loc="left",
                    fontsize=12, weight="bold", color=NAVY, pad=16)
    right.grid(axis="x", color="#e3eaf0", linewidth=0.8)
    right.set_axisbelow(True)
    right.legend(handles=[Patch(color=CORAL, label="Sonde a choisi round-robin"),
                          Patch(color=TEAL, label="Sonde a choisi feedback")],
                 loc="lower right", frameon=False, fontsize=9)
    fig.subplots_adjust(left=0.17, right=0.96, top=0.77, bottom=0.16, wspace=0.4)
    fig.text(0.07, 0.055,
             "Les durées et l'équilibre finaux sont connus après exécution. "
             "Ce diagnostic ne constitue pas une nouvelle règle validée.",
             color=MUTED, fontsize=9)
    save(fig, "08_sonde_decisions_et_controles")


if __name__ == "__main__":
    utilization_figure()
    route_figure()
