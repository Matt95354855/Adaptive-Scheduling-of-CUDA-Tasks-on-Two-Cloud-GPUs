#!/usr/bin/env python3
"""Create publication-ready figures from the archived seed and run summaries.

Run from the repository root with: python experiments/plot_research.py
Requires matplotlib; no GPU or cloud instance is needed.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

INK = "#152A3B"
MUTED = "#5E6F7C"
TEAL = "#087F7B"
CORAL = "#DD725D"
GOLD = "#D9A441"
PAPER = "#FAF9F6"
GRID = "#DCE3E5"


def rows(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summary(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def fmt(value: float, digits: int = 1) -> str:
    return f"{value:+.{digits}f}".replace(".", ",")


def setup() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "svg.fonttype": "none",
            "savefig.facecolor": PAPER,
            "figure.facecolor": PAPER,
        }
    )
    FIGURES.mkdir(exist_ok=True)


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURES / f"{stem}.svg", bbox_inches="tight", pad_inches=0.28)
    fig.savefig(FIGURES / f"{stem}.png", dpi=220, bbox_inches="tight", pad_inches=0.28)
    plt.close(fig)


def prospective() -> None:
    data = rows("strata-validation-seeds-2026-10-23.csv")
    stats = summary("strata-validation-summary-2026-10-23.json")
    fig, ax = plt.subplots(figsize=(11.6, 6.8))
    fig.subplots_adjust(left=0.17, right=0.95, top=0.76, bottom=0.22)
    groups = [("balanced", 1, TEAL, "Équilibré 2–2"), ("imbalanced", 0, CORAL, "Déséquilibré 3–1 / 4–0")]
    for key, y, color, label in groups:
        gains = sorted(float(row["feedback_gain_pct"]) for row in data if (int(row["long_task_imbalance"]) == 0) == (key == "balanced"))
        point_offsets = [((i * 7) % 9 - 4) * 0.037 for i in range(len(gains))]
        ax.scatter(gains, [y + offset for offset in point_offsets], s=64, color=color, alpha=0.82,
                   edgecolor=PAPER, linewidth=1.3, zorder=3)
        info = stats["groups"][key]
        low, high = info["bootstrap_95pct_ci"]
        middle = info["median_gain_pct"]
        ax.plot([low, high], [y - 0.28, y - 0.28], color=INK, linewidth=3, solid_capstyle="round", zorder=4)
        ax.plot([low, low], [y - 0.33, y - 0.23], color=INK, linewidth=2, zorder=4)
        ax.plot([high, high], [y - 0.33, y - 0.23], color=INK, linewidth=2, zorder=4)
        ax.scatter([middle], [y - 0.28], marker="D", s=85, color=INK, edgecolor=PAPER, linewidth=1, zorder=5)
        ax.text(31.5, y + 0.05, f"{info['wins']}/{info['seeds']} victoires", color=color,
                fontsize=10, fontweight="bold", ha="right", va="center")
        ax.text(middle, y - 0.42, f"médiane {fmt(middle)} %", ha="center", fontsize=9.5, color=INK)
    ax.axvline(0, color=MUTED, linewidth=1.3, linestyle=(0, (4, 4)), zorder=1)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_xlim(-17, 33)
    ax.set_ylim(-0.62, 1.48)
    ax.set_yticks([1, 0], [groups[0][3], groups[1][3]], fontweight="bold")
    ax.tick_params(axis="y", length=0, pad=12)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f} %"))
    ax.set_xlabel("Gain du feedback sur le temps total (positif = plus rapide)", labelpad=13)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.text(0.055, 0.93, "01  /  VALIDATION PROSPECTIVE", color=TEAL, fontsize=11, fontweight="bold")
    fig.text(0.055, 0.865, "Le gain dépend du déséquilibre des tâches longues", fontsize=19, fontweight="bold")
    primary = stats["primary"]
    fig.text(0.055, 0.797,
             f"Différence des médianes : {fmt(primary['median_difference_pct_points'])} points  ·  IC bootstrap 95 % [{fmt(primary['bootstrap_95pct_ci'][0])} ; {fmt(primary['bootstrap_95pct_ci'][1])}]",
             fontsize=11.3, color=MUTED)
    fig.text(0.055, 0.055,
             "24 seeds nouvelles (1000–1023), 2 répétitions par politique. Points = gains par seed ; losange = médiane ; trait = IC 95 %.\n"
             "Le groupe est défini par les durées réelles observées après exécution ; ce n’est pas encore un signal utilisable en ligne.",
             fontsize=9.1, color=MUTED, linespacing=1.6)
    save(fig, "01_validation_prospective")


def contention() -> None:
    info = summary("transfer-hidden-contention-2026-10-23.json")
    fig, ax = plt.subplots(figsize=(10.8, 7.0))
    fig.subplots_adjust(left=0.12, right=0.85, top=0.75, bottom=0.20)
    for row in info["per_seed"]:
        before, after = row["h0_gain_pct"], row["h40_gain_pct"]
        color = CORAL if after < before else TEAL
        ax.plot([0, 1], [before, after], color=color, linewidth=1.8, alpha=0.60, zorder=2)
        ax.scatter([0, 1], [before, after], color=color, s=42, edgecolor=PAPER,
                   linewidth=0.9, zorder=3)
    ax.axhline(0, color=MUTED, linewidth=1.2, linestyle=(0, (4, 4)), zorder=1)
    ax.scatter([0, 1], [info["h0"]["median_pct"], info["h40"]["median_pct"]],
               marker="D", s=130, color=INK, edgecolor=PAPER, linewidth=1.2, zorder=5)
    ax.text(1.08, 27.5, "10 / 12 seeds\nperdent du gain", color=CORAL,
            fontsize=12, fontweight="bold", va="top")
    ax.text(1.08, 18.5, "orange = baisse\nvert = hausse", color=MUTED,
            fontsize=9.5, va="top")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_xlim(-0.15, 1.36)
    ax.set_ylim(-25.5, 30.5)
    ax.set_xticks([0, 1],
                  [f"H0 · médiane {fmt(info['h0']['median_pct'])} %\nSans charge ajoutée",
                   f"H40 · médiane {fmt(info['h40']['median_pct'])} %\n40 processus CPU"],
                  fontweight="bold")
    ax.set_ylabel("Gain du feedback par seed (%)", labelpad=13)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f} %"))
    ax.tick_params(axis="x", length=0, pad=14)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.text(0.055, 0.93, "02  /  CONTENTION HÔTE", color=CORAL, fontsize=11, fontweight="bold")
    fig.text(0.055, 0.865, "Le gain s’érode avec la charge CPU", fontsize=20, fontweight="bold")
    interaction = info["interaction"]
    fig.text(0.055, 0.797,
             f"Changement apparié médian : {fmt(interaction['median_pct'])} points  ·  IC bootstrap 95 % [{fmt(interaction['bootstrap_95pct_ci'][0])} ; {fmt(interaction['bootstrap_95pct_ci'][1])}]",
             fontsize=10.8, color=MUTED)
    fig.text(0.055, 0.055,
             "Mêmes 12 seeds (800–811), 2 répétitions par politique et condition. Chaque segment relie une seed.\n"
             "Conditions exécutées successivement : une dérive de l’instance peut contribuer au changement.",
             fontsize=9.1, color=MUTED, linespacing=1.6)
    save(fig, "02_contention_appariee")


def host_gpu_mechanism() -> None:
    h0 = rows("transfer-hidden-h0-task-metrics-2026-10-23.csv")
    h40 = rows("transfer-hidden-h40-task-metrics-2026-10-23.csv")
    run_h0 = rows("transfer-hidden-h0-2026-10-23.csv")
    run_h40 = rows("transfer-hidden-h40-2026-10-23.csv")
    metrics = [
        ("Préparation des entrées", "input_prepare_ms_median", h0, h40, "run"),
        ("Événements GPU cumulés", "gpu_event_ms_sum", h0, h40, "run"),
        ("Temps total", "makespan_ms", run_h0, run_h40, "seed"),
    ]
    fig, ax = plt.subplots(figsize=(11.6, 6.7))
    fig.subplots_adjust(left=0.26, right=0.80, top=0.75, bottom=0.20)
    labels = []
    for index, (label, field, first, second, unit) in enumerate(metrics):
        y0 = 2 - index
        labels.append(label)
        for policy, offset, color in [("round_robin", 0.13, TEAL), ("feedback", -0.13, CORAL)]:
            def aggregate(data: list[dict[str, str]]) -> float:
                selected = [row for row in data if row["policy"] == policy]
                if unit == "seed":
                    seeds = sorted({int(row["seed"]) for row in selected})
                    return median(median(float(row[field]) for row in selected if int(row["seed"]) == seed) for seed in seeds)
                return median(float(row[field]) for row in selected)

            low, high = aggregate(first), aggregate(second)
            ratio = high / low
            ax.plot([1, ratio], [y0 + offset, y0 + offset], color=color, linewidth=3,
                    solid_capstyle="round", zorder=2)
            ax.scatter([ratio], [y0 + offset], color=color, s=90, edgecolor=PAPER,
                       linewidth=1.2, zorder=3)
            ax.text(1.02, y0 + offset, f"{low:.1f} → {high:.1f} ms",
                    transform=ax.get_yaxis_transform(), ha="left", va="center",
                    color=color, fontsize=10, fontweight="bold")
    ax.axvline(1, color=MUTED, linewidth=1.3, linestyle=(0, (4, 4)), zorder=1)
    ax.set_xscale("log")
    ax.set_xlim(0.82, 52)
    ax.set_ylim(-0.5, 2.5)
    ax.set_xticks([1, 2, 5, 10, 20, 40], ["1×", "2×", "5×", "10×", "20×", "40×"])
    ax.set_yticks([2, 1, 0], labels, fontweight="bold")
    ax.tick_params(axis="y", length=0, pad=12)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_xlabel("Rapport H40 / H0 (à droite = plus lent)", labelpad=14)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.text(0.055, 0.93, "03  /  DIAGNOSTIC DU MÉCANISME", color=GOLD, fontsize=11, fontweight="bold")
    fig.text(0.055, 0.865, "Le surcoût apparaît surtout côté hôte", fontsize=20, fontweight="bold")
    fig.text(0.055, 0.797, "Médianes H0 → H40, par politique :", fontsize=11.2, color=MUTED)
    fig.text(0.66, 0.797, "●  round-robin", color=TEAL, fontsize=10.2, fontweight="bold")
    fig.text(0.82, 0.797, "●  feedback", color=CORAL, fontsize=10.2, fontweight="bold")
    fig.text(0.055, 0.052,
             "Préparation et événements GPU : médianes des 24 runs par politique et condition. Temps total : médianes des 12 seeds.\n"
             "Les durées d’événements GPU se chevauchent entre GPU ; elles ne s’additionnent pas au temps total.",
             fontsize=9.0, color=MUTED, linespacing=1.6)
    save(fig, "03_diagnostic_hote_gpu")


def independent_replication() -> None:
    datasets = [
        ("23 oct. · pod initial", rows("strata-validation-seeds-2026-10-23.csv"),
         summary("strata-validation-summary-2026-10-23.json")),
        ("24 oct. · nouveau pod", rows("replication-seeds-2026-10-24.csv"),
         summary("replication-summary-2026-10-24.json")),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 6.7), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.17, right=0.96, top=0.73, bottom=0.22, wspace=0.18)
    for ax, key, color, title in zip(axes,
                                      ("balanced", "imbalanced"),
                                      (TEAL, CORAL),
                                      ("Équilibré 2–2", "Déséquilibré 3–1 / 4–0")):
        for index, (label, data, stats) in enumerate(datasets):
            y = 1 - index
            values = sorted(float(row["feedback_gain_pct"]) for row in data
                            if (int(row["long_task_imbalance"]) == 0) == (key == "balanced"))
            jitter = [((i * 7) % 9 - 4) * 0.028 for i in range(len(values))]
            ax.scatter(values, [y + offset + 0.08 for offset in jitter], s=45,
                       color=color, edgecolor=PAPER, linewidth=0.9, alpha=0.72, zorder=3)
            group = stats["groups"][key]
            low, high = group["bootstrap_95pct_ci"]
            middle = group["median_gain_pct"]
            ax.plot([low, high], [y - 0.22, y - 0.22], color=INK,
                    linewidth=2.8, solid_capstyle="round", zorder=4)
            ax.plot([low, low], [y - 0.27, y - 0.17], color=INK, linewidth=1.8)
            ax.plot([high, high], [y - 0.27, y - 0.17], color=INK, linewidth=1.8)
            ax.scatter([middle], [y - 0.22], marker="D", s=75, color=INK,
                       edgecolor=PAPER, linewidth=0.8, zorder=5)
            ax.text(middle, y - 0.37, f"{fmt(middle)} % · {group['wins']}/{group['seeds']}",
                    color=INK, fontsize=9.3, ha="center", va="top")
        ax.axvline(0, color=MUTED, linewidth=1.2, linestyle=(0, (4, 4)))
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_xlim(-18, 35)
        ax.set_ylim(-0.57, 1.48)
        ax.set_title(title, loc="left", color=color, fontsize=13.5, fontweight="bold", pad=16)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f} %"))
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_yticks([1, 0], [datasets[0][0], datasets[1][0]], fontweight="bold")
    axes[0].tick_params(axis="y", length=0, pad=12)
    axes[1].tick_params(axis="y", length=0)
    fig.supxlabel("Gain du feedback sur le temps total (positif = plus rapide)", y=0.13, fontsize=11)
    fig.text(0.055, 0.93, "04  /  RÉPLICATION INDÉPENDANTE", color=TEAL, fontsize=11, fontweight="bold")
    fig.text(0.055, 0.865, "La séparation se retrouve sur une nouvelle instance", fontsize=19, fontweight="bold")
    fig.text(0.055, 0.797, "Différence des médianes : +20,9 points le 23 oct. ; +26,1 points le 24 oct.",
             fontsize=11, color=MUTED)
    fig.text(0.055, 0.055,
             "Chaque point = une seed ; losange = médiane ; trait = IC bootstrap 95 %. 24 seeds distinctes par instance, 2 répétitions.\n"
             "Le classement repose toujours sur les durées observées après exécution et ne peut pas être appliqué tel quel en ligne.",
             fontsize=9, color=MUTED, linespacing=1.6)
    save(fig, "04_replication_independante")


if __name__ == "__main__":
    setup()
    prospective()
    contention()
    host_gpu_mechanism()
    independent_replication()
    print(f"Figures SVG et PNG créées dans {FIGURES}")
