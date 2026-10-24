#!/usr/bin/env python3
"""Prospective test of the preregistered static-balance interaction."""
import argparse
import csv
import json
import random
import statistics
from pathlib import Path

from seed_stratification import bootstrap_median, load_block


def analyze(rows, draws=10000, seed_start=1000, seed_count=24):
    if len(rows) != seed_count or sorted(r["seed"] for r in rows) != list(range(seed_start, seed_start + seed_count)):
        raise ValueError(f"Expected the complete predeclared seed grid {seed_start}–{seed_start + seed_count - 1}")
    balanced = [r["feedback_gain_pct"] for r in rows if r["long_task_imbalance"] == 0]
    imbalanced = [r["feedback_gain_pct"] for r in rows if r["long_task_imbalance"] > 0]
    if len(balanced) + len(imbalanced) != seed_count or not balanced or not imbalanced:
        raise ValueError("Missing or invalid strata")
    groups = {}
    for name, values in (("balanced", balanced), ("imbalanced", imbalanced)):
        groups[name] = {"seeds": len(values), "wins": sum(x > 0 for x in values),
                        "median_gain_pct": statistics.median(values),
                        "bootstrap_95pct_ci": bootstrap_median(values, draws)}
    rng = random.Random(7)
    differences = sorted(
        statistics.median(imbalanced[rng.randrange(len(imbalanced))]
                          for _ in imbalanced)
        - statistics.median(balanced[rng.randrange(len(balanced))]
                            for _ in balanced)
        for _ in range(draws))
    lo, hi = differences[int(.025 * draws)], differences[int(.975 * draws)]
    primary = {"median_difference_pct_points": groups["imbalanced"]["median_gain_pct"]
               - groups["balanced"]["median_gain_pct"],
               "bootstrap_95pct_ci": [lo, hi],
               "validated_by_protocol": len(balanced) >= 6 and len(imbalanced) >= 6 and lo > 0}
    return {"primary": primary, "groups": groups,
            "analysis_status": "complete" if min(len(balanced), len(imbalanced)) >= 6
            else "insufficient_stratum_size"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs_csv")
    parser.add_argument("tasks_csv")
    parser.add_argument("--out-seeds", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--seed-count", type=int, default=24)
    args = parser.parse_args()
    rows = load_block(args.runs_csv, args.tasks_csv)
    result = analyze(rows, seed_start=args.seed_start, seed_count=args.seed_count)
    args.out_seeds.parent.mkdir(parents=True, exist_ok=True)
    with args.out_seeds.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    args.out_summary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
