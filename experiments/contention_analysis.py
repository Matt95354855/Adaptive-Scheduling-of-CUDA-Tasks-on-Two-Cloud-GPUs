#!/usr/bin/env python3
"""Paired seed-level analysis of the predeclared H0/H40 contention study."""
import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path

POLICIES = {"round_robin", "feedback"}


def read_cells(path, expected_workers):
    cells = {}
    with Path(path).open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["workload"] != "transfer_hidden":
                raise ValueError(f"{path}: unexpected workload {row['workload']}")
            if int(row["host_load_workers"]) != expected_workers:
                raise ValueError(f"{path}: unexpected host load")
            if row["policy"] not in POLICIES:
                raise ValueError(f"{path}: unexpected policy {row['policy']}")
            key = (int(row["seed"]), int(row["repetition"]), row["policy"])
            value = float(row["makespan_ms"])
            if key in cells or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{path}: duplicate or invalid cell {key}")
            cells[key] = value
    if not cells:
        raise ValueError(f"{path}: no measurements")
    return cells


def interval(values, seed=7, draws=10000):
    rng = random.Random(seed)
    size = len(values)
    samples = sorted(statistics.median(values[rng.randrange(size)] for _ in values)
                     for _ in range(draws))
    return [samples[int(.025 * draws)], samples[int(.975 * draws)]]


def analyze(h0, h40):
    control, pressured = read_cells(h0, 0), read_cells(h40, 40)
    if control.keys() != pressured.keys():
        raise ValueError("H0 and H40 must have the same seed/repetition/policy cells")
    seeds = sorted({seed for seed, _, _ in control})
    repetitions = sorted({rep for _, rep, _ in control})
    expected = {(seed, rep, policy) for seed in seeds for rep in repetitions
                for policy in POLICIES}
    if control.keys() != expected:
        raise ValueError("Incomplete seed/repetition/policy grid")
    per_seed = []
    for seed in seeds:
        values = {condition: {policy: statistics.median(cells[seed, rep, policy]
                       for rep in repetitions) for policy in POLICIES}
                  for condition, cells in (("H0", control), ("H40", pressured))}
        gain_h0 = 100 * (1 - values["H0"]["feedback"] / values["H0"]["round_robin"])
        gain_h40 = 100 * (1 - values["H40"]["feedback"] / values["H40"]["round_robin"])
        per_seed.append({"seed": seed, "h0_gain_pct": gain_h0,
                         "h40_gain_pct": gain_h40, "delta_gain_pct": gain_h40 - gain_h0,
                         "h0_median_ms": values["H0"], "h40_median_ms": values["H40"]})
    summary = {"seed_count": len(seeds), "repetitions_per_cell": len(repetitions),
               "per_seed": per_seed}
    for condition, field in (("h0", "h0_gain_pct"), ("h40", "h40_gain_pct"),
                             ("interaction", "delta_gain_pct")):
        gains = [row[field] for row in per_seed]
        ci = interval(gains)
        summary[condition] = {"median_pct": statistics.median(gains),
                              "bootstrap_95pct_ci": ci}
        if condition != "interaction":
            summary[condition]["wins"] = sum(gain > 0 for gain in gains)
            summary[condition]["robust_by_protocol"] = (ci[0] > 0 and
                summary[condition]["wins"] >= 9)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("h0_csv")
    parser.add_argument("h40_csv")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = analyze(args.h0_csv, args.h40_csv)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
