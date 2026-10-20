#!/usr/bin/env python3
"""Analyze paired run-level CSV from study.py without pseudoreplicating repeats."""
import argparse
import csv
import random
import statistics
from collections import defaultdict
from pathlib import Path


def median(values):
    return statistics.median(values)


def bootstrap_ci(values, draws=10000, seed=7):
    rng = random.Random(seed)
    n = len(values)
    samples = [median([values[rng.randrange(n)] for _ in range(n)]) for _ in range(draws)]
    samples.sort()
    return samples[int(.025 * draws)], samples[int(.975 * draws)]


def analyze(path):
    with open(path, newline="") as file:
        source = list(csv.DictReader(file))
    if "policy" in source[0]:
        rows = source
    else:
        workload = Path(path).name.split("-")[0]
        rows = [{"workload": workload, "seed": row["seed"],
                 "repetition": row["rep"], "policy": policy,
                 "makespan_ms": row[policy]}
                for row in source for policy in source[0]
                if policy not in ("seed", "rep")]
    cells = defaultdict(list)
    for row in rows:
        cells[(row["workload"], int(row["seed"]), row["policy"])].append(float(row["makespan_ms"]))
    workloads = sorted({key[0] for key in cells})
    for workload in workloads:
        seeds = sorted({key[1] for key in cells if key[0] == workload})
        policies = sorted({key[2] for key in cells if key[0] == workload})
        expected = {(workload, seed, policy) for seed in seeds for policy in policies}
        if set(key for key in cells if key[0] == workload) != expected:
            raise ValueError("Incomplete seed/policy grid")
        print(f"{workload}: {len(seeds)} independent task orders, {len(rows)} total CSV rows")
        print("policy,median_seed_ms,median_change_vs_rr_pct,seed_wins_vs_rr,bootstrap_95pct_CI_change")
        ref = {seed: median(cells[(workload, seed, "round_robin")]) for seed in seeds}
        for policy in policies:
            seed_values = [median(cells[(workload, seed, policy)]) for seed in seeds]
            deltas = [100 * (value / ref[seed] - 1) for seed, value in zip(seeds, seed_values)]
            lo, hi = bootstrap_ci(deltas)
            wins = sum(value < ref[seed] for seed, value in zip(seeds, seed_values))
            print(f"{policy},{median(seed_values):.3f},{median(deltas):+.2f},{wins}/{len(seeds)},[{lo:+.2f},{hi:+.2f}]")
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    analyze(parser.parse_args().csv)


if __name__ == "__main__":
    main()
