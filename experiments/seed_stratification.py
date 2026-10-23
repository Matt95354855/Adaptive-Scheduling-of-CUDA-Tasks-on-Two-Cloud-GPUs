#!/usr/bin/env python3
"""Describe hidden-duration gains by observed static long-task imbalance."""
import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path


def load_block(runs_path, tasks_path):
    times = defaultdict(lambda: defaultdict(list))
    with open(runs_path, newline="") as stream:
        for row in csv.DictReader(stream):
            if row["workload"] != "transfer_hidden" or int(row["host_load_workers"]) != 0:
                raise ValueError("Expected transfer_hidden H0 runs")
            times[int(row["seed"])][row["policy"]].append(float(row["makespan_ms"]))
    long_counts = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    with open(tasks_path, newline="") as stream:
        for row in csv.DictReader(stream):
            if row["policy"] == "round_robin" and int(row["repeats"]) == 32:
                long_counts[int(row["seed"])][int(row["repetition"])][int(row["gpu"])] += 1
    result = []
    for seed, policies in sorted(times.items()):
        if set(policies) != {"round_robin", "feedback"} or any(len(v) != 2 for v in policies.values()):
            raise ValueError(f"Incomplete paired runs for seed {seed}")
        counts = long_counts[seed]
        if set(counts) != {0, 1} or any(sum(pair) != 4 for pair in counts.values()):
            raise ValueError(f"Incomplete long-task assignment for seed {seed}")
        imbalance = {abs(pair[0] - pair[1]) for pair in counts.values()}
        if len(imbalance) != 1:
            raise ValueError(f"Static imbalance changed across repeats for seed {seed}")
        rr = statistics.median(policies["round_robin"])
        fb = statistics.median(policies["feedback"])
        result.append({"seed": seed, "long_task_imbalance": imbalance.pop(),
                       "round_robin_median_ms": rr, "feedback_median_ms": fb,
                       "feedback_gain_pct": 100 * (1 - fb / rr)})
    return result


def bootstrap_median(values, draws=10000):
    rng = random.Random(7)
    n = len(values)
    samples = sorted(statistics.median(values[rng.randrange(n)] for _ in range(n))
                     for _ in range(draws))
    return [samples[int(.025 * draws)], samples[int(.975 * draws)]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("h0_runs")
    parser.add_argument("h0_tasks")
    parser.add_argument("replication_runs")
    parser.add_argument("replication_tasks")
    parser.add_argument("--out-seeds", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    args = parser.parse_args()
    first = load_block(args.h0_runs, args.h0_tasks)
    second = load_block(args.replication_runs, args.replication_tasks)
    if {r["seed"] for r in first} & {r["seed"] for r in second}:
        raise ValueError("Replication seeds must be independent")
    rows = [{"block": block, **row} for block, group in
            (("initial", first), ("replication", second)) for row in group]
    args.out_seeds.parent.mkdir(parents=True, exist_ok=True)
    with args.out_seeds.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for name, subset in (("balanced", [r for r in rows if r["long_task_imbalance"] == 0]),
                         ("imbalanced", [r for r in rows if r["long_task_imbalance"] > 0]),
                         ("all", rows)):
        gains = [r["feedback_gain_pct"] for r in subset]
        summary[name] = {"seeds": len(gains), "wins": sum(x > 0 for x in gains),
                         "median_gain_pct": statistics.median(gains),
                         "bootstrap_95pct_ci": bootstrap_median(gains)}
    args.out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
