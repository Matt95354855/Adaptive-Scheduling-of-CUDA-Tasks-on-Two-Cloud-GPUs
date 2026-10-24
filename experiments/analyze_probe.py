#!/usr/bin/env python3
"""Paired seed-level evaluation of the fixed four-task online probe policy."""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from contention_analysis import interval

POLICIES = ("round_robin", "feedback", "probe_rr", "probe_feedback", "probe_switch")
CONTROLS = tuple(policy for policy in POLICIES if policy != "probe_switch")


def analyze(path, seed_start=1400, seed_count=24, repetitions=2):
    cells = {}
    routes = defaultdict(list)
    with Path(path).open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["workload"] != "transfer_hidden" or int(row["host_load_workers"]) != 0:
                raise ValueError("Expected transfer_hidden H0 runs")
            key = (int(row["seed"]), int(row["repetition"]), row["policy"])
            if key in cells:
                raise ValueError(f"Duplicate run {key}")
            cells[key] = float(row["makespan_ms"])
            if row["policy"] == "probe_switch":
                routes[int(row["seed"])].append(row["probe_route"])
    expected = {(seed, rep, policy)
                for seed in range(seed_start, seed_start + seed_count)
                for rep in range(repetitions) for policy in POLICIES}
    if set(cells) != expected:
        missing, extra = expected - set(cells), set(cells) - expected
        raise ValueError(f"Incomplete grid: {len(missing)} missing, {len(extra)} extra")
    per_seed = []
    for seed in range(seed_start, seed_start + seed_count):
        times = {policy: statistics.median(cells[seed, rep, policy]
                                             for rep in range(repetitions))
                 for policy in POLICIES}
        row = {"seed": seed, "probe_routes": "/".join(routes[seed]),
               **{f"{policy}_median_ms": times[policy] for policy in POLICIES},
               **{f"gain_vs_{policy}_pct": 100 * (1 - times["probe_switch"] / times[policy])
                  for policy in CONTROLS}}
        per_seed.append(row)
    comparisons = {}
    for policy in CONTROLS:
        gains = [row[f"gain_vs_{policy}_pct"] for row in per_seed]
        comparisons[policy] = {"median_gain_pct": statistics.median(gains),
                               "bootstrap_95pct_ci": interval(gains),
                               "wins": sum(gain > 0 for gain in gains)}
    return per_seed, {"seed_count": seed_count, "repetitions": repetitions,
                      "primary_control": "probe_feedback",
                      "primary_improvement_supported": comparisons["probe_feedback"]["bootstrap_95pct_ci"][0] > 0,
                      "comparisons": comparisons,
                      "switch_routes": {name: sum(name in routes[seed] for seed in routes)
                                        for name in ("round_robin", "feedback")},
                      "route_instability_seeds": sum(len(set(value)) > 1 for value in routes.values())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs_csv")
    parser.add_argument("--seed-start", type=int, default=1400)
    parser.add_argument("--seed-count", type=int, default=24)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--out-seeds", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    args = parser.parse_args()
    rows, summary = analyze(args.runs_csv, args.seed_start, args.seed_count, args.repetitions)
    args.out_seeds.parent.mkdir(parents=True, exist_ok=True)
    with args.out_seeds.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    args.out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
