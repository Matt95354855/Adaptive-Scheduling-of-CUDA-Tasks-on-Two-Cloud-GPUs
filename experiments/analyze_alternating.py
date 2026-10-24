#!/usr/bin/env python3
"""Analyze a complete, alternating H0/H40 campaign from one raw JSON file."""

import argparse
import csv
import json
import statistics
from pathlib import Path

from contention_analysis import analyze
from task_metrics import derive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_json")
    parser.add_argument("--seed-start", type=int, default=1200)
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--out-prefix", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(Path(args.raw_json).read_text())
    runs = data["runs"]
    if len(runs) != args.seeds * args.repetitions * 2 * 2:
        raise ValueError("Incomplete alternating grid")
    orders = {}
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        seen = [run["host_load_workers"] for run in runs if run["seed"] == seed]
        expected = ([0] * (args.repetitions * 2) + [40] * (args.repetitions * 2)
                    if (seed - args.seed_start) % 2 == 0 else
                    [40] * (args.repetitions * 2) + [0] * (args.repetitions * 2))
        if seen != expected:
            raise ValueError(f"Unexpected condition order for seed {seed}: {seen}")
        orders[seed] = "H0→H40" if expected[0] == 0 else "H40→H0"
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for workers in (0, 40):
        subset = [run for run in runs if run["host_load_workers"] == workers]
        path = Path(f"{args.out_prefix}-h{workers}.csv")
        fields = ("workload", "host_load_workers", "seed", "repetition", "policy", "makespan_ms")
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key: run[key] for key in fields} for run in subset)
    result = analyze(f"{args.out_prefix}-h0.csv", f"{args.out_prefix}-h40.csv")
    if result["seed_count"] != args.seeds or result["repetitions_per_cell"] != args.repetitions:
        raise ValueError("Analysis grid differs from protocol")
    result["condition_order_by_seed"] = orders
    with Path(f"{args.out_prefix}-summary.json").open("w") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    metrics = [{"host_load_workers": run["host_load_workers"], **derive(run)} for run in runs]
    with Path(f"{args.out_prefix}-task-metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    diagnostics = {}
    for workers in (0, 40):
        for policy in ("round_robin", "feedback"):
            subset = [row for row in metrics if row["host_load_workers"] == workers and row["policy"] == policy]
            diagnostics[f"h{workers}_{policy}"] = {
                key: statistics.median(row[key] for row in subset)
                for key in ("makespan_ms", "gpu_event_ms_sum", "h2d_ms_sum", "input_prepare_ms_median",
                            "gpu0_gap_ms", "gpu1_gap_ms")}
    with Path(f"{args.out_prefix}-diagnostics.json").open("w") as stream:
        json.dump(diagnostics, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"interaction": result["interaction"], "diagnostics": diagnostics}, indent=2))


if __name__ == "__main__":
    main()
