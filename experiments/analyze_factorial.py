#!/usr/bin/env python3
"""Paired seed-level analysis of the host contention × pinned-input control."""

import argparse
import csv
import json
import statistics
from pathlib import Path

from contention_analysis import analyze, interval
from factorial_host_input import BASE, FIXED_INPUT_SCALE, planned_cells, validate_checkpoint
from task_metrics import derive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_json")
    parser.add_argument("--seed-start", type=int, default=1500)
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--out-prefix", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(Path(args.raw_json).read_text())
    runs = data["runs"]
    if len(runs) != args.seeds * args.repetitions * 2 * 2 * 2:
        raise ValueError("Incomplete factorial grid")
    if (data.get("seed_start") != args.seed_start or
            data.get("seed_count") != args.seeds or
            data.get("repetitions") != args.repetitions or
            data.get("fixed_input_scale") != FIXED_INPUT_SCALE):
        raise ValueError("Factorial metadata does not match the fixed-input protocol")
    validate_checkpoint(data, planned_cells(args.seed_start, args.seeds), args.repetitions)
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        shift = (seed - args.seed_start) % len(BASE)
        sequence = BASE[shift:] + BASE[:shift]
        actual = [(run["mode"] == "host_to_device_reused_pinned", run["host_load_workers"])
                  for run in runs if run["seed"] == seed]
        expected = [cell for cell in sequence for _ in range(args.repetitions * 2)]
        if actual != expected:
            raise ValueError(f"Unexpected condition order for seed {seed}")
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    for reused, mode in ((False, "fresh"), (True, "reused")):
        for workers in (0, 40):
            selected = [run for run in runs
                        if (run["mode"] == "host_to_device_reused_pinned") == reused
                        and run["host_load_workers"] == workers]
            fields = ("workload", "host_load_workers", "seed", "repetition", "policy", "makespan_ms")
            path = Path(f"{args.out_prefix}-{mode}-h{workers}.csv")
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows({field: run[field] for field in fields} for run in selected)
        results[mode] = analyze(f"{args.out_prefix}-{mode}-h0.csv",
                                f"{args.out_prefix}-{mode}-h40.csv")
    fresh = {row["seed"]: row for row in results["fresh"]["per_seed"]}
    reused = {row["seed"]: row for row in results["reused"]["per_seed"]}
    differences = [reused[seed]["delta_gain_pct"] - fresh[seed]["delta_gain_pct"]
                   for seed in sorted(fresh)]
    difference_ci = interval(differences)
    summary = {"seed_count": args.seeds, "repetitions": args.repetitions,
               "fresh_interaction": results["fresh"]["interaction"],
               "reused_interaction": results["reused"]["interaction"],
               "primary_difference_of_interactions_pct_points": {
                   "median": statistics.median(differences),
                   "bootstrap_95pct_ci": difference_ci,
                   "positive_by_protocol": difference_ci[0] > 0},
               "per_seed": [{"seed": seed, "fresh_delta_gain_pct": fresh[seed]["delta_gain_pct"],
                             "reused_delta_gain_pct": reused[seed]["delta_gain_pct"],
                             "difference_pct_points": reused[seed]["delta_gain_pct"] - fresh[seed]["delta_gain_pct"]}
                            for seed in sorted(fresh)]}
    metrics = [{"mode": run["mode"], "host_load_workers": run["host_load_workers"],
                **derive(run)} for run in runs]
    with Path(f"{args.out_prefix}-task-metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    diagnostics = {}
    for mode in ("host_to_device", "host_to_device_reused_pinned"):
        for workers in (0, 40):
            for policy in ("round_robin", "feedback"):
                subset = [row for row in metrics if row["mode"] == mode and
                          row["host_load_workers"] == workers and row["policy"] == policy]
                diagnostics[f"{mode}_h{workers}_{policy}"] = {
                    key: statistics.median(row[key] for row in subset)
                    for key in ("makespan_ms", "gpu_event_ms_sum", "h2d_ms_sum", "input_prepare_ms_median")}
    summary["diagnostics_descriptive"] = diagnostics
    Path(f"{args.out_prefix}-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary["primary_difference_of_interactions_pct_points"], indent=2))


if __name__ == "__main__":
    main()
