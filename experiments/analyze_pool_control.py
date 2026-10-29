#!/usr/bin/env python3
"""Paired block-wall and GPU-utilization analysis for the worker-pool control."""

import argparse
import csv
import json
import statistics
from pathlib import Path

from contention_analysis import interval
from persistent_pool_control import planned_blocks, validate_checkpoint


def dmon_samples(path):
    samples = {0: [], 1: []}
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) < 3 or fields[0] not in ("0", "1"):
            continue
        try:
            samples[int(fields[0])].append(int(fields[1]))
        except ValueError:
            continue
    return samples


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_json", type=Path)
    parser.add_argument("--out-prefix", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.raw_json.read_text())
    blocks = planned_blocks(data["seed_start"], data["seed_count"])
    completed = validate_checkpoint(data, blocks, data["repetitions"])
    if completed != len(blocks):
        raise ValueError(f"Incomplete pool grid: {completed}/{len(blocks)} blocks")
    by_seed = {}
    run_rows, task_rows = [], []
    occupancy = {(arm, gpu): [] for arm in ("cold", "warm") for gpu in (0, 1)}
    for block in data["blocks"]:
        by_seed.setdefault(block["seed"], {})[block["arm"]] = block
        for run in block["runs"]:
            run_rows.append({"seed": block["seed"], "arm": block["arm"],
                             "repetition": run["repetition"], "policy": run["policy"],
                             "makespan_ms": run["makespan_ms"],
                             "gpu0_event_ms": run["gpu0_event_ms"],
                             "gpu1_event_ms": run["gpu1_event_ms"]})
            for task in run["tasks"]:
                task_rows.append({"seed": block["seed"], "arm": block["arm"],
                                  "repetition": run["repetition"],
                                  "policy": run["policy"],
                                  "id": task["id"], "n": task["n"],
                                  "gpu": task["gpu"], "repeats": task["repeats"],
                                  "gpu_ms": task["gpu_ms"], "h2d_ms": task["h2d_ms"],
                                  "input_prepare_ms": task["input_prepare_ms"],
                                  "correct": task["correct"]})
        if block["gpu_util_log"]:
            samples = dmon_samples(args.raw_json.parent / block["gpu_util_log"])
            for gpu in (0, 1):
                occupancy[block["arm"], gpu].extend(samples[gpu])
    seeds = []
    for seed, pair in sorted(by_seed.items()):
        if set(pair) != {"cold", "warm"}:
            raise ValueError(f"Missing arm for seed {seed}")
        cold, warm = pair["cold"], pair["warm"]
        seeds.append({"seed": seed, "cold_wall_ms": cold["wall_ms"],
                      "warm_wall_ms": warm["wall_ms"],
                      "warm_wall_gain_pct": 100 * (1 - warm["wall_ms"] / cold["wall_ms"]),
                      "cold_batch_makespan_sum_ms": cold["batch_makespan_sum_ms"],
                      "warm_batch_makespan_sum_ms": warm["batch_makespan_sum_ms"]})
    gains = [row["warm_wall_gain_pct"] for row in seeds]
    summary = {"protocol": "persistent_pool_v1", "seed_count": len(seeds),
               "block_count": len(data["blocks"]), "run_count": len(run_rows),
               "task_count": len(task_rows), "wall_gain_median_pct": statistics.median(gains),
               "wall_gain_bootstrap_95pct_ci": interval(gains),
               "warm_faster_seeds": sum(gain > 0 for gain in gains),
               "occupancy_descriptive": {}}
    if len(run_rows) != data["seed_count"] * 2 * data["repetitions"] * 2 or len(task_rows) != len(run_rows) * 12:
        raise ValueError("Incomplete run or task count")
    for arm in ("cold", "warm"):
        summary["occupancy_descriptive"][arm] = {}
        for gpu in (0, 1):
            values = occupancy[arm, gpu]
            if not values:
                raise ValueError(f"Missing GPU samples for {arm} GPU {gpu}")
            summary["occupancy_descriptive"][arm][str(gpu)] = {
                "samples": len(values), "mean_sm_pct": statistics.mean(values),
                "pct_samples_ge_70": 100 * sum(value >= 70 for value in values) / len(values),
                "pct_samples_zero": 100 * sum(value == 0 for value in values) / len(values)}
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    write_csv(Path(f"{args.out_prefix}-seeds.csv"), seeds)
    write_csv(Path(f"{args.out_prefix}-runs.csv"), run_rows)
    write_csv(Path(f"{args.out_prefix}-tasks.csv"), task_rows)
    Path(f"{args.out_prefix}-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
