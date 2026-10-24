#!/usr/bin/env python3
"""Summarize raw nvidia-smi dmon samples and CUDA-event occupancy proxies."""

import argparse
import csv
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dmon", type=Path)
    parser.add_argument("runs", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    by_gpu = {0: [], 1: []}
    samples = []
    for line in args.dmon.read_text().splitlines():
        fields = line.split()
        if not fields or fields[0] not in ("0", "1") or len(fields) < 3:
            continue
        try:
            gpu, sm, mem = int(fields[0]), int(fields[1]), int(fields[2])
        except ValueError:
            continue
        sample = {"sample_index": len(by_gpu[gpu]), "gpu": gpu,
                  "sm_pct": sm, "mem_pct": mem}
        by_gpu[gpu].append(sample)
        samples.append(sample)
    if not all(by_gpu.values()):
        raise ValueError("No complete GPU utilization samples")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.with_suffix(".csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=samples[0])
        writer.writeheader()
        writer.writerows(samples)
    runs = json.loads(args.runs.read_text())["runs"]
    result = {"sample_interval_s": 1, "samples": {}, "cuda_event_fraction": {},
              "run_count": len(runs), "task_count": sum(len(row["tasks"]) for row in runs)}
    for gpu in (0, 1):
        values = [row["sm_pct"] for row in by_gpu[gpu]]
        result["samples"][str(gpu)] = {
            "count": len(values), "mean_sm_pct": statistics.mean(values),
            "median_sm_pct": statistics.median(values), "max_sm_pct": max(values),
            "pct_samples_ge_70": 100 * sum(v >= 70 for v in values) / len(values),
            "pct_samples_zero": 100 * sum(v == 0 for v in values) / len(values)}
        fractions = [100 * row[f"gpu{gpu}_event_ms"] / row["makespan_ms"] for row in runs]
        result["cuda_event_fraction"][str(gpu)] = {
            "median_pct": statistics.median(fractions),
            "min_pct": min(fractions), "max_pct": max(fractions)}
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
