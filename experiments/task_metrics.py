#!/usr/bin/env python3
"""Derive per-run worker/queue timing from study.py's task records."""
import argparse
import csv
import json
import statistics


def derive(run):
    tasks = run["tasks"]
    by_gpu = [[task for task in tasks if task["gpu"] == g] for g in range(2)]
    row = {key: run[key] for key in ("workload", "seed", "repetition", "policy", "makespan_ms")}
    for gpu in range(2):
        ordered = sorted(by_gpu[gpu], key=lambda x: x["start_ns"])
        row[f"gpu{gpu}_worker_ms"] = sum((x["finish_ns"] - x["start_ns"]) / 1e6 for x in ordered)
        row[f"gpu{gpu}_gap_ms"] = sum(max(0, (b["start_ns"] - a["finish_ns"]) / 1e6)
                                       for a, b in zip(ordered, ordered[1:]))
    row["queue_wait_ms_median"] = statistics.median(
        (x["start_ns"] - x["dispatch_ns"]) / 1e6 for x in tasks)
    row["reply_delay_ms_median"] = statistics.median(
        (x["reply_ns"] - x["finish_ns"]) / 1e6 for x in tasks)
    row["gpu_event_ms_sum"] = sum(x["gpu_ms"] for x in tasks)
    row["h2d_ms_sum"] = sum(x.get("h2d_ms", 0.0) for x in tasks)
    row["input_prepare_ms_median"] = statistics.median(
        x.get("input_prepare_ms", 0.0) for x in tasks)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json")
    parser.add_argument("csv")
    args = parser.parse_args()
    with open(args.json) as file:
        data = json.load(file)
    rows = [derive(run) for run in data["runs"]]
    with open(args.csv, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
