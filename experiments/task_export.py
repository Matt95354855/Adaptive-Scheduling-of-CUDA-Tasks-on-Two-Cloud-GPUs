#!/usr/bin/env python3
"""Export task-level timing from study.py JSON to a portable CSV."""
import argparse
import csv
import json
from pathlib import Path


FIELDS = (
    "workload", "host_load_workers", "seed", "repetition", "policy",
    "task_id", "gpu", "repeats", "n", "correct", "dispatch_rel_ms",
    "worker_start_rel_ms", "worker_finish_rel_ms", "queue_wait_ms",
    "host_task_ms", "input_prepare_ms", "h2d_ms", "gpu_ms", "reply_delay_ms",
)


def export(source, destination):
    runs = json.loads(Path(source).read_text())["runs"]
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for run in runs:
            tasks = run["tasks"]
            origin = min(task["dispatch_ns"] for task in tasks)
            for task in tasks:
                if not task["correct"]:
                    raise ValueError("Incorrect result in source JSON")
                writer.writerow({
                    "workload": run["workload"],
                    "host_load_workers": run["host_load_workers"],
                    "seed": run["seed"], "repetition": run["repetition"],
                    "policy": run["policy"], "task_id": task["id"],
                    "gpu": task["gpu"], "repeats": task["repeats"],
                    "n": task["n"], "correct": task["correct"],
                    "dispatch_rel_ms": (task["dispatch_ns"] - origin) / 1e6,
                    "worker_start_rel_ms": (task["start_ns"] - origin) / 1e6,
                    "worker_finish_rel_ms": (task["finish_ns"] - origin) / 1e6,
                    "queue_wait_ms": (task["start_ns"] - task["dispatch_ns"]) / 1e6,
                    "host_task_ms": (task["finish_ns"] - task["start_ns"]) / 1e6,
                    "input_prepare_ms": task["input_prepare_ms"],
                    "h2d_ms": task["h2d_ms"], "gpu_ms": task["gpu_ms"],
                    "reply_delay_ms": (task["reply_ns"] - task["finish_ns"]) / 1e6,
                })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json")
    parser.add_argument("csv")
    args = parser.parse_args()
    export(args.json, args.csv)


if __name__ == "__main__":
    main()
