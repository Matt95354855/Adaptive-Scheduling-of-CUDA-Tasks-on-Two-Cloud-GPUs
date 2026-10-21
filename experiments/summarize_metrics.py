#!/usr/bin/env python3
"""Descriptive medians for task_metrics.py output (runs, not independent seeds)."""
import argparse
import csv
import statistics
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    args = parser.parse_args()
    with open(args.csv, newline="") as file:
        rows = list(csv.DictReader(file))
    groups = defaultdict(list)
    for row in rows:
        groups[row["policy"]].append(row)
    print("policy,n_runs,makespan_ms,worker_gap_sum_ms,gpu_event_sum_ms,h2d_ms_sum,input_prepare_ms_median,queue_wait_median_ms,reply_delay_median_ms")
    for policy, items in sorted(groups.items()):
        def med(key):
            return statistics.median(float(item[key]) for item in items)
        gaps = [float(item["gpu0_gap_ms"]) + float(item["gpu1_gap_ms"]) for item in items]
        print(f"{policy},{len(items)},{med('makespan_ms'):.3f},"
              f"{statistics.median(gaps):.3f},{med('gpu_event_ms_sum'):.3f},"
              f"{med('h2d_ms_sum'):.3f},{med('input_prepare_ms_median'):.3f},{med('queue_wait_ms_median'):.3f},{med('reply_delay_ms_median'):.3f}")


if __name__ == "__main__":
    main()
