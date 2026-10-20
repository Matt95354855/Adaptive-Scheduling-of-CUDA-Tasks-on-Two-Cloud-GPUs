#!/usr/bin/env python3
"""Summarize work-balance proxies from study.py's run-level CSV."""
import argparse
import csv
import statistics
from collections import defaultdict


def med(items):
    return statistics.median(items)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    args = parser.parse_args()
    with open(args.csv, newline="") as file:
        rows = list(csv.DictReader(file))
    groups = defaultdict(list)
    for row in rows:
        groups[row["policy"]].append(row)
    print("policy,n_runs,n3_imbalance_ratio,event_imbalance_ms,max_gpu_event_ms,gpu0_tasks,gpu1_tasks")
    for policy, items in sorted(groups.items()):
        n3 = [(float(x["gpu0_n3"]), float(x["gpu1_n3"])) for x in items]
        event = [(float(x["gpu0_event_ms"]), float(x["gpu1_event_ms"])) for x in items]
        ratios = [max(a, b) / min(a, b) if min(a, b) else float("nan") for a, b in n3]
        ratios = [x for x in ratios if x == x]
        print(f"{policy},{len(items)},"
              f"{med(ratios) if ratios else float('nan'):.3f},"
              f"{med(abs(a-b) for a,b in event):.3f},"
              f"{med(max(a,b) for a,b in event):.3f},"
              f"{med(float(x['gpu0_tasks']) for x in items):.1f},"
              f"{med(float(x['gpu1_tasks']) for x in items):.1f}")


if __name__ == "__main__":
    main()
