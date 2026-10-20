#!/usr/bin/env python3
"""Relate hidden execution-count imbalance to paired policy results."""
import argparse
import csv
import random
import statistics
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paired_csv")
    args = parser.parse_args()
    with open(args.paired_csv, newline="") as file:
        rows = list(csv.DictReader(file))
    by_seed = defaultdict(list)
    for row in rows:
        by_seed[int(row["seed"])].append(row)
    print("seed,rr_repeat_gpu0,rr_repeat_gpu1,rr_ratio,rr_ms,feedback_ms,feedback_change_pct")
    for seed, items in sorted(by_seed.items()):
        repeats = [12] * 8 + [1] * 16
        random.Random(seed ^ 0x5A17).shuffle(repeats)
        a, b = sum(repeats[::2]), sum(repeats[1::2])
        rr = statistics.median(float(x["round_robin"]) for x in items)
        feedback = statistics.median(float(x["feedback"]) for x in items)
        print(f"{seed},{a},{b},{max(a,b)/min(a,b):.3f},"
              f"{rr:.3f},{feedback:.3f},{100*(feedback/rr-1):+.2f}")


if __name__ == "__main__":
    main()
