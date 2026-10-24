#!/usr/bin/env python3
"""Post-hoc diagnostics of the fixed online probe; never used to choose runs."""

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs_json", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    by_seed = defaultdict(list)
    data = json.loads(args.runs_json.read_text())
    for run in data["runs"]:
        by_seed[run["seed"]].append(run)
    rows = []
    for seed, runs in sorted(by_seed.items()):
        rr = next(run for run in runs if run["policy"] == "round_robin")
        long_counts = [sum(task["repeats"] == 32 and task["gpu"] == gpu
                           for task in rr["tasks"]) for gpu in (0, 1)]
        switch = [run for run in runs if run["policy"] == "probe_switch"]
        if len(switch) != 2 or len({run["probe_route"] for run in switch}) != 1:
            raise ValueError(f"Unstable or incomplete route for seed {seed}")
        route = switch[0]["probe_route"]
        times = {policy: statistics.median(run["makespan_ms"] for run in runs
                                            if run["policy"] == policy)
                 for policy in ("probe_rr", "probe_feedback", "probe_switch")}
        faster = "round_robin" if times["probe_rr"] <= times["probe_feedback"] else "feedback"
        rows.append({"seed": seed, "rr_long_gpu0": long_counts[0],
                     "rr_long_gpu1": long_counts[1],
                     "rr_imbalanced": int(long_counts[0] != long_counts[1]),
                     "chosen_route": route, "faster_probe_control": faster,
                     "choice_matches_faster_control": int(route == faster),
                     "probe_rr_median_ms": times["probe_rr"],
                     "probe_feedback_median_ms": times["probe_feedback"],
                     "probe_switch_median_ms": times["probe_switch"]})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.with_suffix(".csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    confusion = Counter((row["rr_imbalanced"], row["chosen_route"]) for row in rows)
    summary = {"analysis": "post_hoc_descriptive", "seed_count": len(rows),
               "route_vs_full_rr_imbalance": {
                   f"imbalance_{imbalance}_{route}": confusion[imbalance, route]
                   for imbalance in (0, 1) for route in ("round_robin", "feedback")},
               "choice_matches_faster_probe_control": sum(row["choice_matches_faster_control"]
                                                      for row in rows),
               "faster_probe_control_counts": dict(Counter(row["faster_probe_control"]
                                                           for row in rows))}
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
