#!/usr/bin/env python3
"""Balanced-order H0/H40 × fresh/reused pinned-input campaign."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from study import write_outputs

BASE = ((False, 0), (True, 40), (False, 40), (True, 0))
POLICIES = ("round_robin", "feedback")
TASKS_PER_RUN = 12
FIXED_INPUT_SCALE = 1.0


def planned_cells(seed_start, seeds):
    cells = []
    for seed in range(seed_start, seed_start + seeds):
        shift = (seed - seed_start) % len(BASE)
        sequence = BASE[shift:] + BASE[:shift]
        cells.extend((seed, reused, workers) for reused, workers in sequence)
    return cells


def validate_checkpoint(data, cells, repetitions):
    expected_per_cell = repetitions * len(POLICIES)
    runs = data["runs"]
    if len(runs) % expected_per_cell or len(runs) > len(cells) * expected_per_cell:
        raise ValueError("Checkpoint ends in an incomplete cell or exceeds the plan")
    if data.get("protocol") != "factorial_host_input_v2" or data.get("fixed_input_scale") != FIXED_INPUT_SCALE:
        raise ValueError("Checkpoint belongs to a different protocol")
    for index in range(0, len(runs), expected_per_cell):
        seed, reused, workers = cells[index // expected_per_cell]
        group = runs[index:index + expected_per_cell]
        expected_mode = "host_to_device_reused_pinned" if reused else "host_to_device"
        observed = {(run["repetition"], run["policy"]) for run in group}
        expected = {(rep, policy) for rep in range(repetitions) for policy in POLICIES}
        if observed != expected or len(group) != len(observed):
            raise ValueError(f"Missing or duplicate policy/repetition in cell {index // expected_per_cell}")
        for run in group:
            if (run["seed"] != seed or run["mode"] != expected_mode or
                    run["host_load_workers"] != workers or
                    run.get("fixed_input_scale") != FIXED_INPUT_SCALE or
                    run["workload"] != "transfer_hidden" or
                    len(run["tasks"]) != TASKS_PER_RUN or
                    {task["id"] for task in run["tasks"]} != set(range(TASKS_PER_RUN)) or
                    sum(task["repeats"] == 32 for task in run["tasks"]) != 4 or
                    sum(task["repeats"] == 4 for task in run["tasks"]) != 8 or
                    not all(task["correct"] for task in run["tasks"])):
                raise ValueError(f"Invalid run in cell {index // expected_per_cell}")
    return len(runs) // expected_per_cell


def save_checkpoint(data, path):
    pending = path.with_name(path.stem + ".pending" + path.suffix)
    write_outputs(data, pending)
    os.replace(pending, path)
    os.replace(pending.with_suffix(".csv"), path.with_suffix(".csv"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=1500)
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--plan-only", action="store_true",
                        help="Print the exact grid without importing CUDA or writing files")
    parser.add_argument("--resume", action="store_true",
                        help="Resume a complete-cell checkpoint from the same pod")
    args = parser.parse_args()
    if args.seeds < 1 or args.repetitions < 1:
        raise ValueError("Seeds and repetitions must be positive")
    cells = planned_cells(args.seed_start, args.seeds)
    if args.plan_only:
        print(json.dumps({"protocol": "factorial_host_input_v2",
                          "fixed_input_scale": FIXED_INPUT_SCALE,
                          "cells": [{"seed": seed, "reused": reused, "host_load_workers": workers}
                                    for seed, reused, workers in cells],
                          "cell_count": len(cells),
                          "run_count": len(cells) * args.repetitions * len(POLICIES),
                          "task_count": len(cells) * args.repetitions * len(POLICIES) * TASKS_PER_RUN},
                         indent=2))
        return
    if args.out.suffix != ".json":
        raise ValueError("--out must be a .json file")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.resume:
        if not args.out.exists():
            raise FileNotFoundError(args.out)
        aggregate = json.loads(args.out.read_text())
        if (aggregate.get("seed_start") != args.seed_start or
                aggregate.get("seed_count") != args.seeds or
                aggregate.get("repetitions") != args.repetitions):
            raise ValueError("Checkpoint grid does not match command arguments")
        completed = validate_checkpoint(aggregate, cells, args.repetitions)
    else:
        if args.out.exists():
            raise FileExistsError(f"Refusing to overwrite {args.out}; use --resume")
        aggregate = {"schema_version": 2, "protocol": "factorial_host_input_v2",
                 "seed_start": args.seed_start, "seed_count": args.seeds,
                 "repetitions": args.repetitions, "base_sequence": [list(pair) for pair in BASE],
                 "fixed_input_scale": FIXED_INPUT_SCALE,
                 "runs": []}
        completed = 0
    with tempfile.TemporaryDirectory() as temporary:
        for seed, reused, workers in cells[completed:]:
            mode = "reused" if reused else "fresh"
            output = Path(temporary) / f"seed{seed}-{mode}-h{workers}.json"
            command = [sys.executable, str(Path(__file__).with_name("study.py")),
                       "--out", str(output), "--workload", "transfer_hidden",
                       "--seed-start", str(seed), "--seeds", "1",
                       "--repetitions", str(args.repetitions),
                       "--policies", *POLICIES,
                       "--host-load-workers", str(workers),
                       "--fixed-input-scale", str(FIXED_INPUT_SCALE)]
            if reused:
                command.append("--reuse-pinned-inputs")
            subprocess.run(command, check=True)
            block = json.loads(output.read_text())
            if len(block["runs"]) != len(POLICIES) * args.repetitions:
                raise ValueError(f"Incomplete block: seed {seed}, {mode}, H{workers}")
            aggregate["runs"].extend(block["runs"])
            validate_checkpoint(aggregate, cells, args.repetitions)
            save_checkpoint(aggregate, args.out)
            print(f"checkpoint {seed} {mode} H{workers}: {len(aggregate['runs'])} runs", flush=True)
    print(f"saved {args.out} and {args.out.with_suffix('.csv')}", flush=True)


if __name__ == "__main__":
    main()
