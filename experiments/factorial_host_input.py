#!/usr/bin/env python3
"""Balanced-order H0/H40 × fresh/reused pinned-input campaign."""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from study import write_outputs

BASE = ((False, 0), (True, 40), (False, 40), (True, 0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=1500)
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--repetitions", type=int, default=2)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    aggregate = {"schema_version": 1, "protocol": "factorial_host_input",
                 "seed_start": args.seed_start, "seed_count": args.seeds,
                 "repetitions": args.repetitions, "base_sequence": [list(pair) for pair in BASE],
                 "runs": []}
    with tempfile.TemporaryDirectory() as temporary:
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            shift = (seed - args.seed_start) % len(BASE)
            sequence = BASE[shift:] + BASE[:shift]
            for reused, workers in sequence:
                mode = "reused" if reused else "fresh"
                output = Path(temporary) / f"seed{seed}-{mode}-h{workers}.json"
                command = [sys.executable, str(Path(__file__).with_name("study.py")),
                           "--out", str(output), "--workload", "transfer_hidden",
                           "--seed-start", str(seed), "--seeds", "1",
                           "--repetitions", str(args.repetitions),
                           "--policies", "round_robin", "feedback",
                           "--host-load-workers", str(workers)]
                if reused:
                    command.append("--reuse-pinned-inputs")
                subprocess.run(command, check=True)
                block = json.loads(output.read_text())
                if len(block["runs"]) != 2 * args.repetitions:
                    raise ValueError(f"Incomplete block: seed {seed}, {mode}, H{workers}")
                aggregate["runs"].extend(block["runs"])
                write_outputs(aggregate, args.out)
                print(f"checkpoint {seed} {mode} H{workers}: {len(aggregate['runs'])} runs", flush=True)
    print(f"saved {args.out} and {args.out.with_suffix('.csv')}", flush=True)


if __name__ == "__main__":
    main()
