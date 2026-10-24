#!/usr/bin/env python3
"""Run complete H0/H40 pairs in alternating order, checkpointing raw results."""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from study import write_outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=1200)
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--host-load-workers", type=int, default=40)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    aggregate = {"schema_version": 1, "protocol": "alternating_h0_h40",
                 "seed_start": args.seed_start, "seed_count": args.seeds,
                 "repetitions": args.repetitions, "host_load_workers": args.host_load_workers,
                 "runs": []}
    with tempfile.TemporaryDirectory() as temporary:
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            conditions = (0, args.host_load_workers) if (seed - args.seed_start) % 2 == 0 else (args.host_load_workers, 0)
            for workers in conditions:
                output = Path(temporary) / f"seed{seed}-h{workers}.json"
                command = [sys.executable, str(Path(__file__).with_name("study.py")),
                           "--out", str(output), "--workload", "transfer_hidden",
                           "--seed-start", str(seed), "--seeds", "1",
                           "--repetitions", str(args.repetitions),
                           "--policies", "round_robin", "feedback",
                           "--host-load-workers", str(workers)]
                subprocess.run(command, check=True)
                block = json.loads(output.read_text())
                if len(block["runs"]) != 2 * args.repetitions:
                    raise ValueError(f"Incomplete block for seed {seed}, H{workers}")
                aggregate["runs"].extend(block["runs"])
                write_outputs(aggregate, args.out)
                print(f"checkpoint {seed} H{workers}: {len(aggregate['runs'])} runs", flush=True)
    print(f"saved {args.out} and {args.out.with_suffix('.csv')}", flush=True)


if __name__ == "__main__":
    main()
