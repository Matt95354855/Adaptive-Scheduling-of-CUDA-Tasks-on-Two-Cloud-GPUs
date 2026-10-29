#!/usr/bin/env python3
"""Validate the finished grid and hash every artifact before pod shutdown."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from factorial_host_input import planned_cells, validate_checkpoint


PROFILE_CELLS = [(mode, workers) for mode in ("fresh", "reused")
                 for workers in (0, 40)]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--require-nsight", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.results_dir
    raw = root / "factorial-v2.json"
    data = json.loads(raw.read_text())
    cells = planned_cells(1500, 12)
    completed = validate_checkpoint(data, cells, 2)
    if completed != 48 or len(data["runs"]) != 192:
        raise ValueError(f"Incomplete factorial grid: {completed}/48 cells")
    task_csv = root / "factorial-v2-tasks.csv"
    with task_csv.open(newline="") as stream:
        task_count = sum(1 for _ in csv.DictReader(stream))
    if task_count != 2304:
        raise ValueError(f"Expected 2304 task records, found {task_count}")
    summary = json.loads((root / "factorial-v2-summary.json").read_text())
    if summary.get("seed_count") != 12 or len(summary.get("per_seed", [])) != 12:
        raise ValueError("Incomplete statistical summary")
    required = ["factorial-v2-preflight.json", "factorial-v2-code-revision.txt",
                "factorial-v2-start.txt", "factorial-v2-end.txt",
                "factorial-v2-smoke-fresh.json", "factorial-v2-smoke-reused.json",
                "factorial-v2-gpu-util.log", "factorial-v2-utilization.json"]
    for name in required:
        if not (root / name).is_file():
            raise FileNotFoundError(root / name)
    trace_count = 0
    for mode, workers in PROFILE_CELLS:
        prefix = f"nsys-factorial-v2-{mode}-h{workers}"
        trace = root / f"{prefix}.nsys-rep"
        if trace.is_file():
            trace_count += 1
            for suffix in ("json", "csv", "profile.log", "stats.log"):
                if not (root / f"{prefix}.{suffix}").is_file():
                    raise FileNotFoundError(root / f"{prefix}.{suffix}")
            if len(list(root.glob(f"{prefix}*cuda_gpu_kern_sum.csv"))) != 1:
                raise ValueError(f"Missing CUDA kernel report for {prefix}")
            if len(list(root.glob(f"{prefix}*cuda_gpu_mem_time_sum.csv"))) != 1:
                raise ValueError(f"Missing CUDA memory report for {prefix}")
            if len(list(root.glob(f"{prefix}*nvtx_sum.csv"))) != 1:
                raise ValueError(f"Missing NVTX report for {prefix}")
    if args.require_nsight and trace_count != 4:
        raise ValueError(f"Expected four Nsight traces, found {trace_count}")
    files = sorted(path for path in root.glob("*") if path.is_file()
                   and (path.name.startswith("factorial-v2") or
                        path.name.startswith("nsys-factorial-v2"))
                   and ".pending." not in path.name and path != args.out)
    manifest = {"protocol": "factorial_host_input_v2", "complete_runs": 192,
                "complete_tasks": task_count, "nsight_traces": trace_count,
                "total_bytes": sum(path.stat().st_size for path in files),
                "files": [{"name": path.name, "bytes": path.stat().st_size,
                           "sha256": sha256(path)} for path in files]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in
                      ("complete_runs", "complete_tasks", "nsight_traces", "total_bytes")},
                     indent=2))


if __name__ == "__main__":
    main()
