#!/usr/bin/env python3
"""Paired two-GPU scheduling study; requires CUDA-enabled PyTorch."""
import argparse
import csv
import json
import multiprocessing as mp
import queue
import random
import statistics
import time
from pathlib import Path

POLICIES = ("single", "round_robin", "size_fifo", "size_lpt", "calibrated", "feedback")
SIZES = {"micro": [256, 512, 768, 1024] * 8,
         "compute": [1024, 2048, 3072, 4096] * 6}


def plan(policy, tasks, calibration):
    """Return GPU assignment, or None for online completion feedback."""
    if policy == "feedback":
        return None
    if policy == "single":
        return {i: 0 for i, _ in tasks}
    if policy == "round_robin":
        return {i: i % 2 for i, _ in tasks}
    work = [0.0, 0.0]
    assignment = {}
    sequence = sorted(tasks, key=lambda task: -task[1]) if policy == "size_lpt" else tasks
    for task_id, n in sequence:
        if policy == "calibrated":
            gpu = min(range(2), key=lambda g: (work[g] + calibration[g][n], g))
            work[gpu] += calibration[gpu][n]
        else:
            gpu = min(range(2), key=lambda g: (work[g], g))
            work[gpu] += n ** 3
        assignment[task_id] = gpu
    return assignment


def worker(gpu, requests, replies, sizes):
    import torch
    torch.cuda.set_device(gpu)
    device = f"cuda:{gpu}"
    calibration = {}
    # Per-size and per-GPU timing, outside the measured batch. The timed task
    # still includes allocation and validation, so this is only a cost proxy.
    for n in sizes:
        a = torch.ones((n, n), device=device)
        b = torch.ones((n, n), device=device)
        torch.mm(a, b)
        samples = []
        for _ in range(5):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            torch.mm(a, b)
            end.record()
            end.synchronize()
            samples.append(begin.elapsed_time(end))
        calibration[n] = statistics.median(samples)
    torch.cuda.synchronize(gpu)
    replies.put({"kind": "ready", "gpu": gpu, "calibration": calibration})
    while True:
        task = requests.get()
        if task is None:
            return
        task_id, n, seed = task
        start_ns = time.monotonic_ns()
        scale = ((seed % 7) + 1) / 8.0
        a = torch.full((n, n), scale, device=device)
        b = torch.ones((n, n), device=device)
        torch.cuda.synchronize(gpu)
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.nvtx.range_push(f"task_{task_id}_n{n}_gpu{gpu}")
        begin.record()
        c = torch.mm(a, b)
        end.record()
        end.synchronize()
        torch.cuda.nvtx.range_pop()
        actual = float(c.sum().item())
        finish_ns = time.monotonic_ns()
        expected = scale * n ** 3
        replies.put({"kind": "done", "id": task_id, "n": n, "gpu": gpu,
                     "start_ns": start_ns, "finish_ns": finish_ns,
                     "gpu_ms": begin.elapsed_time(end),
                     "correct": abs(actual - expected) <= max(1.0, expected * 1e-4)})


def run(policy, tasks):
    ctx = mp.get_context("spawn")
    requests = [ctx.Queue() for _ in range(2)]
    replies = ctx.Queue()
    processes = [ctx.Process(target=worker, args=(g, requests[g], replies,
                                                sorted({n for _, n, _ in tasks}))) for g in range(2)]
    for process in processes:
        process.start()
    try:
        calibration = {}
        for _ in processes:
            msg = replies.get(timeout=120)
            if msg["kind"] != "ready":
                raise RuntimeError(msg)
            calibration[msg["gpu"]] = msg["calibration"]
        assignment = plan(policy, [(i, n) for i, n, _ in tasks], calibration)
        task_by_id = {i: (i, n, seed) for i, n, seed in tasks}
        sent, completed, dispatch_ns, dispatch_times = 0, [], 0, {}
        begin_ns = time.monotonic_ns()

        def dispatch(gpu, task):
            nonlocal sent, dispatch_ns
            now = time.monotonic_ns()
            requests[gpu].put(task)
            dispatch_ns += time.monotonic_ns() - now
            dispatch_times[task[0]] = now
            sent += 1

        if assignment is None:
            for gpu, task in enumerate(tasks[:2]):
                dispatch(gpu, task)
        else:
            for task in tasks:
                dispatch(assignment[task[0]], task)
        while len(completed) < len(tasks):
            try:
                msg = replies.get(timeout=120)
            except queue.Empty as exc:
                raise RuntimeError("Worker timeout") from exc
            if msg["kind"] != "done":
                raise RuntimeError(msg)
            msg["dispatch_ns"] = dispatch_times[msg["id"]]
            msg["reply_ns"] = time.monotonic_ns()
            completed.append(msg)
            if assignment is None and sent < len(tasks):
                dispatch(msg["gpu"], tasks[sent])
        end_ns = time.monotonic_ns()
        if not all(item["correct"] for item in completed):
            raise RuntimeError("Incorrect GEMM checksum")
        by_gpu = [[x for x in completed if x["gpu"] == g] for g in range(2)]
        makespan_ms = (end_ns - begin_ns) / 1e6
        record = {"policy": policy, "makespan_ms": makespan_ms,
                  "dispatch_ms": dispatch_ns / 1e6,
                  "gpu0_tasks": len(by_gpu[0]), "gpu1_tasks": len(by_gpu[1]),
                  "gpu0_n3": sum(x["n"] ** 3 for x in by_gpu[0]),
                  "gpu1_n3": sum(x["n"] ** 3 for x in by_gpu[1]),
                  "gpu0_event_ms": sum(x["gpu_ms"] for x in by_gpu[0]),
                  "gpu1_event_ms": sum(x["gpu_ms"] for x in by_gpu[1]),
                  "host_task_ms_median": statistics.median((x["finish_ns"] - x["start_ns"]) / 1e6 for x in completed),
                  "calibration": calibration,
                  "tasks": sorted(completed, key=lambda x: x["id"])}
        return record
    finally:
        for request in requests:
            request.put(None)
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)


def write_outputs(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    fields = ("workload", "seed", "repetition", "order", "policy", "makespan_ms", "dispatch_ms",
              "gpu0_tasks", "gpu1_tasks", "gpu0_n3", "gpu1_n3", "gpu0_event_ms", "gpu1_event_ms",
              "host_task_ms_median")
    with path.with_suffix(".csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for item in data["runs"]:
            writer.writerow({key: item[key] for key in fields})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workload", choices=SIZES, default="compute")
    parser.add_argument("--seed-start", type=int, default=100)
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=POLICIES)
    args = parser.parse_args()
    import torch
    if torch.cuda.device_count() < 2:
        raise RuntimeError("Two CUDA GPUs required")
    data = {"schema_version": 1, "workload": args.workload, "seed_start": args.seed_start,
            "seed_count": args.seeds, "repetitions": args.repetitions,
            "gpus": [torch.cuda.get_device_name(g) for g in range(2)],
            "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "runs": []}
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        sizes = SIZES[args.workload].copy()
        random.Random(seed).shuffle(sizes)
        tasks = [(i, n, seed + i) for i, n in enumerate(sizes)]
        for repetition in range(args.repetitions):
            order = list(args.policies)
            random.Random(seed * 1009 + repetition).shuffle(order)
            for index, policy in enumerate(order):
                record = run(policy, tasks)
                record.update(workload=args.workload, seed=seed, repetition=repetition,
                              order=index)
                data["runs"].append(record)
                write_outputs(data, args.out)
                print(f"{args.workload} seed={seed} rep={repetition} {policy}: "
                      f"{record['makespan_ms']:.3f} ms", flush=True)
    print(f"saved {args.out} and {args.out.with_suffix('.csv')}", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
