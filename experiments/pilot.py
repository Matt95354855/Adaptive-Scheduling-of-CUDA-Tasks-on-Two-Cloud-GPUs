#!/usr/bin/env python3
"""Small, reproducible two-GPU scheduling pilot. Requires PyTorch with CUDA."""
import argparse
import json
import multiprocessing as mp
import os
import queue
import statistics
import time
from pathlib import Path


def worker(gpu, requests, replies):
    import torch

    torch.cuda.set_device(gpu)
    # Warm-up is outside the measured batch, identically for every policy.
    a = torch.ones((256, 256), device=f"cuda:{gpu}")
    torch.mm(a, a)
    torch.cuda.synchronize(gpu)
    replies.put({"kind": "ready", "gpu": gpu})
    while True:
        task = requests.get()
        if task is None:
            return
        task_id, n, seed = task
        start_ns = time.monotonic_ns()
        scale = ((seed % 7) + 1) / 8.0
        a = torch.full((n, n), scale, device=f"cuda:{gpu}")
        b = torch.ones((n, n), device=f"cuda:{gpu}")
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
        expected = scale * (n ** 3)
        replies.put({"kind": "done", "id": task_id, "gpu": gpu, "n": n,
                     "start_ns": start_ns, "finish_ns": finish_ns,
                     "gpu_ms": begin.elapsed_time(end),
                     "correct": abs(actual - expected) <= max(1.0, expected * 1e-4),
                     "actual": actual, "expected": expected})


def run_policy(policy, tasks, gpu_count):
    ctx = mp.get_context("spawn")
    requests = [ctx.Queue() for _ in range(gpu_count)]
    replies = ctx.Queue()
    processes = [ctx.Process(target=worker, args=(g, requests[g], replies)) for g in range(gpu_count)]
    for process in processes:
        process.start()
    try:
        for _ in processes:
            msg = replies.get(timeout=120)
            if msg["kind"] != "ready":
                raise RuntimeError(f"Expected ready: {msg}")
        sent = 0
        completed = []
        dispatch_ns = 0
        begin_ns = time.monotonic_ns()

        def dispatch(gpu):
            nonlocal sent, dispatch_ns
            started = time.monotonic_ns()
            requests[gpu].put(tasks[sent])
            sent += 1
            dispatch_ns += time.monotonic_ns() - started

        if policy == "single":
            for _ in tasks:
                dispatch(0)
        elif policy == "round_robin":
            for index in range(len(tasks)):
                dispatch(index % gpu_count)
        else:
            # Online queue: assign the next task when a GPU reports completion.
            for gpu in range(min(gpu_count, len(tasks))):
                dispatch(gpu)
        while len(completed) < len(tasks):
            try:
                msg = replies.get(timeout=120)
            except queue.Empty as exc:
                raise RuntimeError("Worker did not finish within 120 seconds") from exc
            if msg["kind"] != "done":
                raise RuntimeError(f"Unexpected worker message: {msg}")
            completed.append(msg)
            if policy == "adaptive" and sent < len(tasks):
                dispatch(msg["gpu"])
        end_ns = time.monotonic_ns()
        if not all(item["correct"] for item in completed):
            raise RuntimeError("Incorrect GEMM result; run is invalid")
        seconds = (end_ns - begin_ns) / 1e9
        return {"policy": policy, "gpu_count": gpu_count, "seconds": seconds,
                "tasks_per_second": len(tasks) / seconds,
                "dispatch_ms": dispatch_ns / 1e6,
                "gpu_event_ms_sum": sum(item["gpu_ms"] for item in completed),
                "task_host_ms_median": statistics.median(
                    (item["finish_ns"] - item["start_ns"]) / 1e6 for item in completed),
                "all_correct": True, "tasks": sorted(completed, key=lambda item: item["id"])}
    finally:
        for request in requests:
            request.put(None)
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    import torch

    if torch.cuda.device_count() < 2:
        raise RuntimeError("Two CUDA GPUs are required")
    sizes = [256, 512, 1024, 256, 768, 512, 1024, 256] if args.smoke else [256, 512, 1024, 768] * 8
    tasks = [(i, n, args.seed + i) for i, n in enumerate(sizes)]
    policies = ["single", "round_robin", "adaptive"]
    output = {"schema_version": 1, "seed": args.seed, "sizes": sizes,
              "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
              "gpus": [torch.cuda.get_device_name(i) for i in range(2)],
              "note": "GPU-resident GEMM; no H2D/D2D transfers. Pilot, not final evidence.",
              "runs": []}
    for repetition in range(1 if args.smoke else args.repetitions):
        for policy in policies:
            result = run_policy(policy, tasks, 1 if policy == "single" else 2)
            result["repetition"] = repetition
            output["runs"].append(result)
            print(f"{policy} repeat={repetition}: {result['seconds']:.4f}s, correct={result['all_correct']}", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Saved {out}", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
