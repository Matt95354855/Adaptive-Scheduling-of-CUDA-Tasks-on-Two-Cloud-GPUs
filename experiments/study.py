#!/usr/bin/env python3
"""Paired two-GPU scheduling study; requires CUDA-enabled PyTorch."""
import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import queue
import random
import statistics
import time
from pathlib import Path

POLICIES = ("single", "round_robin", "size_fifo", "size_lpt", "calibrated", "feedback")
SIZES = {"micro": [256, 512, 768, 1024] * 8,
         "compute": [1024, 2048, 3072, 4096] * 6,
         "hidden": [3072] * 24,
         # Independent tasks whose operands are created in pinned host memory
         # and copied to the selected GPU inside the measured window.
         "transfer": [1024, 1536, 2048] * 8,
         # A sustained transfer-plus-compute workload.  Larger matrices and
         # repeated GEMMs make device utilization observable at batch level.
         "transfer_heavy": [4096, 5120, 6144] * 4,
         # Equal visible sizes with a hidden, shuffled compute multiplicity.
         # This preserves the H2D path while testing online feedback.
         "transfer_hidden": [4096] * 12}


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


def host_load_worker(start, stop, ready, active):
    """CPU-bound host-work surrogate used only in the contention control."""
    block = b"cuda-host-contention" * 65536
    ready.put(True)
    start.wait()
    active.put(True)
    while not stop.is_set():
        hashlib.sha512(block).digest()


def worker(gpu, requests, replies, sizes, resident, transfer, reuse_pinned_inputs):
    import torch
    torch.cuda.set_device(gpu)
    device = f"cuda:{gpu}"
    calibration = {}
    # Per-size and per-GPU timing, outside the measured batch. For the transfer
    # workload the proxy includes copies from already allocated pinned memory.
    # Timed tasks still include host allocation and validation.
    for n in sizes:
        a = torch.ones((n, n), device=device)
        b = torch.ones((n, n), device=device)
        torch.mm(a, b)
        if transfer:
            a_host = torch.ones((n, n), pin_memory=True)
            b_host = torch.ones((n, n), pin_memory=True)
        samples = []
        for _ in range(5):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            if transfer:
                a_device = a_host.to(device, non_blocking=True)
                b_device = b_host.to(device, non_blocking=True)
                torch.mm(a_device, b_device)
            else:
                torch.mm(a, b)
            end.record()
            end.synchronize()
            samples.append(begin.elapsed_time(end))
        calibration[n] = statistics.median(samples)
    resident_data = {n: (torch.ones((n, n), device=device),
                         torch.ones((n, n), device=device),
                         torch.empty((n, n), device=device)) for n in sizes} if resident else {}
    # This control retains the H2D copy in every task while removing per-task
    # pinned-host allocation.  Its inputs are constant, so its checksum scale
    # is deliberately fixed at one.
    pinned_data = {n: (torch.ones((n, n), pin_memory=True),
                       torch.ones((n, n), pin_memory=True)) for n in sizes} if reuse_pinned_inputs else {}
    torch.cuda.synchronize(gpu)
    replies.put({"kind": "ready", "gpu": gpu, "calibration": calibration})
    while True:
        task = requests.get()
        if task is None:
            return
        task_id, n, seed, *extra = task
        repeats = extra[0] if extra else 1
        start_ns = time.monotonic_ns()
        scale = 1.0 if (resident or reuse_pinned_inputs) else ((seed % 7) + 1) / 8.0
        h2d_ms = 0.0
        input_prepare_begin_ns = time.monotonic_ns()
        if resident:
            a, b, c = resident_data[n]
            input_prepare_ns = time.monotonic_ns() - input_prepare_begin_ns
        elif transfer and reuse_pinned_inputs:
            a_host, b_host = pinned_data[n]
            input_prepare_ns = time.monotonic_ns() - input_prepare_begin_ns
            torch.cuda.synchronize(gpu)
            transfer_begin, transfer_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            transfer_begin.record()
            a = a_host.to(device, non_blocking=True)
            b = b_host.to(device, non_blocking=True)
            transfer_end.record()
            transfer_end.synchronize()
            h2d_ms = transfer_begin.elapsed_time(transfer_end)
        elif transfer:
            a_host = torch.full((n, n), scale, pin_memory=True)
            b_host = torch.ones((n, n), pin_memory=True)
            input_prepare_ns = time.monotonic_ns() - input_prepare_begin_ns
            torch.cuda.synchronize(gpu)
            transfer_begin, transfer_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            transfer_begin.record()
            a = a_host.to(device, non_blocking=True)
            b = b_host.to(device, non_blocking=True)
            transfer_end.record()
            transfer_end.synchronize()
            h2d_ms = transfer_begin.elapsed_time(transfer_end)
        else:
            a = torch.full((n, n), scale, device=device)
            b = torch.ones((n, n), device=device)
            input_prepare_ns = time.monotonic_ns() - input_prepare_begin_ns
        torch.cuda.synchronize(gpu)
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.nvtx.range_push(f"task_{task_id}_n{n}_gpu{gpu}")
        begin.record()
        for _ in range(repeats):
            if resident:
                torch.mm(a, b, out=c)
            else:
                c = torch.mm(a, b)
        end.record()
        end.synchronize()
        torch.cuda.nvtx.range_pop()
        actual = float(c.sum().item())
        finish_ns = time.monotonic_ns()
        expected = scale * n ** 3
        replies.put({"kind": "done", "id": task_id, "n": n, "repeats": repeats, "gpu": gpu,
                     "start_ns": start_ns, "finish_ns": finish_ns,
                     "gpu_ms": begin.elapsed_time(end), "h2d_ms": h2d_ms,
                     "input_prepare_ms": input_prepare_ns / 1e6,
                     "correct": abs(actual - expected) <= max(1.0, expected * 1e-4)})


def run(policy, tasks, resident=False, transfer=False, reuse_pinned_inputs=False,
        host_load_workers=0):
    ctx = mp.get_context("spawn")
    requests = [ctx.Queue() for _ in range(2)]
    replies = ctx.Queue()
    load_stop, load_processes = None, []
    processes = [ctx.Process(target=worker, args=(g, requests[g], replies,
                                                sorted({task[1] for task in tasks}), resident, transfer,
                                                reuse_pinned_inputs))
                 for g in range(2)]
    for process in processes:
        process.start()
    try:
        calibration = {}
        for _ in processes:
            msg = replies.get(timeout=120)
            if msg["kind"] != "ready":
                raise RuntimeError(msg)
            calibration[msg["gpu"]] = msg["calibration"]
        assignment = plan(policy, [(task[0], task[1]) for task in tasks], calibration)
        # Start the control after worker calibration and before the timer. This
        # makes host pressure an experimental factor rather than setup time.
        if host_load_workers:
            load_start, load_stop = ctx.Event(), ctx.Event()
            load_ready, load_active = ctx.Queue(), ctx.Queue()
            load_processes = [ctx.Process(target=host_load_worker,
                                          args=(load_start, load_stop, load_ready, load_active))
                              for _ in range(host_load_workers)]
            for process in load_processes:
                process.start()
            for _ in load_processes:
                load_ready.get(timeout=30)
            load_start.set()
            for _ in load_processes:
                load_active.get(timeout=30)
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
                  "host_load_workers": host_load_workers,
                  "dispatch_ms": dispatch_ns / 1e6,
                  "gpu0_tasks": len(by_gpu[0]), "gpu1_tasks": len(by_gpu[1]),
                  "gpu0_n3": sum(x["n"] ** 3 for x in by_gpu[0]),
                  "gpu1_n3": sum(x["n"] ** 3 for x in by_gpu[1]),
                  "gpu0_event_ms": sum(x["gpu_ms"] for x in by_gpu[0]),
                  "gpu1_event_ms": sum(x["gpu_ms"] for x in by_gpu[1]),
                  "gpu0_h2d_ms": sum(x["h2d_ms"] for x in by_gpu[0]),
                  "gpu1_h2d_ms": sum(x["h2d_ms"] for x in by_gpu[1]),
                  "input_prepare_ms_median": statistics.median(x["input_prepare_ms"] for x in completed),
                  "host_task_ms_median": statistics.median((x["finish_ns"] - x["start_ns"]) / 1e6 for x in completed),
                  "calibration": calibration,
                  "tasks": sorted(completed, key=lambda x: x["id"])}
        return record
    finally:
        if load_stop is not None:
            load_stop.set()
        for process in load_processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
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
    fields = ("workload", "mode", "task_count", "gemm_repeats", "host_load_workers", "seed", "repetition", "order", "policy", "makespan_ms", "dispatch_ms",
              "gpu0_tasks", "gpu1_tasks", "gpu0_n3", "gpu1_n3", "gpu0_event_ms", "gpu1_event_ms",
              "gpu0_h2d_ms", "gpu1_h2d_ms", "input_prepare_ms_median", "host_task_ms_median")
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
    parser.add_argument("--rep-start", type=int, default=0, help="Repetition index offset for follow-up blocks")
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=POLICIES)
    parser.add_argument("--resident", action="store_true", help="Preallocate operands and output per size")
    parser.add_argument("--reuse-pinned-inputs", action="store_true",
                        help="For transfer: reuse pinned host inputs while retaining H2D copies")
    parser.add_argument("--transfer-tasks", type=int, default=None,
                        help="For transfer: number of tasks, divisible by three (default: 24)")
    parser.add_argument("--gemm-repeats", type=int, default=1,
                        help="Repeat each transfer GEMM this many times after its H2D copy")
    parser.add_argument("--host-load-workers", type=int, default=0,
                        help="CPU host-contention processes, started outside each timed batch")
    args = parser.parse_args()
    is_transfer = args.workload.startswith("transfer")
    if args.resident and is_transfer:
        raise ValueError("--resident cannot be combined with a transfer workload")
    if args.reuse_pinned_inputs and not is_transfer:
        raise ValueError("--reuse-pinned-inputs requires a transfer workload")
    if args.transfer_tasks is not None:
        if not is_transfer or args.transfer_tasks < 3 or args.transfer_tasks % 3:
            raise ValueError("--transfer-tasks requires a transfer workload and a positive multiple of three")
    if args.gemm_repeats < 1:
        raise ValueError("--gemm-repeats must be positive")
    if args.host_load_workers < 0:
        raise ValueError("--host-load-workers must be non-negative")
    import torch
    if torch.cuda.device_count() < 2:
        raise RuntimeError("Two CUDA GPUs required")
    mode = ("resident" if args.resident else
            ("host_to_device_reused_pinned" if args.reuse_pinned_inputs else
             ("host_to_device" if is_transfer else "allocating")))
    task_count = args.transfer_tasks or len(SIZES[args.workload])
    data = {"schema_version": 5, "workload": args.workload, "task_count": task_count,
            "gemm_repeats": args.gemm_repeats, "host_load_workers": args.host_load_workers,
            "mode": mode, "seed_start": args.seed_start,
            "seed_count": args.seeds, "rep_start": args.rep_start,
            "repetitions": args.repetitions,
            "gpus": [torch.cuda.get_device_name(g) for g in range(2)],
            "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "runs": []}
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        sizes = SIZES[args.workload].copy()
        if args.transfer_tasks is not None:
            base_sizes = [4096, 5120, 6144] if args.workload == "transfer_heavy" else [1024, 1536, 2048]
            sizes = base_sizes * (args.transfer_tasks // 3)
        random.Random(seed).shuffle(sizes)
        if args.workload == "hidden":
            repeats = [12] * 8 + [1] * 16
            random.Random(seed ^ 0x5A17).shuffle(repeats)
            tasks = [(i, n, seed + i, repeats[i]) for i, n in enumerate(sizes)]
        elif args.workload == "transfer_hidden":
            # Four tasks are eight times longer than the other eight.  The
            # scheduler receives only equal matrix sizes, never this field.
            repeats = [32] * 4 + [4] * 8
            random.Random(seed ^ 0x5A17).shuffle(repeats)
            tasks = [(i, n, seed + i, repeats[i]) for i, n in enumerate(sizes)]
        elif is_transfer and args.gemm_repeats != 1:
            tasks = [(i, n, seed + i, args.gemm_repeats) for i, n in enumerate(sizes)]
        else:
            tasks = [(i, n, seed + i) for i, n in enumerate(sizes)]
        for repetition in range(args.rep_start, args.rep_start + args.repetitions):
            order = list(args.policies)
            random.Random(seed * 1009 + repetition).shuffle(order)
            for index, policy in enumerate(order):
                record = run(policy, tasks, resident=args.resident,
                             transfer=is_transfer,
                             reuse_pinned_inputs=args.reuse_pinned_inputs,
                             host_load_workers=args.host_load_workers)
                record.update(workload=args.workload, task_count=task_count, gemm_repeats=args.gemm_repeats,
                              host_load_workers=args.host_load_workers,
                              seed=seed, repetition=repetition,
                              order=index, mode=data["mode"])
                data["runs"].append(record)
                write_outputs(data, args.out)
                print(f"{args.workload} seed={seed} rep={repetition} {policy}: "
                      f"{record['makespan_ms']:.3f} ms", flush=True)
    print(f"saved {args.out} and {args.out.with_suffix('.csv')}", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
