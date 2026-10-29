#!/usr/bin/env python3
"""Compare cold workers with a four-batch persistent two-GPU worker pool."""

import argparse
import json
import multiprocessing as mp
import os
import queue
import random
import statistics
import subprocess
import time
from pathlib import Path

from study import SIZES, run as cold_run, worker


POLICIES = ("round_robin", "feedback")
GEMM_REPEATS = 64
FIXED_INPUT_SCALE = 1.0


def planned_blocks(seed_start, seeds):
    return [(seed, arm) for seed in range(seed_start, seed_start + seeds)
            for arm in (("cold", "warm") if (seed - seed_start) % 2 == 0
                        else ("warm", "cold"))]


def task_plan(seed):
    sizes = SIZES["transfer_heavy"].copy()
    random.Random(seed).shuffle(sizes)
    return [(index, n, seed + index, GEMM_REPEATS)
            for index, n in enumerate(sizes)]


def policy_order(seed, repetitions):
    order = []
    for rep in range(repetitions):
        policies = list(POLICIES)
        random.Random(seed * 1009 + rep).shuffle(policies)
        order.extend((rep, policy) for policy in policies)
    return order


class PersistentPool:
    def __init__(self):
        ctx = mp.get_context("spawn")
        self.requests = [ctx.Queue() for _ in range(2)]
        self.replies = ctx.Queue()
        self.processes = [ctx.Process(target=worker,
                                      args=(gpu, self.requests[gpu], self.replies,
                                            sorted(set(SIZES["transfer_heavy"])),
                                            False, True, True, FIXED_INPUT_SCALE))
                          for gpu in range(2)]
        try:
            for process in self.processes:
                process.start()
            ready = [self.replies.get(timeout=120) for _ in range(2)]
            if {message.get("gpu") for message in ready} != {0, 1} or any(
                    message.get("kind") != "ready" for message in ready):
                raise RuntimeError(f"Unexpected warm-worker calibration replies: {ready}")
        except BaseException:
            self.close()
            raise

    def run_batch(self, policy, tasks):
        if policy not in POLICIES:
            raise ValueError(policy)
        sent, completed, dispatch_ns, dispatch_times = 0, [], 0, {}

        def dispatch(gpu, task):
            nonlocal sent, dispatch_ns
            now = time.monotonic_ns()
            self.requests[gpu].put(task)
            dispatch_ns += time.monotonic_ns() - now
            dispatch_times[task[0]] = now
            sent += 1

        begin_ns = time.monotonic_ns()
        if policy == "round_robin":
            for task in tasks:
                dispatch(task[0] % 2, task)
        else:
            for gpu, task in enumerate(tasks[:2]):
                dispatch(gpu, task)
        while len(completed) < len(tasks):
            try:
                message = self.replies.get(timeout=120)
            except queue.Empty as exc:
                raise RuntimeError("Warm worker timeout") from exc
            if message.get("kind") != "done":
                raise RuntimeError(message)
            message["dispatch_ns"] = dispatch_times[message["id"]]
            message["reply_ns"] = time.monotonic_ns()
            completed.append(message)
            if policy == "feedback" and sent < len(tasks):
                dispatch(message["gpu"], tasks[sent])
        end_ns = time.monotonic_ns()
        if not all(task["correct"] for task in completed):
            raise RuntimeError("Incorrect GEMM checksum in warm pool")
        by_gpu = [[task for task in completed if task["gpu"] == gpu] for gpu in (0, 1)]
        return {"policy": policy, "makespan_ms": (end_ns - begin_ns) / 1e6,
                "dispatch_ms": dispatch_ns / 1e6,
                "gpu0_tasks": len(by_gpu[0]), "gpu1_tasks": len(by_gpu[1]),
                "gpu0_event_ms": sum(task["gpu_ms"] for task in by_gpu[0]),
                "gpu1_event_ms": sum(task["gpu_ms"] for task in by_gpu[1]),
                "gpu0_h2d_ms": sum(task["h2d_ms"] for task in by_gpu[0]),
                "gpu1_h2d_ms": sum(task["h2d_ms"] for task in by_gpu[1]),
                "input_prepare_ms_median": statistics.median(
                    task["input_prepare_ms"] for task in completed),
                "tasks": sorted(completed, key=lambda task: task["id"])}

    def close(self):
        for request in self.requests:
            request.put(None)
        for process in self.processes:
            if process.pid is None:
                continue
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)


def validate_checkpoint(data, blocks, repetitions):
    if (data.get("protocol") != "persistent_pool_v1" or
            data.get("gemm_repeats") != GEMM_REPEATS or
            data.get("fixed_input_scale") != FIXED_INPUT_SCALE or
            len(data["blocks"]) > len(blocks)):
        raise ValueError("Checkpoint metadata does not match the plan")
    for index, block in enumerate(data["blocks"]):
        if (block["seed"], block["arm"]) != blocks[index]:
            raise ValueError(f"Unexpected block at position {index}")
        expected = set(policy_order(block["seed"], repetitions))
        runs = block["runs"]
        actual = {(run["repetition"], run["policy"]) for run in runs}
        if len(runs) != len(expected) or actual != expected:
            raise ValueError(f"Missing or duplicate batch in block {index}")
        if any(len(run["tasks"]) != 12 or not all(task["correct"] for task in run["tasks"])
               for run in runs):
            raise ValueError(f"Incorrect tasks in block {index}")
    return len(data["blocks"])


def save_checkpoint(data, path):
    pending = path.with_name(path.stem + ".pending" + path.suffix)
    pending.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    os.replace(pending, path)


def run_block(seed, arm, repetitions, output_dir, monitor):
    tasks = task_plan(seed)
    monitor_log = output_dir / f"pool-v1-seed{seed}-{arm}-gpu-util.log"
    sampler = None
    if monitor:
        log_stream = monitor_log.open("w")
        sampler = subprocess.Popen(["nvidia-smi", "dmon", "-s", "u", "-d", "1",
                                    "-c", "600"], stdout=log_stream,
                                   stderr=subprocess.STDOUT)
    start_ns = time.monotonic_ns()
    runs = []
    try:
        pool = PersistentPool() if arm == "warm" else None
        try:
            for order, (rep, policy) in enumerate(policy_order(seed, repetitions)):
                record = (pool.run_batch(policy, tasks) if pool is not None else
                          cold_run(policy, tasks, transfer=True,
                                   reuse_pinned_inputs=True,
                                   fixed_input_scale=FIXED_INPUT_SCALE))
                record.update(seed=seed, repetition=rep, order=order,
                              arm=arm, workload="transfer_heavy",
                              task_count=12, gemm_repeats=GEMM_REPEATS,
                              fixed_input_scale=FIXED_INPUT_SCALE)
                runs.append(record)
        finally:
            if pool is not None:
                pool.close()
        wall_ms = (time.monotonic_ns() - start_ns) / 1e6
    finally:
        if sampler is not None:
            sampler.terminate()
            try:
                sampler.wait(timeout=5)
            except subprocess.TimeoutExpired:
                sampler.kill()
                sampler.wait()
            log_stream.close()
    return {"seed": seed, "arm": arm, "wall_ms": wall_ms,
            "batch_makespan_sum_ms": sum(run["makespan_ms"] for run in runs),
            "gpu_util_log": monitor_log.name if monitor else None,
            "runs": runs}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=1700)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-gpu-monitor", action="store_true")
    args = parser.parse_args()
    if args.seeds < 1 or args.repetitions < 1 or args.out.suffix != ".json":
        raise ValueError("Positive seeds/repetitions and a .json output are required")
    blocks = planned_blocks(args.seed_start, args.seeds)
    if args.plan_only:
        print(json.dumps({"blocks": blocks, "block_count": len(blocks),
                          "batch_count": len(blocks) * args.repetitions * len(POLICIES),
                          "task_count": len(blocks) * args.repetitions * len(POLICIES) * 12},
                         indent=2))
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.resume:
        if not args.out.exists():
            raise FileNotFoundError(args.out)
        data = json.loads(args.out.read_text())
        if (data.get("seed_start") != args.seed_start or
                data.get("seed_count") != args.seeds or
                data.get("repetitions") != args.repetitions):
            raise ValueError("Checkpoint grid does not match arguments")
        completed = validate_checkpoint(data, blocks, args.repetitions)
    else:
        if args.out.exists():
            raise FileExistsError(args.out)
        data = {"schema_version": 1, "protocol": "persistent_pool_v1",
                "seed_start": args.seed_start, "seed_count": args.seeds,
                "repetitions": args.repetitions, "gemm_repeats": GEMM_REPEATS,
                "fixed_input_scale": FIXED_INPUT_SCALE, "blocks": []}
        completed = 0
    for seed, arm in blocks[completed:]:
        block = run_block(seed, arm, args.repetitions, args.out.parent,
                          monitor=not args.no_gpu_monitor)
        data["blocks"].append(block)
        validate_checkpoint(data, blocks, args.repetitions)
        save_checkpoint(data, args.out)
        print(f"checkpoint seed={seed} arm={arm} wall={block['wall_ms']:.1f} ms; "
              f"{len(data['blocks'])}/{len(blocks)} blocks", flush=True)
    print(f"saved {args.out}: {len(blocks)} blocks", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
