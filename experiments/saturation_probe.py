#!/usr/bin/env python3
"""Sustained two-GPU and host-contention probe for a CUDA instance.

This is a system-stress observation, separate from the scheduling benchmark.
Two GPU processes continuously execute large GEMMs while independent CPU
processes emulate concurrent host-side input preparation.
"""
import argparse
import hashlib
import multiprocessing as mp
import time


def cpu_preprocessor(start, seconds):
    block = b"cuda-host-preprocessing" * 65536  # about 1.5 MiB
    start.wait()
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        hashlib.sha512(block).digest()


def gpu_worker(gpu, start, ready, seconds, size, replies):
    import torch

    torch.set_num_threads(1)
    torch.cuda.set_device(gpu)
    device = f"cuda:{gpu}"
    a = torch.ones((size, size), device=device)
    b = torch.ones((size, size), device=device)
    c = torch.empty((size, size), device=device)
    torch.mm(a, b, out=c)
    torch.cuda.synchronize(gpu)
    ready.put(gpu)
    start.wait()
    deadline = time.monotonic() + seconds
    iterations = 0
    while time.monotonic() < deadline:
        torch.mm(a, b, out=c)
        iterations += 1
    torch.cuda.synchronize(gpu)
    expected = float(size**3)
    actual = float(c.sum().item())
    replies.put((gpu, iterations, abs(actual - expected) <= max(1.0, expected * 1e-4)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--size", type=int, default=6144)
    parser.add_argument("--cpu-workers", type=int, default=20)
    args = parser.parse_args()
    if min(args.seconds, args.size, args.cpu_workers) < 1:
        raise ValueError("all parameters must be positive")

    import torch

    if torch.cuda.device_count() < 2:
        raise RuntimeError("Two CUDA GPUs required")
    ctx = mp.get_context("spawn")
    start, ready, replies = ctx.Event(), ctx.Queue(), ctx.Queue()
    gpus = [ctx.Process(target=gpu_worker, args=(gpu, start, ready, args.seconds, args.size, replies))
            for gpu in range(2)]
    for process in gpus:
        process.start()
    for _ in gpus:
        ready.get(timeout=180)
    cpus = [ctx.Process(target=cpu_preprocessor, args=(start, args.seconds))
            for _ in range(args.cpu_workers)]
    for process in cpus:
        process.start()
    start_time = time.monotonic()
    start.set()
    for process in gpus + cpus:
        process.join(timeout=args.seconds + 30)
    if any(process.is_alive() for process in gpus + cpus):
        for process in gpus + cpus:
            if process.is_alive():
                process.terminate()
        raise RuntimeError("probe process did not stop")
    results = [replies.get(timeout=10) for _ in gpus]
    if not all(correct for _, _, correct in results):
        raise RuntimeError("incorrect GEMM checksum")
    print(f"elapsed_s={time.monotonic() - start_time:.3f}")
    for gpu, iterations, _ in sorted(results):
        print(f"gpu={gpu},gemm_iterations={iterations},size={args.size}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
