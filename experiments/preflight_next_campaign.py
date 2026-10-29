#!/usr/bin/env python3
"""Fail before paid measurements if the two-GPU research pod is unsuitable."""

import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_GPU = "RTX PRO 4000 Blackwell"
NSYS = Path("/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys")


def command(*args):
    result = subprocess.run(args, text=True, capture_output=True, check=True)
    return result.stdout.strip()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import torch

    count = torch.cuda.device_count()
    if count != 2:
        raise RuntimeError(f"Exactly two GPUs required; found {count}")
    names = [torch.cuda.get_device_name(index) for index in range(count)]
    if any(EXPECTED_GPU not in name for name in names):
        raise RuntimeError(f"Expected two {EXPECTED_GPU} GPUs; found {names}")
    memories = [torch.cuda.get_device_properties(index).total_memory for index in range(count)]
    if min(memories) < 23_000_000_000:
        raise RuntimeError(f"Insufficient GPU memory: {memories}")
    if not NSYS.is_file() or not os.access(NSYS, os.X_OK):
        raise RuntimeError(f"Nsight Systems unavailable at expected path: {NSYS}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.out.parent).free < 2_000_000_000:
        raise RuntimeError("Less than 2 GB free for raw results and Nsight traces")
    manifest = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_names": names,
        "gpu_memory_bytes": memories,
        "nvidia_smi_query": command("nvidia-smi", "--query-gpu=index,name,memory.total,driver_version,pci.bus_id", "--format=csv,noheader"),
        "nvidia_smi_topology": command("nvidia-smi", "topo", "-m"),
        "nsight_version": command(str(NSYS), "--version"),
        "cpu_max": Path("/sys/fs/cgroup/cpu.max").read_text().strip()
                   if Path("/sys/fs/cgroup/cpu.max").exists() else None,
        "cpu_stat_before": Path("/sys/fs/cgroup/cpu.stat").read_text()
                           if Path("/sys/fs/cgroup/cpu.stat").exists() else None,
        "free_disk_bytes": shutil.disk_usage(args.out.parent).free,
    }
    args.out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "ready", "gpu_names": names,
                      "free_disk_bytes": manifest["free_disk_bytes"],
                      "nsight_version": manifest["nsight_version"]}, indent=2))


if __name__ == "__main__":
    main()
