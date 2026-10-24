#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p results
monitor_log=results/sustained-gpu-util-2026-10-24.log
nvidia-smi dmon -s u -d 1 -c 3600 > "$monitor_log" 2>&1 &
monitor_pid=$!
cleanup() {
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
}
trap cleanup EXIT
python experiments/study.py \
  --out results/sustained-2026-10-24.json \
  --workload transfer_heavy --reuse-pinned-inputs --gemm-repeats 64 \
  --seed-start 1600 --seeds 4 --repetitions 2 \
  --policies round_robin feedback
