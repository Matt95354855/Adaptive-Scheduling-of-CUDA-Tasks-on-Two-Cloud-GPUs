#!/usr/bin/env bash
# Run only after the user rents a two-RTX PRO 4000 Blackwell pod.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p results
resume=()
if [[ "${1:-}" == "--resume" && $# == 1 ]]; then
  resume=(--resume)
elif [[ $# != 0 ]]; then
  echo "usage: bash experiments/run_factorial_campaign.sh [--resume]" >&2
  exit 2
fi

if [[ ${#resume[@]} == 0 ]]; then
  python experiments/preflight_next_campaign.py --out results/factorial-v2-preflight.json
  git rev-parse HEAD > results/factorial-v2-code-revision.txt
  python experiments/study.py --out results/factorial-v2-smoke-fresh.json \
    --workload transfer_hidden --seed-start 1499 --seeds 1 --repetitions 1 \
    --policies round_robin feedback --fixed-input-scale 1
  python experiments/study.py --out results/factorial-v2-smoke-reused.json \
    --workload transfer_hidden --seed-start 1499 --seeds 1 --repetitions 1 \
    --policies round_robin feedback --fixed-input-scale 1 --reuse-pinned-inputs
  date -u '+%Y-%m-%dT%H:%M:%SZ' > results/factorial-v2-start.txt
  if [[ -f /sys/fs/cgroup/cpu.stat ]]; then
    cp /sys/fs/cgroup/cpu.stat results/factorial-v2-cpu-stat-before.txt
  fi
else
  python experiments/preflight_next_campaign.py --out results/factorial-v2-preflight-resume.json
  current_revision=$(git rev-parse HEAD)
  recorded_revision=$(cat results/factorial-v2-code-revision.txt)
  [[ "$current_revision" == "$recorded_revision" ]] || {
    echo "Refusing to resume with a different code revision" >&2
    exit 1
  }
  date -u '+resume %Y-%m-%dT%H:%M:%SZ' >> results/factorial-v2-sessions.log
fi
nvidia-smi dmon -s u -d 1 -c 7200 >> results/factorial-v2-gpu-util.log 2>&1 &
monitor_pid=$!
cleanup() {
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
}
trap cleanup EXIT

python experiments/factorial_host_input.py \
  --out results/factorial-v2.json \
  --seed-start 1500 --seeds 12 --repetitions 2 "${resume[@]}"

cleanup
trap - EXIT
date -u '+%Y-%m-%dT%H:%M:%SZ' > results/factorial-v2-end.txt
date -u '+complete %Y-%m-%dT%H:%M:%SZ' >> results/factorial-v2-sessions.log
if [[ -f /sys/fs/cgroup/cpu.stat ]]; then
  cp /sys/fs/cgroup/cpu.stat results/factorial-v2-cpu-stat-after.txt
fi
python experiments/task_export.py \
  results/factorial-v2.json results/factorial-v2-tasks.csv
python experiments/analyze_factorial.py \
  results/factorial-v2.json --seed-start 1500 --seeds 12 --repetitions 2 \
  --out-prefix results/factorial-v2
python experiments/analyze_utilization.py \
  results/factorial-v2-gpu-util.log results/factorial-v2.json \
  --out results/factorial-v2-utilization.json
python experiments/audit_factorial_exports.py \
  --results-dir results --out results/factorial-v2-manifest.json
echo "Factorial campaign complete: 192 runs, 2304 tasks. Export every result before stopping the pod."
