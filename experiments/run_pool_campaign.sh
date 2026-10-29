#!/usr/bin/env bash
# Secondary campaign; run only after the factorial and Nsight exports are safe.
set -euo pipefail

cd "$(dirname "$0")/.."
resume=()
if [[ "${1:-}" == "--resume" && $# == 1 ]]; then
  resume=(--resume)
elif [[ $# != 0 ]]; then
  echo "usage: bash experiments/run_pool_campaign.sh [--resume]" >&2
  exit 2
fi
[[ -f results/factorial-v2-summary.json ]] || {
  echo "Factorial result must be complete before the secondary pool control" >&2
  exit 1
}
current_revision=$(git rev-parse HEAD)
recorded_revision=$(cat results/factorial-v2-code-revision.txt)
git diff --quiet "$recorded_revision" "$current_revision" -- experiments || {
  echo "Pool test requires unchanged experiment code since the factorial" >&2
  exit 1
}
git diff --quiet -- experiments || {
  echo "Pool test requires a clean experiment-code checkout" >&2
  exit 1
}
if [[ ${#resume[@]} == 0 ]]; then
  python experiments/persistent_pool_control.py --out results/pool-v1-smoke.json \
    --seed-start 1699 --seeds 1 --repetitions 1 --no-gpu-monitor
  date -u '+%Y-%m-%dT%H:%M:%SZ' > results/pool-v1-start.txt
fi
python experiments/persistent_pool_control.py --out results/pool-v1.json \
  --seed-start 1700 --seeds 8 --repetitions 2 "${resume[@]}"
date -u '+%Y-%m-%dT%H:%M:%SZ' > results/pool-v1-end.txt
python experiments/analyze_pool_control.py results/pool-v1.json \
  --out-prefix results/pool-v1
python experiments/plot_pool_control.py --prefix results/pool-v1
echo "Worker-pool control complete: 16 blocks, 64 runs, 768 tasks. Export all files."
