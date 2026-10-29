#!/usr/bin/env bash
# Secondary profiles only; seed 1512 is excluded from the 192 statistical runs.
set -euo pipefail

cd "$(dirname "$0")/.."
nsys=/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys
[[ -x "$nsys" ]] || { echo "Nsight Systems not found: $nsys" >&2; exit 1; }
[[ -f results/factorial-v2-summary.json ]] || {
  echo "Finish and analyze the factorial grid before profiling" >&2
  exit 1
}

for mode in fresh reused; do
  for workers in 0 40; do
    prefix="results/nsys-factorial-v2-${mode}-h${workers}"
    args=(--workload transfer_hidden --seed-start 1512 --seeds 1
          --repetitions 1 --policies feedback --fixed-input-scale 1
          --host-load-workers "$workers")
    if [[ "$mode" == reused ]]; then args+=(--reuse-pinned-inputs); fi
    "$nsys" profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
      --force-overwrite=true -o "$prefix" \
      python experiments/study.py --out "${prefix}.json" "${args[@]}" \
      > "${prefix}.profile.log" 2>&1
    "$nsys" stats --report cuda_gpu_kern_sum --report cuda_gpu_mem_time_sum \
      --report nvtx_sum --format csv --output . "${prefix}.nsys-rep" \
      > "${prefix}.stats.log" 2>&1
    echo "saved Nsight profile and reports: $prefix"
  done
done
python experiments/audit_factorial_exports.py \
  --results-dir results --require-nsight --out results/factorial-v2-manifest.json
