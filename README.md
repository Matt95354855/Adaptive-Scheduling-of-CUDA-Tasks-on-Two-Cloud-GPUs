# Adaptive Scheduling of CUDA Tasks on Two Cloud GPUs

This repository studies how independent CUDA tasks are assigned to two cloud GPUs. The current implementation is a **research pilot**, not a demonstrated general-purpose scheduler. It compares one GPU, fixed round-robin assignment, and a completion-feedback policy that sends the next task to the first available GPU.

## Pilot completed on 20 October 2026

Two NVIDIA RTX PRO 4000 Blackwell GPUs were tested on RunPod with a spending cap of $2. The short pilot found that the feedback policy helped on task orders that left round-robin imbalanced, while round-robin was better on a deliberately balanced order and on very small tasks. All checked output sums passed. These results are limited to one instance and three repetitions per cell.

- [Full observations and limitations](docs/observations-2026-10-20.md)
- [Individual run summary (CSV)](results/pilot-summary-2026-10-20.csv)
- [Reproduction protocol](docs/protocol.md)
- [Benchmark code](experiments/pilot.py) and [system diagnostic](experiments/diagnose.sh)

Nsight Systems 2025.1.1 captured CUDA and NVTX activity during the smoke test. The binary trace and per-task JSON files were stored on a pod without persistent storage and are not included here. The CSV preserves the run-level observations. No host-to-device or inter-GPU transfer was tested; transfer-aware scheduling remains future work.

## Quick start

Requires Python with CUDA-enabled PyTorch and two visible NVIDIA GPUs.

```bash
bash experiments/diagnose.sh
python experiments/pilot.py --smoke --out results/smoke.json
python experiments/pilot.py --repetitions 3 --out results/pilot.json
python experiments/pilot.py --workload compute --shuffle --seed 43 --repetitions 3 --out results/compute-shuffled.json
```

See the protocol for the balanced control and Nsight command. This code is designed for short, reproducible pilot runs. It is not yet a transfer-cost model, DAG scheduler, or production runtime.

## License

MIT; see [LICENSE](LICENSE).
