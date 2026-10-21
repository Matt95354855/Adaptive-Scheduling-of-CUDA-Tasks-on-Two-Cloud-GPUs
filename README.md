# Adaptive Scheduling of CUDA Tasks on Two Cloud GPUs

This repository studies how independent CUDA tasks are assigned to two cloud GPUs. The current implementation is a **research pilot**, not a demonstrated general-purpose scheduler. It compares one GPU, fixed round-robin assignment, and a completion-feedback policy that sends the next task to the first available GPU.

## Host-to-device transfer study completed on 21 October 2026

The transfer study tests H2D-plus-GEMM tasks on two RTX PRO 4000 Blackwell GPUs. Its key result is conditional on input lifetime and task duration: with fresh pinned-host allocation in every short task, `single` was 23.07% faster than round-robin across all eight task orders; retaining the H2D copies but reusing pinned inputs reversed the result, with round-robin 35.86% faster. On a sustained workload of 4096–6144 matrices and 16 GEMMs per task, round-robin was 64.86% faster and monitoring recorded 100% utilization medians on both GPUs. When duration was deliberately hidden from the scheduler, feedback showed a descriptive 8.11% median gain over round-robin, with an interval still consistent with no gain. Nsight Systems confirms that H2D dominates captured GPU memory-operation time, while the controls show that the input lifecycle and compute duration determine which policy wins.

- [Transfer results, Nsight analysis, conclusions, and limits](docs/transfer-results-2026-10-21.md)
- [Allocated-input paired measurements](results/transfer-paired-2026-10-21.csv)
- [Reused-pinned-input control](results/transfer-reused-pinned-paired-2026-10-21.csv)
- [Sustained heavy-load measurements](results/transfer-heavy-paired-2026-10-21.csv)
- [Hidden-duration adaptive measurements](results/transfer-hidden-paired-2026-10-21.csv)
- [Host and GPU saturation probe](results/saturation-probe-summary-2026-10-21.csv)
- [Transfer benchmark code](experiments/study.py)

The pod uses ephemeral disk by request. The repository therefore retains paired results and trace aggregates, but not raw task JSON or the binary Nsight trace.

## Next prepared campaign

The next campaign will replicate the hidden-duration result and vary controlled host CPU contention before selecting the condition for an independent replication.

- [Predeclared next-campaign protocol and ready-to-run commands](docs/next-campaign-protocol-2026-10-21.md)

## Follow-up study completed on 20 October 2026

The follow-up uses twelve independent task orders per workload, randomized policy order, paired seed-level analysis, static size-aware baselines, a resident-input control, a hidden-duration stress test, and a focused Nsight Systems capture. The central result is conditional: static size-aware scheduling wins when cost is visible; completion feedback wins when costs are hidden and round-robin is imbalanced; feedback loses on short tasks and already balanced orders.

- [Complete results, post-analysis, Nsight summary, and limits](docs/followup-results-2026-10-20.md)
- [Predeclared follow-up protocol](docs/followup-protocol-2026-10-20.md)
- [Paired compute measurements](results/compute-paired-2026-10-20.csv)
- [Paired hidden-duration measurements](results/hidden-paired-2026-10-20.csv)
- [Follow-up benchmark and analysis scripts](experiments/study.py)

The RunPod pod used for the 20 October study was stopped after that study. Raw task JSON and the binary Nsight trace were on a container without persistent storage; published paired measurements and the trace summary are retained in this repository.

## Initial pilot completed on 20 October 2026

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
