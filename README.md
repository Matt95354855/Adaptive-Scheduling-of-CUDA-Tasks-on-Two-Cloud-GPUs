# Adaptive Scheduling of CUDA Tasks on Two Cloud GPUs

This repository studies how independent CUDA tasks are assigned to two cloud GPUs. The current implementation is a **research pilot**, not a demonstrated general-purpose scheduler. It compares one GPU, fixed round-robin assignment, and a completion-feedback policy that sends the next task to the first available GPU.

## Independent replication and online decision, 24 October 2026

On another two-RTX PRO 4000 Blackwell pod, a prospective 24-seed replication confirmed that feedback's gain depends strongly on the final long-task allocation: median **+18.18%** on 17 imbalanced orders and **−7.88%** on seven balanced orders; the predeclared gap was **+26.05 percentage points** [17.24, 30.11]. An alternating H0/H40 experiment found a median change of **−13.08 points**, but its confidence interval [−21.81, +5.19] does not establish the sign of the CPU-contention effect.

A predeclared four-task online probe then tested whether that retrospective imbalance could guide a real decision. Across **240 runs and 2,880 tasks**, `probe_switch` did **not** beat the same-prefix `probe_feedback` control: median **−0.12%**, 95% bootstrap interval [−0.59, +0.23]. It was **14.62% slower** than feedback without the probe. Post-hoc diagnostics show that `probe_feedback` was faster than `probe_rr` on every seed, so none of the six choices of round-robin helped. These results narrow the scheduling claim: the conditional retrospective effect is reproducible, while this online rule fails its prospective test.

Continuous GPU sampling also explains why an active pod can show 0% between runs. The short online-probe campaign averaged **13.99% / 14.29% SM occupancy** across GPU 0 / GPU 1 despite bursts to 100%. A separate dense control reached **96.27% / 97.41% median CUDA-event time within each measured batch**, but only **48.37% / 49.47% mean SM occupancy across the whole campaign** because workers and calibration restart between batches. The 70% whole-campaign target was not met. Full raw JSON, task CSVs, telemetry, scripts, figures, and conclusions are published for this pod; a new Nsight trace and the predeclared H0/H40 × input-lifetime factorial remain pending.

- [Replication protocol](docs/replication-protocol-2026-10-24.md) and [results](docs/replication-results-2026-10-24.md)
- [Alternating contention results and paired chart](docs/alternating-contention-results-2026-10-24.md)
- [Online probe protocol](docs/online-probe-protocol-2026-10-24.md), [results and chart](docs/online-probe-results-2026-10-24.md)
- [Sustained-utilization protocol](docs/sustained-utilization-protocol-2026-10-24.md) and [results](docs/sustained-utilization-results-2026-10-24.md)
- [Two new visual analyses: whole-campaign GPU occupancy and probe decisions](docs/new-visual-observations-2026-10-24.md)
- [Next factorial protocol](docs/factorial-input-protocol-2026-10-24.md) and [current research checklist](docs/research-status-2026-10-24.md)

## Host contention and independent replication, 23 October 2026

On a new two-RTX PRO 4000 Blackwell pod, a predeclared paired study tested hidden-duration H2D-plus-GEMM tasks with 0 and 40 competing CPU processes. In the first 12 seeds, `feedback` had a descriptive median 16.18% gain over round-robin without CPU load, but its 95% bootstrap interval included zero. Under CPU contention, its median gain was −0.56%; the paired change in gain was −11.09 percentage points [−16.64, −1.80]. Independent H0 replication on 12 new seeds found no overall gain (−1.16% gain, 4/12 wins). Exploratory task-level analysis explains the reversal: feedback won all 11 task orders where round-robin assigned the four long tasks unevenly, but only 2 of 13 already balanced orders. This conditional split was discovered after measurement and then tested prospectively.

That prospective validation is now complete on 24 further seeds (1000–1023): the gain difference between imbalanced and balanced orders was **20.94 percentage points** [13.02, 27.66], meeting its predeclared criterion. Feedback won 14/14 imbalanced orders, while the balanced group's median gain was −1.60% with an interval crossing zero. This validates a conditional result for this synthetic workload on this instance; the true long-task imbalance is known only after execution and is not yet an online scheduling signal.

- [Protocol and complete analysis](docs/contention-results-2026-10-23.md)
- [Illustrated conclusions and three publication-ready charts](docs/visual-analysis-2026-10-23.md) ([plotting code](experiments/plot_research.py))
- [Paired H0 and H40 interaction](results/transfer-hidden-contention-2026-10-23.json)
- [Independent replication runs](results/transfer-hidden-replication-h0-2026-10-23.csv)
- [Seed-level balance analysis](results/transfer-hidden-seed-strata-2026-10-23.csv)
- [Prospective validation protocol](docs/strata-validation-protocol-2026-10-23.md) and [verdict](results/strata-validation-summary-2026-10-23.json)
- [Research phase checklist and next experiments](docs/research-status-2026-10-23.md)
- [Analysis code](experiments/contention_analysis.py) and [task export](experiments/task_export.py)

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

## Predeclared campaign protocol

The host contention and independent replication campaign above followed this protocol.

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
