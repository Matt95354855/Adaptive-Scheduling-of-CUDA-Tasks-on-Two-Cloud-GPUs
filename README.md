# Adaptive Scheduling of CUDA Tasks on Two Cloud GPUs

**When does a smarter scheduler actually beat round-robin on multi-GPU?** A controlled benchmark of six task-assignment policies on two NVIDIA RTX PRO 4000 Blackwell GPUs (RunPod), with paired, randomized runs and published raw data.

[![CI](https://github.com/Matt95354855/Adaptive-Scheduling-of-CUDA-Tasks-on-Two-Cloud-GPUs/actions/workflows/ci.yml/badge.svg)](https://github.com/Matt95354855/Adaptive-Scheduling-of-CUDA-Tasks-on-Two-Cloud-GPUs/actions/workflows/ci.yml) ![Python](https://img.shields.io/badge/Python-3.12-3776AB) ![PyTorch](https://img.shields.io/badge/PyTorch-2.8%20%2B%20CUDA%2012.8-EE4C2C) ![GPU](https://img.shields.io/badge/GPU-2×%20RTX%20PRO%204000%20Blackwell-76B900) ![License](https://img.shields.io/badge/license-MIT-green)

[English](README.md) · [Français](README.fr.md)

![Median batch makespan per policy](docs/images/results.png)

## Key findings

Paired analysis over 12 independent task orders per workload (8 for transfers) (`experiments/analyze.py`), change vs round-robin with 95 % bootstrap CI:

| Workload | Best policy | Speed-up vs 1 GPU | vs round-robin | Orders won |
|---|---|---:|---:|---:|
| 24 compute-bound GEMMs, mixed sizes | **Size-aware LPT** (static, knows `n³`) | **1.53×** | **−7.2 %** [−10.0, −4.0] | 12/12 |
| 32 small GEMMs (≤ 1024) | Round-robin (static) | 1.13× | completion feedback **+12.5 %** slower [+9.6, +15.7] | — |
| Same-size tasks with hidden cost (12× repeats) | **Completion feedback** | **1.84×** | −10.9 % [−16.1, +2.5] | 8/12 |
| 24 H2D + GEMM tasks, pinned inputs allocated per task | **1 GPU** | — | −23.1 % [−32.8, −13.9] | 8/8 |
| Same copies, pinned inputs reused | **Round-robin** | 1.31× | 1 GPU is +35.9 % slower [+28.0, +39.2] | 8/8 |

- **Task count is a poor balance metric.** Round-robin gave each GPU 12 tasks but a 1.39× imbalance in `n³` work; the calibrated cost-model policy brought it to 1.06×.
- **Reactive feedback only pays off when durations are unpredictable** — and even then the gain is not yet statistically conclusive (CI crosses zero). When metadata predicts cost, a static size-aware partition wins with zero orchestration overhead.
- **With host-to-device transfers, input lifetime decides the winner.** Allocating pinned host memory in every task made two GPUs *slower* than one (−23 % for a single GPU); reusing the pinned buffers with the same copies flipped the result (one GPU became 36 % slower than round-robin). GEMM and H2D times were nearly identical across policies: the host allocation path is the bottleneck.
- **On small tasks the host dominates**: CUDA kernels account for ~3–4 ms of a 25–30 ms batch; allocations and inter-process messaging decide the result.
- Cloud runs under a hard **$2 spending cap** per session (pilot: $0.25); every output checksum validated; the CI re-runs the paired analysis on all published data.

## Method

```mermaid
flowchart LR
    A[Task batch<br/>GEMM sizes, seeds] --> P{Policy}
    P -->|single| G0
    P -->|round_robin / size_fifo / size_lpt / calibrated| Q[Static queues]
    P -->|feedback| F[One task in flight<br/>per GPU]
    Q --> G0[Worker GPU 0]
    Q --> G1[Worker GPU 1]
    F --> G0
    F --> G1
    G0 & G1 --> M[CUDA events + NVTX<br/>checksum validation]
    M --> R[Paired CSV results]
```

- Six policies: `single`, `round_robin`, `size_fifo`, `size_lpt`, `calibrated` (per-GPU cost model from CUDA events), `feedback`.
- Each `(seed, repetition)` block runs all policies in a **randomized order** to cancel drift; 12 seeds × 2–3 repetitions per workload.
- Fresh `spawn` processes, warm-up and calibration outside the timed window; Nsight Systems trace of the smoke test.
- Pre-registered protocol: the hidden-cost workload was documented **before** it was measured.

## Quick start

Requires CUDA-enabled PyTorch and two visible NVIDIA GPUs.

```bash
bash experiments/diagnose.sh                       # GPU topology, driver, CUDA check
python experiments/pilot.py --smoke --out results/smoke.json
python experiments/study.py --workload compute --out results/compute.json        # paired study, randomized policy order
python experiments/analyze.py results/compute-paired-2026-10-20.csv              # paired analysis of published data
```

## Repository structure

```text
experiments/   benchmark runner (pilot.py, study.py), metrics and analysis scripts, diagnostics
results/       paired per-run CSVs and balance / diagnostic summaries (2026-10-20)
docs/          protocols, observations and limitations, figures
```

## Limitations

Single instance, identical GPUs over PCIe (no NVLink), independent tasks, 8–12 task orders per workload. Raw per-task JSON and binary Nsight traces were on ephemeral storage; paired results and trace aggregates are retained. These results describe this implementation, not schedulers in general. Next: inter-GPU data locality, DAG workloads, larger samples.

## Documentation

- [Host-to-device transfer results and Nsight analysis](docs/transfer-results-2026-10-21.md) (FR)
- [Follow-up results, post-analysis and Nsight summary](docs/followup-results-2026-10-20.md) (FR)
- [Follow-up protocol](docs/followup-protocol-2026-10-20.md) (FR) · [Pilot protocol](docs/protocol.md) (FR)
- [Pilot observations and limitations](docs/observations-2026-10-20.md) (FR)
- Raw data: [`results/`](results/)

## License

MIT — see [LICENSE](LICENSE).
