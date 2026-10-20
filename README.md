# Adaptive-Scheduling-of-CUDA-Tasks-on-Two-Cloud-GPUs
This project evaluates scheduling strategies for independent CUDA tasks across two cloud GPUs. It compares single-GPU, static, dynamic, and adaptive scheduling, measuring runtime, throughput, transfer overhead, variability, and estimated cost.




# Two-GPU CUDA Task Scheduling in the Cloud

This research project investigates how independent CUDA tasks can be scheduled across two cloud-based NVIDIA GPUs. It compares scheduling strategies to measure their effects on execution time, throughput, data-transfer overhead, performance variability, and estimated cost.

## Research goal

Evaluate whether using two rented GPUs improves the performance and cost-efficiency of parallel CUDA workloads compared with running them on a single GPU.

## Scheduling strategies

- **Single-GPU execution** as a baseline
- **Static scheduling**, assigning tasks to GPUs in advance
- **Dynamic scheduling**, assigning tasks as GPUs become available
- **Adaptive scheduling**, adjusting assignments based on observed performance

## Evaluation metrics

- Total execution time
- Throughput
- GPU utilization
- Data-transfer and scheduling overhead
- Performance variability across repeated runs
- Estimated cloud GPU cost

## Experimental setup

Experiments will use cloud-rented NVIDIA GPUs and workloads composed of independent CUDA tasks. GPU models, workload configurations, software versions, and measurement procedures will be documented to support reproducibility.

## Repository structure

```text
├── src/             # CUDA workloads and scheduling implementations
├── experiments/     # Experiment configurations and scripts
├── results/         # Measurements and analysis
├── docs/            # Research notes and paper materials
└── README.md
