#!/usr/bin/env bash
set -u
mkdir -p results
{
  date -u
  uname -a
  cat /etc/os-release
  nvidia-smi
  nvidia-smi topo -m
  command -v nvcc && nvcc --version || true
  command -v nsys && nsys --version || true
  python - <<'PY'
import sys
print('Python:', sys.version)
try:
 import torch
 print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda)
 print('Visible GPUs:', torch.cuda.device_count())
 for i in range(torch.cuda.device_count()):
  print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_properties(i))
except Exception as exc:
 print('PyTorch diagnostic error:', repr(exc))
PY
} > results/diagnostic.txt 2>&1
cat results/diagnostic.txt
