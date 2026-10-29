"""CPU-only validation of the planned persistent-worker control."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))
from persistent_pool_control import planned_blocks, task_plan, validate_checkpoint


class PoolPreparationTests(unittest.TestCase):
    def test_order_alternates_and_tasks_are_identical_across_arms(self):
        self.assertEqual(planned_blocks(1700, 2),
                         [(1700, "cold"), (1700, "warm"),
                          (1701, "warm"), (1701, "cold")])
        tasks = task_plan(1700)
        self.assertEqual(len(tasks), 12)
        self.assertEqual(sorted(task[1] for task in tasks),
                         [4096] * 4 + [5120] * 4 + [6144] * 4)
        self.assertEqual({task[3] for task in tasks}, {64})

    def test_complete_synthetic_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            blocks = []
            for seed, arm in planned_blocks(1700, 2):
                log_name = f"pool-v1-seed{seed}-{arm}-gpu-util.log"
                sm = 80 if arm == "warm" else 20
                (root / log_name).write_text(f"# gpu sm mem\n0 {sm} 0\n1 {sm} 0\n")
                runs = []
                for policy in ("round_robin", "feedback"):
                    runs.append({"repetition": 0, "policy": policy,
                                 "makespan_ms": 100, "gpu0_event_ms": 80,
                                 "gpu1_event_ms": 80,
                                 "tasks": [{"id": index, "n": 4096,
                                            "gpu": index % 2, "repeats": 64,
                                            "gpu_ms": 10, "h2d_ms": 2,
                                            "input_prepare_ms": 0,
                                            "correct": True}
                                           for index in range(12)]})
                blocks.append({"seed": seed, "arm": arm,
                               "wall_ms": 250 if arm == "warm" else 400,
                               "batch_makespan_sum_ms": 200,
                               "gpu_util_log": log_name, "runs": runs})
            data = {"protocol": "persistent_pool_v1", "seed_start": 1700,
                    "seed_count": 2, "repetitions": 1, "gemm_repeats": 64,
                    "fixed_input_scale": 1.0, "blocks": blocks}
            self.assertEqual(validate_checkpoint(data, planned_blocks(1700, 2), 1), 4)
            raw = root / "pool-v1.json"
            raw.write_text(json.dumps(data))
            prefix = root / "pool-v1"
            subprocess.run([sys.executable, str(EXPERIMENTS / "analyze_pool_control.py"),
                            str(raw), "--out-prefix", str(prefix)],
                           check=True, capture_output=True, text=True)
            summary = json.loads((root / "pool-v1-summary.json").read_text())
            self.assertEqual(summary["run_count"], 8)
            self.assertEqual(summary["task_count"], 96)
            self.assertEqual(summary["warm_faster_seeds"], 2)
            self.assertGreater(summary["occupancy_descriptive"]["warm"]["0"]["mean_sm_pct"],
                               summary["occupancy_descriptive"]["cold"]["0"]["mean_sm_pct"])


if __name__ == "__main__":
    unittest.main()
