"""CPU-only checks for a costly, preregistered GPU campaign."""

import sys
import unittest
from collections import Counter
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from factorial_host_input import FIXED_INPUT_SCALE, planned_cells, validate_checkpoint
from study import operand_scale


class FactorialPreparationTests(unittest.TestCase):
    def sample_checkpoint(self):
        cells = planned_cells(1500, 2)
        runs = []
        for seed, reused, workers in cells:
            mode = "host_to_device_reused_pinned" if reused else "host_to_device"
            for rep in range(2):
                for policy in ("round_robin", "feedback"):
                    runs.append({"seed": seed, "mode": mode,
                                 "host_load_workers": workers,
                                 "fixed_input_scale": FIXED_INPUT_SCALE,
                                 "workload": "transfer_hidden",
                                 "repetition": rep, "policy": policy,
                                 "tasks": [{"id": index, "repeats": 32 if index < 4 else 4,
                                            "correct": True}
                                           for index in range(12)]})
        return cells, {"protocol": "factorial_host_input_v2",
                       "fixed_input_scale": FIXED_INPUT_SCALE, "runs": runs}

    def test_latin_order_balances_each_position(self):
        cells = planned_cells(1500, 12)
        self.assertEqual(len(cells), 48)
        for position in range(4):
            counts = Counter((reused, workers) for _, reused, workers
                             in cells[position::4])
            self.assertEqual(set(counts.values()), {3})
            self.assertEqual(len(counts), 4)

    def test_fixed_scale_removes_numeric_difference_and_preserves_legacy_default(self):
        self.assertEqual(operand_scale(1500, False, False, 1.0), 1.0)
        self.assertEqual(operand_scale(1500, False, True, 1.0), 1.0)
        self.assertEqual(operand_scale(1500, False, True), 1.0)
        self.assertNotEqual(operand_scale(1500, False, False), 1.0)

    def test_checkpoint_accepts_complete_cells_and_rejects_corruption(self):
        cells, data = self.sample_checkpoint()
        self.assertEqual(validate_checkpoint(data, cells, 2), 8)
        data["runs"][0]["policy"] = "feedback"
        with self.assertRaisesRegex(ValueError, "Missing or duplicate"):
            validate_checkpoint(data, cells, 2)
        data["runs"][0]["policy"] = "round_robin"
        data["runs"][0]["tasks"][0]["correct"] = False
        with self.assertRaisesRegex(ValueError, "Invalid run"):
            validate_checkpoint(data, cells, 2)

    def test_checkpoint_refuses_partial_cell(self):
        cells, data = self.sample_checkpoint()
        data["runs"].pop()
        with self.assertRaisesRegex(ValueError, "incomplete cell"):
            validate_checkpoint(data, cells, 2)


if __name__ == "__main__":
    unittest.main()
