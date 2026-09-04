"""Robust dilated/eroded topology experiments."""

from __future__ import annotations

import unittest

from topology import run_robustness_sweep, run_topology


def _coupler_devices() -> list[dict]:
    return [
        {"type": "waveguide", "length_um": 100.0},
        {"type": "coupler", "gap_nm": 200.0, "length_um": 12.0},
    ]


class TopologySweepTest(unittest.TestCase):
    def test_single_run_includes_eta_curve(self):
        result = run_topology(
            450.0, 220.0, 1550.0, 1.33, "TE", _coupler_devices(), steps=2, kappa_target=0.12
        )
        curve = result["eta_curve"]
        self.assertIsNotNone(curve)
        etas = [p["eta"] for p in curve["points"]]
        self.assertIn(0.3, etas)
        self.assertIn(0.5, etas)
        self.assertIn(0.7, etas)
        self.assertGreater(curve["spread"], 0.0)
        self.assertGreaterEqual(curve["worst_abs_dkappa"], 0.0)

    def test_sweep_compares_robust_and_intermediate(self):
        result = run_robustness_sweep(
            450.0, 220.0, 1550.0, 1.33, "TE", _coupler_devices(), steps=1, kappa_target=0.12
        )
        self.assertEqual(len(result["rows"]), 6)
        modes = {r["mode"] for r in result["rows"]}
        self.assertEqual(modes, {"robust", "intermediate"})
        self.assertEqual(sorted(result["mfs_nm"]), [100.0, 150.0, 220.0])
        for row in result["rows"]:
            self.assertIn(row["worst"], ("dilated", "intermediate", "eroded", None))
            self.assertIsNotNone(row["worst_abs_dkappa"])
        self.assertIsNotNone(result["eta_curve"])

    def test_sweep_needs_a_coupler(self):
        result = run_robustness_sweep(
            450.0, 220.0, 1550.0, 1.33, "TE", [{"type": "waveguide", "length_um": 100.0}]
        )
        self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main()
