"""Uniform-ΔT thermo-optic corners on compact models."""

from __future__ import annotations

import unittest

from circuit import analyze_circuit
from thermo import DN_DT_SI, dn_eff_dt, kappa0_at, resonance_shift_nm


def _circuit() -> dict:
    return analyze_circuit(
        width_nm=450.0,
        height_nm=220.0,
        wavelength_nm=1550.0,
        n_clad=1.33,
        polarization="TE",
        devices=[
            {"type": "waveguide", "length_um": 100.0},
            {"type": "coupler", "gap_nm": 200.0, "length_um": 12.0},
            {"type": "ring", "radius_um": 10.0, "config": "all-pass"},
        ],
    )


class ThermoCornersTest(unittest.TestCase):
    def test_dn_eff_dt_is_between_clad_and_silicon(self):
        dndt = dn_eff_dt(0.81, 0.10, 1.444)
        self.assertGreater(dndt, 0.0)
        self.assertLess(dndt, DN_DT_SI)
        self.assertAlmostEqual(dndt, 0.81 * DN_DT_SI + 0.10 * 1.0e-5, places=9)

    def test_resonance_redshifts_and_kappa_drops_when_hot(self):
        result = _circuit()
        thermal = result["thermal_corners"]
        self.assertIsNotNone(thermal)
        by_dt = {p["delta_t_k"]: p for p in thermal["points"]}
        cold = by_dt[-20.0]
        hot = by_dt[40.0]
        self.assertGreater(hot["resonance_nm"], cold["resonance_nm"])
        self.assertLess(hot["kappa"], cold["kappa"])
        self.assertAlmostEqual(by_dt[0.0]["shift_nm"], 0.0, places=9)
        self.assertAlmostEqual(
            thermal["dlambda_dt_nm_per_k"],
            resonance_shift_nm(result["resonance_nm"], result["mode"]["n_g"], thermal["dn_eff_dt"], 1.0),
            places=9,
        )

    def test_worst_thermal_corners_are_extreme_delta_t(self):
        thermal = _circuit()["thermal_corners"]
        self.assertIn(thermal["worst_lambda"]["delta_t_k"], (-20.0, 40.0))
        self.assertIn(thermal["worst_kappa"]["delta_t_k"], (-20.0, 40.0))
        self.assertGreater(thermal["worst_lambda"]["abs_dlambda_nm"], 0.0)

    def test_hotter_core_weakens_kappa0(self):
        self.assertLess(kappa0_at(0.05, 1.5e-4, 1.444, 20.0), 0.05)
        self.assertGreater(kappa0_at(0.05, 1.5e-4, 1.444, -20.0), 0.05)


if __name__ == "__main__":
    unittest.main()
