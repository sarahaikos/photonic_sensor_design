"""Uniform-ΔT corners, heater heat solve, and thermo-optic TO."""

from __future__ import annotations

import unittest

import numpy as np

from circuit import analyze_circuit
from thermo import (
    DN_DT_SI,
    G_SINK,
    default_coupler_heater,
    dn_eff_dt,
    heater_layout,
    kappa0_at,
    paint_heaters,
    resonance_shift_nm,
    sheet_k,
    solve_circuit_heat,
    solve_heat,
)
from topology import run_topology


def _devices(heater: bool = False) -> list[dict]:
    out = [
        {"type": "waveguide", "length_um": 100.0, "x": 260, "y": 210},
        {"type": "coupler", "gap_nm": 200.0, "length_um": 12.0, "x": 250, "y": 210},
        {"type": "ring", "radius_um": 10.0, "config": "all-pass", "x": 250, "y": 134},
    ]
    if heater:
        out.append(
            {
                "type": "heater",
                "x": 235,
                "y": 134,
                "power_mw": 10.0,
                "width_um": 2.0,
                "length_um": 40.0,
            }
        )
    return out


def _circuit(heater: bool = False) -> dict:
    return analyze_circuit(
        width_nm=450.0,
        height_nm=220.0,
        wavelength_nm=1550.0,
        n_clad=1.33,
        polarization="TE",
        devices=_devices(heater),
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


class HeatSolveTest(unittest.TestCase):
    def test_temperature_peaks_under_the_heater(self):
        xs = np.linspace(-4.0, 4.0, 41)
        ys = np.linspace(-4.0, 4.0, 41)
        xx, yy = np.meshgrid(xs, ys, indexing="ij")
        q = np.where((np.abs(xx) < 1.0) & (np.abs(yy) < 2.0), 8.0e7, 0.0)
        temp = solve_heat(xs, ys, q, sheet_k(np.zeros_like(q)), G_SINK)
        self.assertGreater(temp.max(), 2.0)
        self.assertGreater(temp[20, 20], temp[2, 2])

    def test_heat_is_linear_in_power(self):
        xs = np.linspace(-3.0, 3.0, 31)
        ys = np.linspace(-3.0, 3.0, 31)
        heaters = [default_coupler_heater(0.45, 3.0, 1.0, power_mw=10.0)]
        k = sheet_k(np.zeros((xs.size, ys.size)))
        t1 = solve_heat(xs, ys, paint_heaters(xs, ys, heaters, 1.0), k)
        t2 = solve_heat(xs, ys, paint_heaters(xs, ys, heaters, 0.5), k)
        self.assertGreater(t1.max(), 0.5)
        self.assertAlmostEqual(t1.max() / (t2.max() + 1e-12), 2.0, places=1)

    def test_heater_layout_sits_by_the_ring(self):
        layout = heater_layout(_devices(heater=True), 450.0)
        self.assertEqual(len(layout), 1)
        self.assertGreater(layout[0]["power_mw"], 0.0)
        self.assertLess(layout[0]["width"], 5.0)

    def test_heater_outline_sits_on_the_hot_spot(self):
        heat = solve_circuit_heat(_devices(heater=True), 450.0)
        self.assertIsNotNone(heat)
        field = heat["field"]
        self.assertTrue(field["polygons"])
        ys = [p["y"] for p in field["polygons"][0]]
        temp = np.asarray(field["intensity"])
        ix = int(np.unravel_index(int(temp.argmax()), temp.shape)[0])
        x_hot = field["z_um"][ix]
        self.assertGreaterEqual(max(ys) + 1.0, x_hot)
        self.assertLessEqual(min(ys) - 1.0, x_hot)
        guides = field["guides"]
        self.assertTrue(any(g["height"] < 3.5 for g in guides))

    def test_circuit_heater_redshifts_more_than_heater_off(self):
        heat = solve_circuit_heat(_devices(heater=True), 450.0)
        self.assertIsNotNone(heat)
        self.assertGreater(heat["dt_max_k"], 1.0)
        thermal = _circuit(heater=True)["thermal_corners"]
        self.assertIsNotNone(thermal["heater"])
        by_s = {p["scale"]: p for p in thermal["heater"]["points"]}
        self.assertAlmostEqual(by_s[0.0]["shift_nm"], 0.0, places=9)
        self.assertGreater(by_s[1.0]["shift_nm"], by_s[0.0]["shift_nm"])
        self.assertGreater(thermal["heater"]["field"]["dt_max_k"], 0.0)


class ThermoOpticLoopTest(unittest.TestCase):
    def test_thermo_optic_to_scores_cold_and_hot(self):
        result = run_topology(
            450.0,
            220.0,
            1550.0,
            1.33,
            "TE",
            _devices(heater=True),
            steps=1,
            kappa_target=0.12,
            thermo_optic=True,
        )
        self.assertTrue(result["thermo_optic"])
        self.assertIsNotNone(result["thermal"])
        self.assertIsNotNone(result["field_t"])
        self.assertGreater(result["thermal"]["dt_max_k"], 0.0)
        self.assertIn("t_drop_off", result["thermal"])
        self.assertIn("t_drop_hot", result["thermal"])
        if result["history"]:
            self.assertIn(result["history"][0].get("thermal"), ("off", "hot"))


if __name__ == "__main__":
    unittest.main()
