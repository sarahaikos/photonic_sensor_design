"""Unit tests for compact-model vs analytic Theory checks, including edges."""

from __future__ import annotations

import math
import unittest

from circuit import analyze_circuit
from coupler import analyze_coupler
from ring import kappa_for_critical, radius_for_laser
from sparams import power_db, scale_spectrum_power
from theory import circuit_theory
from waveguide import analyze_waveguide


def _circuit(
    config: str = "all-pass",
    *,
    width_nm: float = 450.0,
    height_nm: float = 220.0,
    wavelength_nm: float = 1550.0,
    n_clad: float = 1.33,
    radius_um: float = 10.0,
    gap_nm: float = 200.0,
    length_um: float = 12.0,
    extra_drop_bus: bool = False,
) -> dict:
    devices: list[dict] = [
        {"type": "waveguide", "length_um": 100.0},
        {"type": "coupler", "gap_nm": gap_nm, "length_um": length_um},
        {"type": "ring", "radius_um": radius_um, "config": config},
    ]
    if extra_drop_bus:
        devices.insert(1, {"type": "waveguide", "length_um": 100.0})
        devices.insert(3, {"type": "coupler", "gap_nm": gap_nm, "length_um": length_um})
    return analyze_circuit(
        width_nm=width_nm,
        height_nm=height_nm,
        wavelength_nm=wavelength_nm,
        n_clad=n_clad,
        polarization="TE",
        devices=devices,
    )


def _starter(config: str = "all-pass") -> dict:
    return _circuit(config)


def _by_id(theory: dict) -> dict:
    return {c["id"]: c for c in theory["checks"]}


def _synthetic_ring(*, n_eff: float, n_g: float, radius_um: float, a: float, q: float) -> dict:
    L = 2 * math.pi * radius_um
    m = 100
    lam0 = n_eff * L / m * 1e3
    fsr = ((lam0 * 1e-3) ** 2) / (n_g * L) * 1e3
    return {
        "resonance_nm": lam0,
        "fsr_nm": fsr,
        "mode_order": m,
        "round_trip_amplitude": a,
        "q": q,
        "extracted": {"q_loaded": q},
        "coupling_regime": "critical",
    }


class TheoryChecksTest(unittest.TestCase):
    def test_all_pass_theory_shape(self):
        result = _starter("all-pass")
        theory = result["theory"]
        self.assertIsNotNone(theory)
        ids = [c["id"] for c in theory["checks"]]
        self.assertEqual(ids, ["fsr", "resonance", "kappa_crit", "q_loaded", "beat_length"])
        by_id = {c["id"]: c for c in theory["checks"]}
        self.assertTrue(by_id["fsr"]["ok"])
        self.assertTrue(by_id["resonance"]["ok"])
        self.assertTrue(by_id["q_loaded"]["ok"])
        self.assertTrue(by_id["beat_length"]["ok"])
        # Default gap is strongly overcoupled vs tiny κ_crit.
        self.assertFalse(by_id["kappa_crit"]["ok"])

    def test_add_drop_theory_and_critical(self):
        result = _starter("add-drop")
        theory = result["theory"]
        self.assertIsNotNone(theory)
        self.assertEqual(result.get("coupling_regime"), "overcoupled")
        by_id = {c["id"]: c for c in theory["checks"]}
        self.assertTrue(by_id["fsr"]["ok"])
        self.assertTrue(by_id["resonance"]["ok"])
        self.assertFalse(by_id["kappa_crit"]["ok"])

        # Add-drop κ_crit formula: κ = 1 − a.
        a = result["critical"]["a"]
        self.assertAlmostEqual(kappa_for_critical(a, "add-drop"), 1.0 - a, places=6)
        self.assertAlmostEqual(result["critical"]["kappa"], 1.0 - a, places=6)

        # Same as the UI "Set critical κ" action: patch preferred gap or length.
        crit = result["critical"]
        self.assertTrue(crit["reachable"])
        gap_nm = 200.0
        length_um = 12.0
        if crit["preferred"] == "gap" and crit["gap_nm"] is not None:
            gap_nm = crit["gap_nm"]
        else:
            length_um = crit["length_um"]

        tuned = analyze_circuit(
            width_nm=450.0,
            height_nm=220.0,
            wavelength_nm=1550.0,
            n_clad=1.33,
            polarization="TE",
            devices=[
                {"type": "waveguide", "length_um": 100.0},
                {"type": "waveguide", "length_um": 100.0},
                {"type": "coupler", "gap_nm": gap_nm, "length_um": length_um},
                {"type": "coupler", "gap_nm": gap_nm, "length_um": length_um},
                {"type": "ring", "radius_um": 10.0, "config": "add-drop"},
            ],
        )
        self.assertEqual(tuned.get("coupling_regime"), "critical")
        tuned_by_id = {c["id"]: c for c in tuned["theory"]["checks"]}
        self.assertTrue(tuned_by_id["kappa_crit"]["ok"])
        self.assertTrue(tuned_by_id["fsr"]["ok"])
        self.assertTrue(tuned_by_id["q_loaded"]["ok"])

    def test_circuit_theory_resonance_identity(self):
        ring = {
            "resonance_nm": 1550.0,
            "fsr_nm": 8.7,
            "mode_order": 100,
            "round_trip_amplitude": 0.99,
            "q": 2000.0,
            "extracted": {"q_loaded": 2010.0},
            "coupling_regime": "undercoupled",
        }
        n_eff = 2.4
        radius = 10.0
        L = 2 * 3.141592653589793 * radius
        m = 100
        # Build a ring dict whose resonance matches n_eff L / m.
        lam0 = n_eff * L / m * 1e3
        ring["resonance_nm"] = lam0
        ring["mode_order"] = m
        out = circuit_theory(
            wavelength_nm=1550.0,
            radius_um=radius,
            n_eff=n_eff,
            n_g=4.0,
            kappa=0.01,
            config="all-pass",
            ring=ring,
            critical_kappa=0.02,
            coupler=None,
        )
        res = next(c for c in out["checks"] if c["id"] == "resonance")
        self.assertTrue(res["ok"])
        self.assertAlmostEqual(res["model"], res["theory"], places=9)

    def test_all_pass_kappa_crit_is_one_minus_a_squared(self):
        a = 0.99
        self.assertAlmostEqual(kappa_for_critical(a, "all-pass"), 1.0 - a * a, places=6)
        self.assertAlmostEqual(kappa_for_critical(a, "add-drop"), 1.0 - a, places=6)

    def test_all_pass_set_critical_kappa_passes_theory(self):
        result = _starter("all-pass")
        crit = result["critical"]
        self.assertAlmostEqual(crit["kappa"], 1.0 - crit["a"] ** 2, places=6)
        self.assertTrue(crit["reachable"])
        gap_nm, length_um = 200.0, 12.0
        if crit["preferred"] == "gap" and crit["gap_nm"] is not None:
            gap_nm = crit["gap_nm"]
        else:
            length_um = crit["length_um"]
        tuned = _circuit("all-pass", gap_nm=gap_nm, length_um=length_um)
        self.assertEqual(tuned.get("coupling_regime"), "critical")
        checks = _by_id(tuned["theory"])
        self.assertTrue(checks["kappa_crit"]["ok"])
        self.assertTrue(checks["fsr"]["ok"])
        self.assertTrue(checks["q_loaded"]["ok"])

    def test_fsr_uses_n_g_resonance_uses_n_eff(self):
        n_eff, n_g, radius = 2.4, 4.2, 10.0
        ring = _synthetic_ring(n_eff=n_eff, n_g=n_g, radius_um=radius, a=0.99, q=2500.0)
        out = circuit_theory(
            wavelength_nm=ring["resonance_nm"],
            radius_um=radius,
            n_eff=n_eff,
            n_g=n_g,
            kappa=1.0 - 0.99**2,
            config="all-pass",
            ring=ring,
            critical_kappa=1.0 - 0.99**2,
        )
        checks = _by_id(out)
        self.assertTrue(checks["fsr"]["ok"])
        self.assertTrue(checks["resonance"]["ok"])
        L = 2 * math.pi * radius
        self.assertAlmostEqual(
            checks["fsr"]["theory"],
            ((ring["resonance_nm"] * 1e-3) ** 2) / (n_g * L) * 1e3,
            places=9,
        )
        self.assertAlmostEqual(
            checks["resonance"]["theory"],
            n_eff * L / ring["mode_order"] * 1e3,
            places=9,
        )

    def test_fsr_check_fails_when_laser_is_far_from_resonance(self):
        """Model FSR uses laser λ; theory uses λ₀ — large detuning exceeds 2% tol."""
        n_eff, n_g, radius = 2.4, 4.0, 10.0
        ring = _synthetic_ring(n_eff=n_eff, n_g=n_g, radius_um=radius, a=0.99, q=2000.0)
        laser_nm = ring["resonance_nm"] + 80.0
        L = 2 * math.pi * radius
        ring["fsr_nm"] = ((laser_nm * 1e-3) ** 2) / (n_g * L) * 1e3
        out = circuit_theory(
            wavelength_nm=laser_nm,
            radius_um=radius,
            n_eff=n_eff,
            n_g=n_g,
            kappa=0.01,
            config="all-pass",
            ring=ring,
            critical_kappa=0.02,
        )
        fsr = _by_id(out)["fsr"]
        self.assertFalse(fsr["ok"])
        expected = abs(laser_nm**2 - ring["resonance_nm"] ** 2) / ring["resonance_nm"] ** 2
        self.assertAlmostEqual(fsr["rel_error"], expected, places=6)

    def test_resonance_na_when_mode_order_is_zero(self):
        ring = _synthetic_ring(n_eff=2.4, n_g=4.0, radius_um=10.0, a=0.99, q=2000.0)
        ring["mode_order"] = 0
        out = circuit_theory(
            wavelength_nm=1550.0,
            radius_um=10.0,
            n_eff=2.4,
            n_g=4.0,
            kappa=0.01,
            config="all-pass",
            ring=ring,
            critical_kappa=0.02,
        )
        res = _by_id(out)["resonance"]
        self.assertIsNone(res["theory"])
        self.assertIsNone(res["ok"])
        self.assertEqual(out["summary"]["na"], 1)

    def test_q_loaded_fails_when_spectrum_fit_disagrees(self):
        ring = _synthetic_ring(n_eff=2.4, n_g=4.0, radius_um=10.0, a=0.99, q=2000.0)
        ring["extracted"] = {"q_loaded": 4000.0}
        out = circuit_theory(
            wavelength_nm=ring["resonance_nm"],
            radius_um=10.0,
            n_eff=2.4,
            n_g=4.0,
            kappa=0.02,
            config="all-pass",
            ring=ring,
            critical_kappa=0.02,
        )
        q_row = _by_id(out)["q_loaded"]
        self.assertFalse(q_row["ok"])
        self.assertGreater(q_row["rel_error"], q_row["tol"])

    def test_no_beat_length_row_without_coupler(self):
        ring = _synthetic_ring(n_eff=2.4, n_g=4.0, radius_um=10.0, a=0.99, q=2000.0)
        out = circuit_theory(
            wavelength_nm=ring["resonance_nm"],
            radius_um=10.0,
            n_eff=2.4,
            n_g=4.0,
            kappa=0.02,
            config="all-pass",
            ring=ring,
            critical_kappa=0.02,
            coupler=None,
        )
        self.assertEqual([c["id"] for c in out["checks"]], ["fsr", "resonance", "kappa_crit", "q_loaded"])

    def test_beat_length_identity_and_power_split(self):
        coupler = analyze_coupler(200.0, 12.0, 450.0, 1550.0, "TE", 220.0)
        result = _starter("all-pass")
        beat = _by_id(result["theory"])["beat_length"]
        self.assertTrue(beat["ok"])
        self.assertAlmostEqual(
            coupler["beat_length_um"],
            (math.pi / 2) / coupler["kappa0_per_um"],
            places=9,
        )
        phase = coupler["kappa0_per_um"] * 12.0
        self.assertAlmostEqual(coupler["kappa"], math.sin(phase) ** 2, places=9)
        self.assertAlmostEqual(coupler["t_through"] + coupler["t_drop"], 1.0, places=9)

    def test_coupler_wide_gap_weak_coupling_narrow_gap_stronger(self):
        wide = analyze_coupler(500.0, 12.0, 450.0, 1550.0, "TE", 220.0)
        mid = analyze_coupler(200.0, 12.0, 450.0, 1550.0, "TE", 220.0)
        tight = analyze_coupler(50.0, 12.0, 450.0, 1550.0, "TE", 220.0)
        self.assertLess(wide["kappa"], mid["kappa"])
        self.assertLess(mid["kappa0_per_um"], tight["kappa0_per_um"])
        self.assertGreaterEqual(wide["kappa"], 0.0)
        self.assertLessEqual(tight["kappa"], 1.0)
        # Long interaction length wraps κ = sin²(κ₀ L) back toward through.
        long_tight = analyze_coupler(50.0, 80.0, 450.0, 1550.0, "TE", 220.0)
        self.assertGreaterEqual(long_tight["kappa"], 0.0)
        self.assertLessEqual(long_tight["kappa"], 1.0)
        self.assertAlmostEqual(long_tight["t_through"] + long_tight["t_drop"], 1.0, places=9)

    def test_kappa_for_critical_clamps_extreme_round_trip(self):
        self.assertAlmostEqual(kappa_for_critical(1e-9, "all-pass"), 0.95, places=6)
        self.assertAlmostEqual(kappa_for_critical(0.9999999, "all-pass"), 1e-4, places=9)
        self.assertAlmostEqual(kappa_for_critical(1e-9, "add-drop"), 0.95, places=6)
        self.assertGreaterEqual(kappa_for_critical(0.9999999, "add-drop"), 1e-4)

    def test_cut_off_waveguide_rejects_theory(self):
        too_narrow = analyze_waveguide(200.0, 220.0, 1550.0, 1.33, "TE")
        too_thin = analyze_waveguide(450.0, 100.0, 1550.0, 1.33, "TE")
        self.assertIsNone(too_narrow)
        self.assertIsNone(too_thin)
        circuit = analyze_circuit(
            width_nm=200.0,
            height_nm=220.0,
            wavelength_nm=1550.0,
            n_clad=1.33,
            polarization="TE",
            devices=[{"type": "ring", "radius_um": 10.0, "config": "all-pass"}],
        )
        self.assertIn("error", circuit)
        self.assertIsNone(circuit.get("theory"))

    def test_radius_for_laser_satisfies_resonance_condition(self):
        n_eff = 2.4
        wavelength_nm = 1550.0
        r = radius_for_laser(n_eff, wavelength_nm, 10.0)
        L = 2 * math.pi * r
        m = round(n_eff * L / (wavelength_nm * 1e-3))
        lam0 = n_eff * L / m * 1e3
        self.assertAlmostEqual(lam0, wavelength_nm, places=6)

    def test_small_radius_increases_bend_loss_and_kappa_crit(self):
        large = _circuit("all-pass", radius_um=20.0)
        tiny = _circuit("all-pass", radius_um=3.0)
        self.assertLess(tiny["critical"]["a"], large["critical"]["a"])
        self.assertGreater(tiny["critical"]["kappa"], large["critical"]["kappa"])
        self.assertTrue(_by_id(tiny["theory"])["resonance"]["ok"])
        self.assertTrue(_by_id(large["theory"])["resonance"]["ok"])

    def test_spectrum_power_scale_updates_linear_and_db(self):
        spec = {
            "wavelength_nm": [1550.0, 1551.0],
            "through": [1.0, 0.25],
            "t_through": [1.0, 0.25],
            "drop": [0.0, 0.5],
            "t_drop": [0.0, 0.5],
            "s21_db": [0.0, -6.020599913279624],
            "s31_db": [-180.0, -3.010299956639812],
        }
        out = scale_spectrum_power(spec, 0.5)
        self.assertAlmostEqual(out["through"][0], 0.5)
        self.assertAlmostEqual(out["t_through"][1], 0.125)
        self.assertAlmostEqual(out["drop"][1], 0.25)
        self.assertAlmostEqual(out["s21_db"][0], power_db(0.5), places=6)
        self.assertAlmostEqual(out["s21_db"][1], power_db(0.125), places=5)
        self.assertIs(scale_spectrum_power(spec, 1.0), spec)
        self.assertIsNone(scale_spectrum_power(None, 0.5))

    def test_circuit_spectrum_includes_db_and_tracks_waveguide_loss(self):
        result = _starter("all-pass")
        spec = result["spectrum"]
        self.assertGreater(len(spec["wavelength_nm"]), 10)
        self.assertEqual(len(spec["through"]), len(spec["s21_db"]))
        i0 = min(range(len(spec["wavelength_nm"])), key=lambda i: spec["through"][i])
        self.assertAlmostEqual(spec["s21_db"][i0], power_db(spec["through"][i0]), places=4)
        lossy = _circuit("all-pass")
        # Extra bus length only changes the constant T scale, not λ₀.
        longer = analyze_circuit(
            width_nm=450.0,
            height_nm=220.0,
            wavelength_nm=1550.0,
            n_clad=1.33,
            polarization="TE",
            devices=[
                {"type": "waveguide", "length_um": 2000.0},
                {"type": "coupler", "gap_nm": 200.0, "length_um": 12.0},
                {"type": "ring", "radius_um": 10.0, "config": "all-pass"},
            ],
        )
        self.assertLess(min(longer["spectrum"]["through"]), min(lossy["spectrum"]["through"]))
        self.assertAlmostEqual(
            longer["spectrum"]["s21_db"][0] - power_db(longer["spectrum"]["through"][0]),
            0.0,
            places=4,
        )
        self.assertAlmostEqual(longer["resonance_nm"], lossy["resonance_nm"], places=4)

    def test_air_cladding_still_matches_resonance_and_fsr(self):
        result = _circuit("all-pass", n_clad=1.0)
        self.assertIsNotNone(result.get("theory"))
        checks = _by_id(result["theory"])
        self.assertTrue(checks["resonance"]["ok"])
        self.assertTrue(checks["fsr"]["ok"])
        self.assertTrue(checks["beat_length"]["ok"])


if __name__ == "__main__":
    unittest.main()
