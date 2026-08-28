"""Tiny unit tests for compact-model vs analytic Theory checks."""

from __future__ import annotations

import unittest

from circuit import analyze_circuit
from ring import kappa_for_critical
from theory import circuit_theory


def _starter(config: str = "all-pass") -> dict:
    return analyze_circuit(
        width_nm=450.0,
        height_nm=220.0,
        wavelength_nm=1550.0,
        n_clad=1.33,
        polarization="TE",
        devices=[
            {"type": "waveguide", "length_um": 100.0},
            {"type": "coupler", "gap_nm": 200.0, "length_um": 12.0},
            {"type": "ring", "radius_um": 10.0, "config": config},
        ],
    )


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


if __name__ == "__main__":
    unittest.main()
