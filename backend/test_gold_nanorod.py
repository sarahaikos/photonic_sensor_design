"""Single-frequency gold permittivity checked on spheres (Mie) and nanorods (Gans)."""

from __future__ import annotations

import math
import unittest

import numpy as np

from fdtd.materials import C0, GOLD, HBAR_EV_S, MATERIALS, permittivity
from fdtd.nanorod import (
    absorption_cross_section,
    depolarization_prolate,
    lspr,
    sphere_comparison,
    spheroid_volume,
    survey_nanorods,
)


class PermittivityTest(unittest.TestCase):
    def test_rakic_formula_in_ev_matches_the_pole_sum(self):
        for nm in (450.0, 600.0, 750.0, 1000.0):
            omega = 2.0 * math.pi * C0 / (nm * 1e-9)
            got = permittivity(GOLD, omega)
            np.testing.assert_allclose(got, _rakic_ev(omega * HBAR_EV_S), rtol=1e-12, atol=0.0)

    def test_gold_is_passive_and_metallic_across_the_visible(self):
        for nm in range(450, 1001, 50):
            eps = permittivity(GOLD, 2.0 * math.pi * C0 / (nm * 1e-9))
            self.assertGreater(eps.imag, 0.0)
            self.assertLess(eps.real, 0.0)

    def test_drude_pole_rejects_zero_frequency(self):
        with self.assertRaises(ValueError):
            permittivity(GOLD, 0.0)

    def test_static_lorentz_and_debye_keep_their_oscillator_strength(self):
        silicon = MATERIALS["si"]
        eps = permittivity(silicon, 0.0)
        self.assertAlmostEqual(eps.real, silicon.eps_inf + silicon.poles[0].eps_delta, places=12)
        self.assertAlmostEqual(eps.imag, 0.0, places=12)
        water = MATERIALS["water"]
        static = permittivity(water, 0.0)
        self.assertAlmostEqual(static.real, water.eps_inf + water.poles[0].eps_delta, places=12)


class NanorodTest(unittest.TestCase):
    def test_sphere_and_needle_limits_of_the_depolarization(self):
        self.assertAlmostEqual(depolarization_prolate(1.0), 1.0 / 3.0, places=12)
        self.assertAlmostEqual(depolarization_prolate(3.0), _textbook_l(3.0), places=12)
        self.assertLess(depolarization_prolate(5.0), depolarization_prolate(2.0))
        self.assertLess(depolarization_prolate(8.0), 0.03)

    def test_longer_rods_resonate_further_to_the_red(self):
        peaks = [
            lspr(aspect, 10e-9, 1.333)["lspr_nm"]
            for aspect in (1.0, 2.0, 3.0, 4.0, 5.0)
        ]
        self.assertEqual(peaks, sorted(peaks))
        self.assertGreater(peaks[0], 500.0)
        self.assertLess(peaks[0], 540.0)
        self.assertGreater(peaks[2], 690.0)
        self.assertLess(peaks[2], 730.0)
        self.assertGreater(peaks[4], 900.0)

    def test_peak_sits_on_the_real_permittivity_condition(self):
        row = lspr(3.0, 10e-9, 1.333)
        # Im ε moves the absorption peak a fraction of a linewidth off Re ε = ε_m (1 − 1/L).
        self.assertLess(abs(row["eps"].real - row["target_eps"]), 0.25)
        self.assertGreater(row["c_abs_m2"], 0.0)

    def test_transverse_resonance_stays_blue_of_the_longitudinal_one(self):
        long = lspr(4.0, 10e-9, 1.333, longitudinal=True)
        trans = lspr(4.0, 10e-9, 1.333, longitudinal=False)
        self.assertLess(trans["lspr_nm"], long["lspr_nm"] - 200.0)
        self.assertGreater(trans["depolarization"], long["depolarization"])

    def test_each_wavelength_uses_its_own_permittivity(self):
        """A permittivity frozen at 500 nm is not the resonant value at the LSPR."""
        row = lspr(3.0, 12e-9, 1.333)
        eps_500 = permittivity(GOLD, 2.0 * math.pi * C0 / 500e-9)
        medium = 1.333**2
        volume = spheroid_volume(3.0, 12e-9)
        on = absorption_cross_section(
            row["eps"], medium, row["depolarization"], volume, row["lspr_nm"] * 1e-9
        )
        off = absorption_cross_section(
            eps_500, medium, row["depolarization"], volume, row["lspr_nm"] * 1e-9
        )
        self.assertAlmostEqual(on, row["c_abs_m2"], places=6)
        self.assertLess(off, 0.25 * on)


class MieLimitTest(unittest.TestCase):
    def test_small_gold_sphere_matches_mie(self):
        for nm in (500.0, 600.0, 800.0):
            row = sphere_comparison(5e-9, nm * 1e-9)
            self.assertLess(row["relative_error"], 0.03)
            self.assertGreater(row["mie_m2"], 0.0)

    def test_forty_nanometer_sphere_leaves_the_quasi_static_model(self):
        row = sphere_comparison(40e-9, 530e-9)
        self.assertGreater(row["relative_error"], 0.15)

    def test_survey_reports_the_sphere_mie_check(self):
        sphere, rod = survey_nanorods((1.0, 3.0), semi_minor_m=10e-9)
        self.assertAlmostEqual(
            sphere["longitudinal"]["lspr_nm"], sphere["transverse"]["lspr_nm"], places=6
        )
        self.assertLess(sphere["mie"]["relative_error"], 0.05)
        self.assertLess(rod["transverse"]["lspr_nm"], rod["longitudinal"]["lspr_nm"])


def _rakic_ev(energy_ev: complex) -> complex:
    """Table I of Rakić et al. (1998), written in eV and independent of ``Pole``."""
    plasma = 9.03
    eps = 1.0 - 0.760 * plasma**2 / (energy_ev * (energy_ev + 1j * 0.053))
    table = (
        (0.024, 0.241, 0.415),
        (0.010, 0.345, 0.830),
        (0.071, 0.870, 2.969),
        (0.601, 2.494, 4.304),
        (4.384, 2.214, 13.32),
    )
    for strength, gamma, omega0 in table:
        eps += strength * plasma**2 / ((omega0**2 - energy_ev**2) - 1j * energy_ev * gamma)
    return eps


def _textbook_l(aspect: float) -> float:
    ecc = math.sqrt(1.0 - 1.0 / aspect**2)
    return (1.0 - ecc**2) / ecc**2 * (-1.0 + math.log((1.0 + ecc) / (1.0 - ecc)) / (2.0 * ecc))


if __name__ == "__main__":
    unittest.main()
