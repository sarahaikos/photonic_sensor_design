"""Lorentz dispersion in the Yee stepper and in the FDFD resonance sweep."""

from __future__ import annotations

import unittest

import numpy as np

from fdtd.materials import EPS0, Pole, _pole_permittivity, ade_lorentz_coeff
from fdtd.resonance import (
    block_mask,
    block_permittivity,
    compare_dispersive_resonance,
    fdfd_resonance,
)


_SHAPE = (16, 12)
_SPACING = (40e-9, 50e-9)


class LorentzUpdateTest(unittest.TestCase):
    def test_driven_polarization_matches_the_permittivity_pole(self):
        pole = Pole("lorentz", 0.7, 1.8e15, 5.0e13)
        omega = 1.1e15
        coarse = _polarization_error(pole, omega, 4.0e-18)
        fine = _polarization_error(pole, omega, 1.0e-18)
        self.assertLess(fine, 1e-3)
        # The one-sided electric drive is a truncation error, so a shorter step shrinks it.
        self.assertLess(fine, coarse / 2.0)


class DispersiveResonanceTest(unittest.TestCase):
    def test_both_engines_follow_the_pole_not_a_frozen_permittivity(self):
        eps_inf = block_permittivity(_SHAPE, eps_core=2.25, eps_bg=1.0)
        mask = block_mask(_SHAPE)
        bare = fdfd_resonance(eps_inf, _SPACING)
        # The pole sits above the cavity, so the resonance is on the low-frequency
        # side where the susceptibility is larger than its static value.
        pole = Pole("lorentz", 0.85, 1.55 * bare["omega"], 0.06 * 1.55 * bare["omega"])
        report = compare_dispersive_resonance(
            eps_inf, mask, pole, _SPACING, frozen_omega=0.0, cfl=0.45, steps=2000
        )
        live = report["fdfd"]
        time = report["fdtd"]
        frozen = report["frozen"]

        self.assertLess(time["fit_residual"], 5e-5)
        self.assertGreater(live["wavelength_nm"], frozen["wavelength_nm"])
        self.assertGreater(frozen["wavelength_nm"], bare["wavelength_nm"])
        self.assertLess(_rel(time["omega_corrected"], live["omega"]), 1e-3)
        gap = abs(time["wavelength_corrected_nm"] - live["wavelength_nm"])
        wrong = abs(frozen["wavelength_nm"] - live["wavelength_nm"])
        self.assertGreater(wrong, 20.0 * gap)

        self.assertLess(live["gamma"], 0.0)
        self.assertLess(time["gamma"], 0.0)
        self.assertLess(_rel(time["gamma"], live["gamma"]), 0.02)
        self.assertLess(_rel(time["q_corrected"], live["q"]), 0.02)


def _rel(got: float, want: float) -> float:
    return abs(got - want) / abs(want)


def _polarization_error(pole: Pole, omega: float, dt: float) -> float:
    coeff = ade_lorentz_coeff(pole, dt)
    chi = _pole_permittivity(pole, omega)

    def field(step: int) -> float:
        return float(np.cos(omega * step * dt))

    def polarization(step: int) -> float:
        t = step * dt
        return float(EPS0 * (chi.real * np.cos(omega * t) + chi.imag * np.sin(omega * t)))

    current = polarization(0)
    previous = polarization(-1)
    errors = []
    for step in range(1, 16000):
        averaged = field(step) + 2.0 * field(step - 1) + field(step - 2)
        updated = coeff["c1"] * current + coeff["c2"] * previous + coeff["c3"] * averaged
        if step > 8000:
            errors.append(updated - polarization(step))
        previous, current = current, updated
    scale = EPS0 * abs(chi) + 1e-30
    return float(np.sqrt(np.mean(np.square(errors))) / scale)


if __name__ == "__main__":
    unittest.main()
