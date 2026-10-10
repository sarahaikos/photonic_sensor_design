"""FDTD ringdown against the FDFD curl-curl resonance of the same cavity."""

from __future__ import annotations

import unittest

from fdtd.resonance import block_permittivity, compare_resonances

# Unequal cells, so the comparison is not a square-grid special case.
_SHAPE = (18, 14)
_SPACING = (45e-9, 55e-9)


class ResonanceParityTest(unittest.TestCase):
    def test_defect_index_shifts_both_engines_by_the_same_amount(self):
        cold = block_permittivity(_SHAPE, eps_core=4.0, eps_bg=1.0)
        hot = block_permittivity(_SHAPE, eps_core=6.0, eps_bg=1.0)
        report = compare_resonances((cold, hot), _SPACING, alpha=0.0, cfl=0.5, steps=480)
        self.assertGreater(report["shift_fdfd_nm"], 50.0)
        _assert_wavelengths(self, report, shift_rtol=2e-4, omega_rtol=1e-8)

    def test_quality_factor_tracks_the_same_loss_tangent(self):
        # α = σ / (ε₀ εᵣ). Q = Re(ω) / α, and the amplitude decays as e^{-α t / 2}.
        alpha = 2.0e13
        cold = block_permittivity(_SHAPE, eps_core=4.0, eps_bg=1.0)
        hot = block_permittivity(_SHAPE, eps_core=6.0, eps_bg=1.0)
        report = compare_resonances(
            (cold, hot), _SPACING, alpha=alpha, cfl=0.5, steps=700
        )
        self.assertGreater(report["shift_fdfd_nm"], 50.0)
        # Loss moves the leapfrog map off the lossless arcsin relation by ~1e-7.
        _assert_wavelengths(self, report, shift_rtol=2e-4, omega_rtol=5e-7)
        for row in report["rows"]:
            freq, time = row["fdfd"], row["fdtd"]
            self.assertLess(time["fit_residual"], 1e-8)
            self.assertLess(_rel(time["gamma"], freq["gamma"]), 1e-6)
            self.assertLess(_rel(time["q_corrected"], freq["q"]), 1e-6)
            self.assertLess(_rel(time["q"], freq["q"]), 2e-4)
            self.assertGreater(freq["q"], 10.0)
        q_cold = report["rows"][0]["fdfd"]["q"]
        q_hot = report["rows"][1]["fdfd"]["q"]
        self.assertLess(q_hot, q_cold)
        dq_f = q_hot - q_cold
        dq_t = report["rows"][1]["fdtd"]["q_corrected"] - report["rows"][0]["fdtd"]["q_corrected"]
        self.assertLess(_rel(dq_t, dq_f), 1e-5)


def _assert_wavelengths(
    case: unittest.TestCase, report: dict, shift_rtol: float, omega_rtol: float
) -> None:
    for row in report["rows"]:
        freq, time = row["fdfd"], row["fdtd"]
        case.assertLess(time["fit_residual"], 1e-8)
        case.assertLess(_rel(time["omega_corrected"], freq["omega"]), omega_rtol)
    case.assertLess(_rel(report["shift_corrected_nm"], report["shift_fdfd_nm"]), 1e-6)
    case.assertLess(_rel(report["shift_fdtd_nm"], report["shift_fdfd_nm"]), shift_rtol)


def _rel(got: float, want: float) -> float:
    return abs(got - want) / abs(want)


if __name__ == "__main__":
    unittest.main()
