"""FDFD / FDTD checks against discrete plane waves, slab modes, and adjoints."""

from __future__ import annotations
import unittest
import numpy as np
from ceviche import C_0, EPSILON_0, ETA_0, MU_0, Q_E, fdtd, fdfd_ez, fdfd_hz, jacobian
from ceviche.constants import Q_e
from ceviche.derivatives import compute_derivative_matrices, s_factor_1d
from ceviche.modes import get_modes
from ceviche.optimizers import adam_optimize
from ceviche.sources import gaussian_source, modal_source
from ceviche.utils import omega_from_wavelength
from fdfd.operators import deriv

class ApiTest(unittest.TestCase):
    def test_constants_match_si(self):
        self.assertAlmostEqual(C_0, 1.0 / np.sqrt(EPSILON_0 * MU_0), places=6)
        self.assertAlmostEqual(ETA_0, np.sqrt(MU_0 / EPSILON_0), places=6)
        self.assertEqual(Q_E, Q_e)


class DerivativeTest(unittest.TestCase):
    def test_matches_yee_operators_without_pml(self):
        shape = (5, 7)
        dL = 0.2
        phase = (0.4, -0.25)
        dxf, dxb, dyf, dyb = compute_derivative_matrices(1.0, shape, (0, 0), dL, *phase)
        ref_f = deriv("x", shape, dL, kind="forward", bc="periodic", bloch=phase)
        ref_b = deriv("y", shape, dL, kind="backward", bc="periodic", bloch=phase)
        np.testing.assert_allclose(dxf.toarray(), ref_f.toarray(), atol=1e-12)
        np.testing.assert_allclose(dyb.toarray(), ref_b.toarray(), atol=1e-12)
        wave = np.exp(1j * phase[0] * np.arange(shape[0]) / shape[0])
        field = np.broadcast_to(wave[:, None], shape).copy()
        theta = phase[0] / shape[0]
        want = (np.exp(1j * theta) - 1.0) / dL * field
        got = (dxf @ field.ravel()).reshape(shape)
        np.testing.assert_allclose(got, want, atol=1e-12)
        self.assertEqual(dxb.shape, (35, 35))

    def test_pml_stretch_is_unity_in_the_bulk(self):
        omega = 2e15
        n, n_pml = 20, 4
        s_f = s_factor_1d("f", omega, 1e-8, n, n_pml)
        self.assertAlmostEqual(s_f[n // 2], 1.0)
        self.assertGreater(abs(s_f[0]), 1.0)
        dxf, _, _, _ = compute_derivative_matrices(omega, (n, 3), (n_pml, 0), 1e-8)
        bare, _, _, _ = compute_derivative_matrices(omega, (n, 3), (0, 0), 1e-8)
        self.assertGreater(abs(dxf[0, 0] - bare[0, 0]), 0.0)


class PlaneWaveTest(unittest.TestCase):
    def test_ez_manufactured_solution(self):
        nx, ny = 18, 14
        dL = 40e-9
        eps = np.full((nx, ny), 2.25)
        wave = _plane(nx, ny, dL, 1, 1)
        omega = omega_from_wavelength(18 * dL)
        sim = fdfd_ez(omega, dL, eps, 0)
        b = sim.system_matrix() @ wave.ravel()
        _, _, ez = sim.solve((b / (1j * omega)).reshape(wave.shape))
        np.testing.assert_allclose(ez, wave, atol=1e-8)

    def test_hz_manufactured_solution(self):
        nx, ny = 16, 12
        dL = 50e-9
        eps_r = 4.0
        eps = np.full((nx, ny), eps_r)
        wave = _plane(nx, ny, dL, 1, 0)
        omega = omega_from_wavelength(20 * dL)
        sim = fdfd_hz(omega, dL, eps, 0)
        b = sim.system_matrix() @ wave.ravel()
        _, _, hz = sim.solve((b / (1j * omega)).reshape(wave.shape))
        np.testing.assert_allclose(hz, wave, atol=1e-8)

    def test_ez_stencil_matches_analytic_laplacian(self):
        nx, ny = 20, 16
        dL = 40e-9
        eps = np.full((nx, ny), 2.25 + 0.01j)
        wave = _plane(nx, ny, dL, 1, 1)
        kx = 2.0 * np.pi / (nx * dL)
        ky = 2.0 * np.pi / (ny * dL)
        lap = -4.0 * np.sin(kx * dL / 2) ** 2 / dL**2 - 4.0 * np.sin(ky * dL / 2) ** 2 / dL**2
        omega = omega_from_wavelength(12 * dL)
        sim = fdfd_ez(omega, dL, eps, 0)
        got = (sim.system_matrix() @ wave.ravel()).reshape(wave.shape)
        want = -lap / MU_0 * wave - omega**2 * EPSILON_0 * eps * wave
        np.testing.assert_allclose(got, want, atol=1e-6 * np.max(np.abs(want)))


class AdjointTest(unittest.TestCase):
    def test_ez_grad_matches_central_difference(self):
        rng = np.random.default_rng(0)
        shape = (8, 8)
        dL = 40e-9
        omega = omega_from_wavelength(10 * dL)
        eps = rng.uniform(1.4, 2.4, shape) + 0.08j
        src = gaussian_source(shape, (3.2, 4.1), (1.4, 1.6))
        sim = fdfd_ez(omega, dL, eps, 0)
        grad = sim.grad_eps(src, _ez(omega, dL, eps, src))
        delta = 1e-4
        for i, j in ((2, 3), (5, 6), (1, 7)):
            plus = eps.copy()
            minus = eps.copy()
            plus[i, j] += delta
            minus[i, j] -= delta
            fd = (_objective_ez(omega, dL, plus, src) - _objective_ez(omega, dL, minus, src)) / (2 * delta)
            self.assertAlmostEqual(grad[i, j], fd, delta=2e-3 * max(1.0, abs(fd)))

    def test_hz_grad_matches_central_difference(self):
        rng = np.random.default_rng(1)
        shape = (7, 7)
        dL = 40e-9
        omega = omega_from_wavelength(9 * dL)
        eps = rng.uniform(1.5, 2.2, shape) + 0.05j
        src = gaussian_source(shape, (3.0, 3.0), (1.5, 1.2))
        sim = fdfd_hz(omega, dL, eps, 0)
        grad = sim.grad_eps(src, _hz(omega, dL, eps, src))
        delta = 1e-4
        for i, j in ((1, 2), (4, 5)):
            plus = eps.copy()
            minus = eps.copy()
            plus[i, j] += delta
            minus[i, j] -= delta
            fd = (_objective_hz(omega, dL, plus, src) - _objective_hz(omega, dL, minus, src)) / (2 * delta)
            self.assertAlmostEqual(grad[i, j], fd, delta=2e-3 * max(1.0, abs(fd)))


class ModeTest(unittest.TestCase):
    def test_slab_neff_near_analytic(self):
        wavelength = 1.55e-6
        dL = wavelength / 40.0
        omega = omega_from_wavelength(wavelength)
        n_core, n_clad = 3.0, 1.0
        a = 0.30e-6
        half = int(round(a / dL))
        nx = 160
        eps = np.full(nx, n_clad**2)
        mid = nx // 2
        eps[mid - half : mid + half] = n_core**2
        vals, vecs = get_modes(eps, omega, dL, npml=12, m=4)
        neff = np.sqrt(vals[0])
        analytic = _even_slab_neff(2 * np.pi / wavelength, half * dL, n_core, n_clad)
        self.assertGreater(np.real(neff), n_clad + 0.05)
        self.assertLess(np.real(neff), n_core)
        self.assertAlmostEqual(float(np.real(neff)), analytic, delta=0.03)
        self.assertAlmostEqual(np.sum(np.abs(vecs[:, 0]) ** 2), 1.0, places=10)

    def test_modal_source_propagates_in_a_straight_guide(self):
        wavelength = 1.55e-6
        dL = wavelength / 20.0
        omega = omega_from_wavelength(wavelength)
        nx, ny = 40, 48
        x = (np.arange(nx) + 0.5) * dL
        eps = np.ones((nx, ny))
        eps[np.abs(x - x.mean()) <= 0.22e-6, :] = 2.25**2
        npml = 6
        y_src = npml + 2
        src, vals = modal_source(eps, omega, dL, y_src, npml=npml, m=1)
        self.assertGreater(np.real(np.sqrt(vals[0])), 1.2)
        _, _, ez = fdfd_ez(omega, dL, eps, npml).solve(src)
        y_mon = ny - npml - 4
        corr = _abs_correlation(src[:, y_src], ez[:, y_mon])
        self.assertGreater(corr, 0.85)


class JacobianOptimizerTest(unittest.TestCase):
    def test_numerical_jacobian_of_squares(self):
        def fun(x):
            return x**2

        x0 = np.array([1.0, -2.0, 0.5])
        jac = jacobian(fun, mode="reverse", step_size=1e-6)(x0)
        np.testing.assert_allclose(jac, np.diag(2 * x0), atol=1e-5)

    def test_adam_finds_the_minimum(self):
        def objective(p):
            return np.sum((p - 3.0) ** 2)

        def grad(p):
            return 2.0 * (p - 3.0)

        params, history = adam_optimize(objective, np.zeros(4), grad, step_size=0.05, Nsteps=250, verbose=False)
        self.assertLess(np.max(np.abs(params - 3.0)), 0.08)
        self.assertLess(history[-1], history[0])


class FdtdTest(unittest.TestCase):
    def test_vacuum_pulse_stays_bounded_and_pml_absorbs(self):
        n = 36
        dL = 20e-9
        eps = np.ones((n, n))
        pulse = np.exp(-((np.arange(n) - n / 2) ** 2) / 8.0)
        ez0 = np.outer(pulse, pulse)

        open_box = fdtd(eps, dL, 0)
        open_box.Ez = ez0[:, :, None]
        start = np.sum(open_box.Ez**2)
        open_box.run(80)
        self.assertTrue(np.isfinite(open_box.Ez).all())
        self.assertLess(np.max(np.abs(open_box.Ez)), 5.0 * np.max(np.abs(ez0)))

        closed = fdtd(eps, dL, 8)
        closed.Ez = ez0[:, :, None]
        closed.run(160)
        left = np.sum(closed.Ez**2)
        self.assertLess(left, 0.35 * start)
        self.assertTrue(np.isfinite(closed.Ez).all())


def _plane(nx, ny, dL, mx, my):
    kx = 2.0 * np.pi * mx / (nx * dL)
    ky = 2.0 * np.pi * my / (ny * dL)
    ix = np.arange(nx)[:, None]
    iy = np.arange(ny)[None, :]
    return np.exp(1j * (kx * ix * dL + ky * iy * dL))


def _ez(omega, dL, eps, src):
    _, _, ez = fdfd_ez(omega, dL, eps, 0).solve(src)
    return ez


def _hz(omega, dL, eps, src):
    _, _, hz = fdfd_hz(omega, dL, eps, 0).solve(src)
    return hz


def _objective_ez(omega, dL, eps, src):
    return float(np.sum(np.abs(_ez(omega, dL, eps, src)) ** 2))


def _objective_hz(omega, dL, eps, src):
    return float(np.sum(np.abs(_hz(omega, dL, eps, src)) ** 2))


def _abs_correlation(a, b):
    a = np.abs(np.asarray(a))
    b = np.abs(np.asarray(b))
    return float(np.vdot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def _even_slab_neff(k0, a, n_core, n_clad):
    """Lowest even TMz slab mode: h tan(h a) = p."""
    vconst = k0**2 * (n_core**2 - n_clad**2)
    h_cap = min(np.sqrt(vconst) * (1.0 - 1e-9), 0.5 * np.pi / a * (1.0 - 1e-6))

    def mismatch(h):
        p = np.sqrt(max(vconst - h**2, 0.0))
        return h * np.tan(h * a) - p

    lo, hi = 1e-8, h_cap
    flo, fhi = mismatch(lo), mismatch(hi)
    if flo * fhi > 0.0:
        grid = np.linspace(lo, hi, 400)
        vals = np.array([mismatch(h) for h in grid])
        sign = np.where(vals[:-1] * vals[1:] <= 0.0)[0]
        if sign.size == 0:
            raise AssertionError("slab dispersion has no root")
        lo, hi = grid[sign[0]], grid[sign[0] + 1]
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if mismatch(lo) * mismatch(mid) <= 0.0:
            hi = mid
        else:
            lo = mid
    h = 0.5 * (lo + hi)
    return float(np.sqrt(n_core**2 - (h / k0) ** 2))

if __name__ == "__main__":
    unittest.main()