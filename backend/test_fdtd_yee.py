"""2D and 3D Yee FDTD checked against the spatial operators and their dispersion."""

from __future__ import annotations

import unittest

import numpy as np

from fdfd import from_yee_grid, pack_field, unpack_field, yee_operators
from fdtd.materials import EPS0, MU0
from fdtd.mesh import YeeGrid
from fdtd.yee import C_LIGHT, YeeFdtd


class ConstructionTest(unittest.TestCase):
    def test_cfl_uses_axes_longer_than_one_cell(self):
        dx, dy, dz = 2e-8, 3e-8, 5e-8
        sim = YeeFdtd(np.ones((5, 7)), (dx, dy))
        self.assertEqual(sim.ndim, 2)
        self.assertEqual(sim.shape, (5, 7, 1))
        expect = 1.0 / (C_LIGHT * np.sqrt(1.0 / dx**2 + 1.0 / dy**2))
        self.assertAlmostEqual(sim.dt_cfl, expect, places=12)
        self.assertAlmostEqual(sim.courant, sim.cfl, places=12)

        solid = YeeFdtd(np.full((4, 4, 6), 4.0), (dx, dy, dz), cfl=0.5)
        self.assertEqual(solid.ndim, 3)
        fastest = C_LIGHT / 2.0
        expect3 = 1.0 / (fastest * np.sqrt(1.0 / dx**2 + 1.0 / dy**2 + 1.0 / dz**2))
        self.assertAlmostEqual(solid.dt_cfl, expect3, places=12)
        self.assertAlmostEqual(solid.dt, 0.5 * expect3, places=12)

    def test_fastest_material_sets_the_time_step(self):
        dx = 2e-8
        eps = np.ones((6, 6))
        eps[1, 2] = 0.25
        sim = YeeFdtd(eps, dx)
        fastest = C_LIGHT / 0.5
        expect = 1.0 / (fastest * np.sqrt(2.0 / dx**2))
        self.assertAlmostEqual(sim.dt_cfl, expect, places=12)

    def test_rejects_bad_grid_and_cfl(self):
        with self.assertRaises(ValueError):
            YeeFdtd(np.ones((4,)), 1e-8)
        with self.assertRaises(ValueError):
            YeeFdtd(np.full((4, 4), -1.0), 1e-8)
        with self.assertRaises(ValueError):
            YeeFdtd(np.ones((4, 4)), 1e-8, cfl=1.2)
        with self.assertRaises(ValueError):
            YeeFdtd(np.ones((4, 4)), 1e-8, dt=0.0)


class OperatorStepTest(unittest.TestCase):
    def test_2d_periodic_step_matches_fresh_operators(self):
        self._matches((8, 6), (1.5e-8, 2.5e-8), np.full((8, 6), 2.25), "periodic", seed=1)

    def test_3d_pec_step_matches_fresh_operators(self):
        shape = (5, 4, 6)
        self._matches(shape, (2e-8, 1.5e-8, 2.5e-8), np.full(shape, 1.7), "pec", seed=2)

    def test_mixed_bc_step_matches_fresh_operators(self):
        self._matches((6, 5), 2e-8, np.ones((6, 5)), ("pec", "periodic"), seed=3)

    def _matches(self, shape, spacing, eps, bc, seed):
        rng = np.random.default_rng(seed)
        sim = YeeFdtd(eps, spacing, bc=bc, cfl=0.8)
        parts_e = [rng.normal(size=sim.shape) for _ in range(3)]
        parts_h = [rng.normal(size=sim.shape) for _ in range(3)]
        sim.set_state(*parts_e, *parts_h)
        ops = yee_operators(shape, spacing, bc=bc)
        e = pack_field(*parts_e)
        h = pack_field(*parts_h) - (sim.dt / MU0) * (ops.Ce @ e)
        tiled = np.asarray(eps, dtype=float)
        if tiled.ndim == 2:
            tiled = tiled[:, :, None]
        flat = np.broadcast_to(tiled, sim.shape).ravel()
        eps_vec = np.concatenate((flat, flat, flat))
        e_new = e + (sim.dt / EPS0) * ((ops.Ch @ h) / eps_vec)
        sim.step()
        np.testing.assert_allclose(sim.pack_h(), h, atol=1e-12)
        np.testing.assert_allclose(sim.pack_e(), e_new, atol=1e-12)
        self.assertEqual(sim.t_index, 1)
        self.assertAlmostEqual(sim.time, sim.dt, places=18)

    def test_2d_tm_step_matches_periodic_rolls(self):
        nx, ny = 9, 7
        dx, dy = 2e-8, 3e-8
        rng = np.random.default_rng(4)
        ez = rng.normal(size=(nx, ny))
        sim = YeeFdtd(np.ones((nx, ny)), (dx, dy), bc="periodic", cfl=0.7)
        sim.set_electric(ez=ez)
        sim.step()
        hx = -(sim.dt / MU0) * (np.roll(ez, -1, axis=1) - ez) / dy
        hy = (sim.dt / MU0) * (np.roll(ez, -1, axis=0) - ez) / dx
        np.testing.assert_allclose(sim.Hx[:, :, 0], hx, atol=1e-12)
        np.testing.assert_allclose(sim.Hy[:, :, 0], hy, atol=1e-12)
        curl_z = (hy - np.roll(hy, 1, axis=0)) / dx - (hx - np.roll(hx, 1, axis=1)) / dy
        # H was zero at the half-step before this update, so E moves with the new H.
        np.testing.assert_allclose(sim.Ez[:, :, 0], ez + (sim.dt / EPS0) * curl_z, atol=1e-12)
        np.testing.assert_allclose(sim.Hz, 0.0, atol=1e-12)

    def test_current_is_added_on_ez(self):
        sim = YeeFdtd(np.full((5, 4), 2.0), 2e-8, bc="pec", cfl=0.5)
        jz = np.zeros(sim.shape)
        jz[2, 1, 0] = 3.0
        sim.step(jz=jz)
        np.testing.assert_allclose(sim.Ex, 0.0, atol=0.0)
        np.testing.assert_allclose(sim.Ey, 0.0, atol=0.0)
        np.testing.assert_allclose(sim.Hx, 0.0, atol=0.0)
        expected = np.zeros(sim.shape)
        expected[2, 1, 0] = -sim.dt * 3.0 / (EPS0 * 2.0)
        np.testing.assert_allclose(sim.Ez, expected, atol=1e-20)

    def test_from_yee_grid_uses_mesh_spacing(self):
        grid = YeeGrid(
            x=np.linspace(0, 1, 5),
            y=np.linspace(0, 1, 6),
            dx=0.2e-6,
            dy=0.25e-6,
            eps_r=np.full((5, 6), 2.25),
            npml=2,
            n_max=1.5,
            conformal=False,
        )
        sim = YeeFdtd.from_yee_grid(grid, bc="pec", cfl=0.9)
        ops = from_yee_grid(grid, bc="pec")
        rng = np.random.default_rng(5)
        sim.set_state(*(rng.normal(size=sim.shape) for _ in range(6)))
        probe = rng.normal(size=ops.n_dof)
        np.testing.assert_allclose(sim.operators.Ce @ probe, ops.Ce @ probe, atol=1e-12)
        sim.run(4)
        self.assertTrue(np.isfinite(sim.pack_e()).all())
        self.assertEqual(sim.ndim, 2)


class DispersionTest(unittest.TestCase):
    def test_2d_resolved_wave_tracks_cosine_and_c(self):
        nx, ny = 32, 8
        dx = dy = 50e-9
        mx = 1
        sim, e0, omega = _prepare_mode((nx, ny), (dx, dy), 1.0, mx, 0, cfl=0.9)
        k = 2.0 * np.pi * mx / (nx * dx)
        cells = nx / mx
        self.assertGreaterEqual(cells, 20.0)
        self.assertLess(abs(omega / (k * C_LIGHT) - 1.0), 0.01)
        _assert_cosine(self, sim, e0, omega, steps=36)

    def test_3d_unequal_cells_track_discrete_dispersion(self):
        shape = (18, 12, 8)
        spacing = (25e-9, 40e-9, 30e-9)
        sim, e0, omega = _prepare_mode(shape, spacing, 1.0, 1, 1, cfl=0.85)
        self.assertEqual(sim.ndim, 3)
        _assert_cosine(self, sim, e0, omega, steps=24)

    def test_dielectric_index_scales_the_argument_of_dispersion(self):
        shape = (16, 10)
        spacing = (40e-9, 55e-9)
        n = 1.5
        sim, e0, omega = _prepare_mode(shape, spacing, n**2, 1, 0, cfl=0.8)
        _assert_cosine(self, sim, e0, omega, steps=20)
        vacuum, _, omega_vac = _prepare_mode(shape, spacing, 1.0, 1, 0, cfl=0.8, dt=sim.dt)
        arg = np.sin(omega * sim.dt / 2.0)
        arg_vac = np.sin(omega_vac * vacuum.dt / 2.0)
        self.assertAlmostEqual(arg * n, arg_vac, places=10)


class ConstraintTest(unittest.TestCase):
    def test_divergence_of_h_and_e_is_invariant(self):
        for shape, spacing in (((10, 8), 2e-8), ((6, 5, 4), (2e-8, 2.5e-8, 3e-8))):
            sim = YeeFdtd(np.ones(shape), spacing, bc="periodic", cfl=0.9)
            rng = np.random.default_rng(6)
            ex, ey, ez = (rng.normal(size=sim.shape) for _ in range(3))
            e = pack_field(ex, ey, ez)
            hx, hy, hz = unpack_field(sim.operators.Ce @ e, sim.shape)
            sim.set_state(ex, ey, ez, hx, hy, hz)
            div_h0 = sim.operators.div_h() @ sim.pack_h()
            div_e0 = sim.operators.div_e() @ sim.pack_e()
            self.assertLess(_div_rel(sim, sim.pack_h(), div_h0), 1e-8)
            sim.run(12)
            self.assertLess(_div_rel(sim, sim.pack_h(), sim.operators.div_h() @ sim.pack_h()), 1e-8)
            self.assertLess(_div_rel(sim, sim.pack_e(), sim.operators.div_e() @ sim.pack_e() - div_e0), 1e-8)

    def test_2d_tm_fields_stay_dark(self):
        nx, ny = 11, 9
        rng = np.random.default_rng(7)
        ez = rng.normal(size=(nx, ny))
        sim = YeeFdtd(np.full((nx, ny), 1.4), (2e-8, 3e-8), bc="periodic", cfl=0.9)
        sim.set_electric(ez=ez, cosine_peak=True)
        sim.run(15)
        self.assertLess(np.max(np.abs(sim.Ex)), 1e-12)
        self.assertLess(np.max(np.abs(sim.Ey)), 1e-12)
        self.assertLess(np.max(np.abs(sim.Hz)), 1e-12)

    def test_3d_z_invariant_tm_fields_stay_dark(self):
        nx, ny, nz = 8, 7, 5
        rng = np.random.default_rng(8)
        ez = np.broadcast_to(rng.normal(size=(nx, ny))[:, :, None], (nx, ny, nz)).copy()
        sim = YeeFdtd(np.ones((nx, ny, nz)), 2e-8, bc="periodic", cfl=0.9)
        sim.set_electric(ez=ez, cosine_peak=True)
        sim.run(12)
        self.assertLess(np.max(np.abs(sim.Ex)), 1e-12)
        self.assertLess(np.max(np.abs(sim.Ey)), 1e-12)
        self.assertLess(np.max(np.abs(sim.Hz)), 1e-12)

    def test_cosine_peak_gaussian_does_not_grow(self):
        for shape in ((18, 14), (12, 10, 8)):
            sim = YeeFdtd(np.ones(shape), 20e-9, bc="periodic", cfl=0.9)
            sim.set_electric(ez=_gaussian(sim.shape), cosine_peak=True)
            e0 = np.linalg.norm(sim.pack_e())
            moved = False
            for _ in range(20):
                sim.step()
                norm = np.linalg.norm(sim.pack_e())
                self.assertTrue(np.isfinite(norm))
                self.assertLessEqual(norm, e0 * (1.0 + 1e-5))
                moved = moved or norm < 0.95 * e0
            self.assertTrue(moved)
            self.assertGreater(sim.field_energy(), 0.0)


class StabilityTest(unittest.TestCase):
    def test_sub_cfl_nyquist_stays_bounded(self):
        sim = _nyquist(2, cfl=0.99)
        amp0 = np.max(np.abs(sim.pack_e()))
        sim.run(60)
        self.assertTrue(np.isfinite(sim.pack_e()).all())
        self.assertLess(np.max(np.abs(sim.pack_e())), 1.05 * amp0)

        solid = _nyquist(3, cfl=0.99)
        amp0 = np.max(np.abs(solid.pack_e()))
        solid.run(40)
        self.assertTrue(np.isfinite(solid.pack_e()).all())
        self.assertLess(np.max(np.abs(solid.pack_e())), 1.05 * amp0)

    def test_super_cfl_nyquist_grows(self):
        for ndim in (2, 3):
            ref = _nyquist(ndim, cfl=0.5)
            sim = _nyquist(ndim, dt=1.08 * ref.dt_cfl)
            amp0 = np.max(np.abs(sim.pack_e()))
            sim.run(12)
            amp = np.max(np.abs(sim.pack_e()))
            self.assertTrue(np.isfinite(amp))
            self.assertGreater(amp, 10.0 * amp0)


def _prepare_mode(shape, spacing, eps_r, mx, my, *, cfl, dt=None):
    sim = YeeFdtd(np.full(shape, eps_r), spacing, bc="periodic", cfl=cfl, dt=dt)
    nx, ny, nz = sim.shape
    ix = np.arange(nx)[:, None, None]
    iy = np.arange(ny)[None, :, None]
    ez = np.cos(2.0 * np.pi * mx * ix / nx) * np.cos(2.0 * np.pi * my * iy / ny)
    ez = np.broadcast_to(ez, sim.shape).copy()
    sim.set_electric(ez=ez, cosine_peak=True)
    e0 = sim.pack_e().copy()
    curl = sim.operators.Ch @ (sim.operators.Ce @ e0)
    lam = float(np.real(np.vdot(e0, curl) / np.vdot(e0, e0)))
    residual = np.linalg.norm(curl - lam * e0) / np.linalg.norm(curl)
    if residual > 1e-8:
        raise AssertionError(f"mode is not a curl-curl eigenvector (residual {residual})")
    dx, dy, dz = sim.spacing
    kx = 2.0 * np.pi * mx / (nx * dx)
    ky = 2.0 * np.pi * my / (ny * dy)
    lam_ref = (2.0 * np.sin(kx * dx / 2.0) / dx) ** 2 + (2.0 * np.sin(ky * dy / 2.0) / dy) ** 2
    if abs(lam - lam_ref) > 1e-6 * lam_ref:
        raise AssertionError(f"eigenvalue {lam} != Yee {lam_ref}")
    speed = C_LIGHT / np.sqrt(eps_r)
    arg = 0.5 * sim.dt * speed * np.sqrt(lam)
    if arg > 1.0:
        raise AssertionError("mode is outside the CFL limit")
    omega = (2.0 / sim.dt) * np.arcsin(arg)
    return sim, e0, omega


def _assert_cosine(case: unittest.TestCase, sim: YeeFdtd, e0: np.ndarray, omega: float, steps: int):
    for n in range(1, steps + 1):
        sim.step()
        want = np.cos(omega * n * sim.dt) * e0
        np.testing.assert_allclose(sim.pack_e(), want, rtol=1e-8, atol=1e-8)
    case.assertEqual(sim.t_index, steps)


def _div_rel(sim: YeeFdtd, field: np.ndarray, div: np.ndarray) -> float:
    """‖∇·F‖ / (‖F‖ / Δ), so a roundoff divergence is ~1e-15 rather than O(1/Δ²)."""
    wave = max(1.0 / delta for delta in sim.spacing)
    return float(np.linalg.norm(div) / (np.linalg.norm(field) * wave + 1e-30))


def _gaussian(shape: tuple[int, int, int]) -> np.ndarray:
    axes = [np.arange(n) for n in shape]
    grids = np.meshgrid(*axes, indexing="ij")
    acc = np.zeros(shape, dtype=float)
    for grid, n in zip(grids, shape):
        if n == 1:
            continue
        acc += ((grid - 0.5 * (n - 1)) / 1.6) ** 2
    return np.exp(-acc)


def _nyquist(ndim: int, *, cfl: float = 0.99, dt: float | None = None) -> YeeFdtd:
    dx = 20e-9
    if ndim == 2:
        nx = ny = 12
        sim = YeeFdtd(np.ones((nx, ny)), dx, bc="periodic", cfl=cfl, dt=dt)
        ix = np.arange(nx)[:, None]
        iy = np.arange(ny)[None, :]
        sim.set_electric(ez=(-1.0) ** (ix + iy), cosine_peak=True)
        return sim
    n = 8
    sim = YeeFdtd(np.ones((n, n, n)), dx, bc="periodic", cfl=cfl, dt=dt)
    ix = np.arange(n)[:, None, None]
    iy = np.arange(n)[None, :, None]
    iz = np.arange(n)[None, None, :]
    s = (-1.0) ** (ix + iy + iz)
    sim.set_electric(ex=s, ey=-s, cosine_peak=True)
    lam_max = 4.0 * (3.0 / dx**2)
    e0 = sim.pack_e()
    curl = sim.operators.Ch @ (sim.operators.Ce @ e0)
    lam = float(np.real(np.vdot(e0, curl) / np.vdot(e0, e0)))
    if abs(lam - lam_max) / lam_max > 1e-8:
        raise AssertionError(f"3D Nyquist eigenvalue {lam} != {lam_max}")
    return sim


if __name__ == "__main__":
    unittest.main()
