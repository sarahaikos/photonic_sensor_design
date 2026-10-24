"""Circular and notched-square minimum-feature brushes."""

from __future__ import annotations

import unittest

import numpy as np

from mfs_brush import (
    LengthScaleBrushes,
    circular_offsets,
    feature_penalty,
    length_scale_brushes,
    mfs_penalty,
    notched_square_offsets,
    soft_dilate,
)
from topology import _chain_density_gradient, helmholtz_filter, tanh_project


def _spec(radius=2, half=2, notch=1, dx_um=0.05, mfs_nm=150.0):
    return LengthScaleBrushes(
        radius_px=radius,
        half_px=half,
        notch_px=notch,
        circular=circular_offsets(radius),
        notched=notched_square_offsets(half, notch),
        mfs_nm=mfs_nm,
        dx_um=dx_um,
    )


class BrushGeometryTest(unittest.TestCase):
    def test_circle_is_a_disk_and_the_square_loses_its_corners(self):
        disk = set(circular_offsets(2))
        square = set(notched_square_offsets(2, 1))
        self.assertIn((0, 0), disk)
        self.assertIn((2, 0), disk)
        self.assertNotIn((2, 2), disk)
        self.assertIn((2, 0), square)
        self.assertNotIn((2, 2), square)
        self.assertIn((1, 1), square)
        self.assertNotEqual(disk, square)

    def test_one_pixel_radius_keeps_the_square_corners(self):
        brushes = length_scale_brushes(80.0, 0.05, 0.45)
        self.assertEqual(brushes.radius_px, 1)
        self.assertEqual(brushes.notch_px, 0)
        self.assertEqual(len(brushes.circular), 5)
        self.assertEqual(len(brushes.notched), 9)

    def test_brush_fits_inside_the_guide(self):
        brushes = length_scale_brushes(400.0, 0.05, 0.45)
        guide_px = 0.45 / 0.05
        self.assertLessEqual(2 * brushes.radius_px + 1, guide_px + 1e-6)


class BrushPenaltyTest(unittest.TestCase):
    def test_constant_density_is_a_fixed_point(self):
        rho = np.full((12, 12), 0.35)
        value = soft_dilate(rho, circular_offsets(2), beta=12.0)
        self.assertTrue(np.allclose(value, 0.35, atol=1e-10))
        terms, grad = mfs_penalty(rho, _spec(), np.ones_like(rho, dtype=bool), beta=8.0)
        self.assertLess(terms["total"], 1e-12)
        self.assertLess(float(np.max(np.abs(grad))), 1e-8)

    def test_thin_line_is_worse_than_a_union_of_disks(self):
        mask = np.ones((28, 28), dtype=bool)
        spec = _spec(radius=2, half=2, notch=1)
        legal = np.zeros((28, 28))
        for center in range(8, 20):
            for di, dj in circular_offsets(2):
                legal[14 + di, center + dj] = 1.0
        thin = np.zeros((28, 28))
        thin[12:16, 14] = 1.0
        thin_p, _ = mfs_penalty(thin, spec, mask, beta=24.0)
        legal_p, _ = mfs_penalty(legal, spec, mask, beta=24.0)
        self.assertGreater(thin_p["circular_width"], 5.0 * legal_p["circular_width"])

    def test_narrow_gap_is_worse_than_a_wide_gap(self):
        mask = np.ones((28, 32), dtype=bool)
        spec = _spec(radius=2, half=2, notch=1)

        def blocks(gap: int) -> np.ndarray:
            rho = np.zeros((28, 32))
            rho[8:20, 4:12] = 1.0
            rho[8:20, 12 + gap : 20 + gap] = 1.0
            return rho

        narrow, _ = mfs_penalty(blocks(1), spec, mask, beta=24.0)
        wide, _ = mfs_penalty(blocks(8), spec, mask, beta=24.0)
        self.assertGreater(narrow["circular_gap"], 4.0 * wide["circular_gap"])

    def test_notched_square_flags_a_sidewall_notch(self):
        mask = np.ones((32, 32), dtype=bool)
        spec = _spec(radius=2, half=2, notch=1)
        clean = np.zeros((32, 32))
        clean[8:24, 8:24] = 1.0
        jog = clean.copy()
        jog[14:17, 8:11] = 0.0
        clean_p, _ = mfs_penalty(clean, spec, mask, beta=24.0)
        jog_p, _ = mfs_penalty(jog, spec, mask, beta=24.0)
        self.assertGreater(jog_p["notched_gap"], clean_p["notched_gap"] + 1e-4)
        self.assertGreater(jog_p["notch"], clean_p["notch"])

    def test_descent_shrinks_the_width_penalty(self):
        rho = np.zeros((24, 24))
        rho[10:14, 12] = 1.0
        mask = np.ones_like(rho, dtype=bool)
        spec = _spec()
        before, grad = mfs_penalty(rho, spec, mask, beta=8.0)
        self.assertGreater(float(np.sum(grad * grad)), 0.0)
        peak = float(np.max(np.abs(grad)))
        stepped = np.clip(rho - 0.05 * grad / peak, 0.0, 1.0)
        after, _ = mfs_penalty(stepped, spec, mask, beta=8.0)
        self.assertLess(after["circular_width"], before["circular_width"])

    def test_penalty_gradient_matches_finite_differences(self):
        rng = np.random.default_rng(0)
        rho = rng.random((12, 12))
        mask = np.zeros_like(rho, dtype=bool)
        mask[2:10, 2:10] = True
        spec = _spec(radius=1, half=2, notch=1)
        beta = 6.0
        _terms, grad = mfs_penalty(rho, spec, mask, beta=beta)
        eps = 1e-6
        for i, j in ((3, 4), (0, 0), (11, 5), (6, 11), (2, 2), (9, 9)):
            up = rho.copy()
            up[i, j] += eps
            down = rho.copy()
            down[i, j] -= eps
            plus, _ = mfs_penalty(up, spec, mask, beta=beta)
            minus, _ = mfs_penalty(down, spec, mask, beta=beta)
            fd = (plus["total"] - minus["total"]) / (2.0 * eps)
            self.assertAlmostEqual(grad[i, j], fd, delta=max(1e-4, 2e-3 * abs(fd)))

    def test_opening_gradient_matches_on_its_own(self):
        rng = np.random.default_rng(1)
        rho = rng.random((10, 11))
        mask = np.ones_like(rho, dtype=bool)
        offsets = circular_offsets(1)
        beta = 5.0
        _value, grad = feature_penalty(rho, offsets, beta, mask, "width")
        eps = 1e-6
        for i, j in ((0, 0), (5, 5), (9, 10)):
            up = rho.copy()
            up[i, j] += eps
            down = rho.copy()
            down[i, j] -= eps
            plus, _ = feature_penalty(up, offsets, beta, mask, "width")
            minus, _ = feature_penalty(down, offsets, beta, mask, "width")
            fd = (plus - minus) / (2.0 * eps)
            self.assertAlmostEqual(grad[i, j], fd, delta=max(1e-4, 2e-3 * abs(fd)))


class BrushOptimizerTest(unittest.TestCase):
    def test_topology_step_includes_the_brush_gradient(self):
        import topology as topo

        dx = 0.08
        width = 0.45
        length = 1.6
        gap = 0.2
        xs, ys, rho0, design, _guides = topo._raster(dx, width, length, gap, pad=0.3)
        brushes = length_scale_brushes(150.0, dx, width)
        self.assertTrue(design.any())

        def fake_split(*_args, **_kwargs):
            field = np.ones((xs.size, ys.size), dtype=complex)
            return (0.88, 0.12), field, True

        def fake_sens(*_args, **_kwargs):
            return np.zeros_like(rho0, dtype=float)

        originals = (topo._fdfd_split, topo._desensitization_sensitivity)
        topo._fdfd_split = fake_split
        topo._desensitization_sensitivity = fake_sens
        try:
            with_brush, hist_brush, _th, _dr, _field = topo._optimize(
                rho0.copy(),
                design,
                xs,
                ys,
                dx,
                1.2,
                4.0,
                2,
                2.4,
                1.33,
                1550.0,
                width,
                gap,
                0.12,
                brushes=brushes,
            )
            without, hist_plain, _th2, _dr2, _field2 = topo._optimize(
                rho0.copy(),
                design,
                xs,
                ys,
                dx,
                1.2,
                4.0,
                2,
                2.4,
                1.33,
                1550.0,
                width,
                gap,
                0.12,
            )
        finally:
            topo._fdfd_split, topo._desensitization_sensitivity = originals

        self.assertIn("brush_width", hist_brush[0])
        self.assertIn("brush_gap", hist_brush[0])
        self.assertIn("brush_notch", hist_brush[0])
        self.assertTrue(np.isfinite(hist_brush[0]["brush"]))
        self.assertNotIn("brush", hist_plain[0])
        self.assertFalse(np.allclose(with_brush[design], without[design]))


class BrushChainTest(unittest.TestCase):
    def test_projection_chain_matches_finite_differences(self):
        rng = np.random.default_rng(2)
        rho = rng.random((14, 16))
        design = np.zeros_like(rho, dtype=bool)
        design[3:11, 3:13] = True
        r_px = 1.1
        beta = 4.0
        eta = 0.5
        morph_beta = 6.0
        spec = _spec(radius=1, half=2, notch=1)
        rho_f = helmholtz_filter(rho, r_px)
        rho_p = np.where(design, tanh_project(rho_f, beta, eta), rho)
        _terms, g_sim = mfs_penalty(rho_p, spec, design, beta=morph_beta)
        chained = _chain_density_gradient(g_sim, rho_p, rho_f, design, beta, eta, r_px, None)

        def objective(density: np.ndarray) -> float:
            filtered = helmholtz_filter(density, r_px)
            projected = np.where(design, tanh_project(filtered, beta, eta), density)
            terms, _ = mfs_penalty(projected, spec, design, beta=morph_beta)
            return terms["total"]

        eps = 1e-5
        for i, j in ((4, 5), (7, 8), (10, 12)):
            up = rho.copy()
            up[i, j] += eps
            down = rho.copy()
            down[i, j] -= eps
            fd = (objective(up) - objective(down)) / (2.0 * eps)
            self.assertAlmostEqual(chained[i, j], fd, delta=max(2e-4, 3e-2 * abs(fd)))
        self.assertTrue(np.allclose(chained[~design], 0.0))


if __name__ == "__main__":
    unittest.main()
