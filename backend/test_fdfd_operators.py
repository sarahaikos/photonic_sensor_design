"""Discrete Yee-grid derivative and curl identities for FDFD operators."""

from __future__ import annotations

import unittest

import numpy as np
from scipy import sparse

from fdfd import (
    curl_e,
    curl_h,
    deriv,
    derivs,
    div,
    from_yee_grid,
    grad,
    pack_field,
    unpack_field,
    yee_operators,
)
from fdtd.mesh import YeeGrid


def _sparse_max(a: sparse.spmatrix) -> float:
    if a.nnz == 0:
        return 0.0
    return float(np.max(np.abs(a.data)))


def _numpy_deriv(field: np.ndarray, axis: int, d: float, kind: str, bc: str, phase: float = 0.0):
    """Independent FD reference along one axis of a 3D array."""
    if kind == "forward":
        shifted = np.roll(field, -1, axis=axis)
        out = (shifted - field) / d
        last = [slice(None)] * field.ndim
        last[axis] = -1
        first = [slice(None)] * field.ndim
        first[axis] = 0
        if bc == "pec":
            out[tuple(last)] = (0.0 - field[tuple(last)]) / d
        elif abs(phase) > 0.0:
            out[tuple(last)] = (field[tuple(first)] * np.exp(1j * phase) - field[tuple(last)]) / d
        return out
    shifted = np.roll(field, 1, axis=axis)
    out = (field - shifted) / d
    last = [slice(None)] * field.ndim
    last[axis] = -1
    first = [slice(None)] * field.ndim
    first[axis] = 0
    if bc == "pec":
        out[tuple(first)] = (field[tuple(first)] - 0.0) / d
    elif abs(phase) > 0.0:
        out[tuple(first)] = (field[tuple(first)] - field[tuple(last)] * np.exp(-1j * phase)) / d
    return out


class DerivMatrixTest(unittest.TestCase):
    def test_forward_matches_numpy_pec_each_axis(self):
        shape = (4, 5, 6)
        spacing = (0.2, 0.3, 0.4)
        rng = np.random.default_rng(0)
        field = rng.normal(size=shape)
        for ax, d in enumerate(spacing):
            op = deriv(ax, shape, spacing, kind="forward", bc="pec")
            got = (op @ field.ravel()).reshape(shape)
            want = _numpy_deriv(field, ax, d, "forward", "pec")
            np.testing.assert_allclose(got, want, atol=1e-12)

    def test_backward_matches_numpy_periodic(self):
        shape = (5, 4, 3)
        spacing = (0.1, 0.2, 0.25)
        rng = np.random.default_rng(1)
        field = rng.normal(size=shape)
        for ax, d in enumerate(spacing):
            op = deriv(ax, shape, spacing, kind="backward", bc="periodic")
            got = (op @ field.ravel()).reshape(shape)
            want = _numpy_deriv(field, ax, d, "backward", "periodic")
            np.testing.assert_allclose(got, want, atol=1e-12)

    def test_backward_is_minus_forward_transpose_pec(self):
        shape = (3, 4, 5)
        spacing = (0.5, 0.4, 0.3)
        for ax in range(3):
            df = deriv(ax, shape, spacing, kind="forward", bc="pec")
            db = deriv(ax, shape, spacing, kind="backward", bc="pec")
            self.assertLess(_sparse_max(db + df.T), 1e-12)

    def test_backward_is_minus_forward_h_bloch(self):
        shape = (4, 4, 4)
        spacing = 0.2
        phases = (0.3, -0.7, 1.1)
        for ax in range(3):
            df = deriv(ax, shape, spacing, kind="forward", bc="periodic", bloch=phases)
            db = deriv(ax, shape, spacing, kind="backward", bc="periodic", bloch=phases)
            self.assertLess(_sparse_max(db + df.conj().T), 1e-12)

    def test_periodic_kills_constants(self):
        shape = (6, 5, 4)
        ones = np.ones(int(np.prod(shape)))
        for ax in range(3):
            df = deriv(ax, shape, 0.3, kind="forward", bc="periodic")
            np.testing.assert_allclose(df @ ones, 0.0, atol=1e-12)

    def test_plane_wave_forward_difference(self):
        nx, dx = 16, 0.1
        kx = 2.0 * np.pi / (nx * dx)
        x = np.arange(nx) * dx
        f = np.exp(1j * kx * x)
        # 3D grid, wave along x, uniform in y,z
        field = np.broadcast_to(f[:, None, None], (nx, 3, 2)).copy()
        op = deriv(0, field.shape, (dx, 0.2, 0.2), kind="forward", bc="periodic")
        got = (op @ field.ravel()).reshape(field.shape)
        want = (np.exp(1j * kx * dx) - 1.0) / dx * field
        np.testing.assert_allclose(got, want, atol=1e-12)

    def test_2d_dz_is_zero(self):
        dx, dy, dz = derivs((5, 6), (0.1, 0.2), kind="forward", bc="pec")
        self.assertEqual(dz.nnz, 0)
        self.assertEqual(dx.shape, (30, 30))

    def test_string_axis_and_mixed_bc(self):
        shape = (4, 5, 3)
        dx = deriv("x", shape, 0.1, kind="forward", bc=("periodic", "pec", "pec"))
        rng = np.random.default_rng(2)
        field = rng.normal(size=shape)
        got = (dx @ field.ravel()).reshape(shape)
        want = _numpy_deriv(field, 0, 0.1, "forward", "periodic")
        np.testing.assert_allclose(got, want, atol=1e-12)

    def test_bloch_on_pec_raises(self):
        with self.assertRaises(ValueError):
            deriv(0, (4, 4, 4), 0.1, bc="pec", bloch=(0.5, 0.0, 0.0))


class CurlOperatorTest(unittest.TestCase):
    def test_ch_is_ce_transpose_pec(self):
        ops = yee_operators((3, 4, 5), (0.2, 0.3, 0.4), bc="pec")
        self.assertLess(_sparse_max(ops.Ch - ops.Ce.T), 1e-12)

    def test_ch_is_ce_h_bloch(self):
        ops = yee_operators((4, 3, 3), 0.25, bc="periodic", bloch=(0.4, -0.2, 0.9))
        self.assertLess(_sparse_max(ops.Ch - ops.Ce.conj().T), 1e-12)

    def test_div_curl_e_is_zero(self):
        ops = yee_operators((4, 5, 3), (0.1, 0.12, 0.15), bc="pec")
        self.assertLess(_sparse_max(ops.div_h() @ ops.Ce), 1e-10)

    def test_div_curl_h_is_zero(self):
        ops = yee_operators((4, 5, 3), (0.1, 0.12, 0.15), bc="pec")
        self.assertLess(_sparse_max(ops.div_e() @ ops.Ch), 1e-10)

    def test_curl_grad_e_is_zero(self):
        ops = yee_operators((3, 4, 5), 0.2, bc="periodic")
        self.assertLess(_sparse_max(ops.Ce @ ops.grad_e()), 1e-10)

    def test_curl_grad_h_is_zero(self):
        ops = yee_operators((3, 4, 5), 0.2, bc="periodic")
        self.assertLess(_sparse_max(ops.Ch @ ops.grad_h()), 1e-10)

    def test_curl_matches_component_numpy(self):
        shape = (4, 5, 3)
        spacing = (0.2, 0.3, 0.4)
        rng = np.random.default_rng(3)
        ex, ey, ez = (rng.normal(size=shape) for _ in range(3))
        ce = curl_e(shape, spacing, bc="pec")
        hx, hy, hz = unpack_field(ce @ pack_field(ex, ey, ez), shape)

        def d(kind, ax, fld):
            return _numpy_deriv(fld, ax, spacing[ax], kind, "pec")

        np.testing.assert_allclose(hx, d("forward", 1, ez) - d("forward", 2, ey), atol=1e-12)
        np.testing.assert_allclose(hy, d("forward", 2, ex) - d("forward", 0, ez), atol=1e-12)
        np.testing.assert_allclose(hz, d("forward", 0, ey) - d("forward", 1, ex), atol=1e-12)

    def test_2d_ez_curl_matches_yee_engine(self):
        """(∇ × E)_x = dEz/dy, (∇ × E)_y = −dEz/dx on a 2D grid (Dz = 0)."""
        nx, ny = 6, 7
        dx, dy = 0.05, 0.08
        rng = np.random.default_rng(4)
        ez = rng.normal(size=(nx, ny))
        z = np.zeros_like(ez)
        ce = curl_e((nx, ny), (dx, dy), bc="pec")
        hx, hy, hz = unpack_field(ce @ pack_field(z, z, ez), (nx, ny))
        dez_dy = _numpy_deriv(ez[:, :, None], 1, dy, "forward", "pec")[:, :, 0]
        dez_dx = _numpy_deriv(ez[:, :, None], 0, dx, "forward", "pec")[:, :, 0]
        np.testing.assert_allclose(hx, dez_dy, atol=1e-12)
        np.testing.assert_allclose(hy, -dez_dx, atol=1e-12)
        np.testing.assert_allclose(hz, 0.0, atol=1e-12)

    def test_free_functions_match_bundle(self):
        shape = (3, 3, 3)
        spacing = 0.1
        ops = yee_operators(shape, spacing, bc="periodic")
        self.assertLess(_sparse_max(ops.Ce - curl_e(shape, spacing, bc="periodic")), 1e-12)
        self.assertLess(_sparse_max(ops.Ch - curl_h(shape, spacing, bc="periodic")), 1e-12)
        self.assertLess(_sparse_max(ops.grad_e() - grad(shape, spacing, kind="forward", bc="periodic")), 1e-12)
        self.assertLess(_sparse_max(ops.div_e() - div(shape, spacing, kind="backward", bc="periodic")), 1e-12)

    def test_curl_curl_symmetric_pec(self):
        ops = yee_operators((3, 3, 4), 0.3, bc="pec")
        cc = ops.curl_curl_e()
        self.assertLess(_sparse_max(cc - cc.T), 1e-10)

    def test_stretch_scales_derivative(self):
        shape = (3, 3, 2)
        n = int(np.prod(shape))
        sx = 2.0 * np.ones(n)
        ops = yee_operators(shape, 0.1, bc="pec", stretch=(sx, None, None))
        plain = deriv(0, shape, 0.1, kind="forward", bc="pec")
        self.assertLess(_sparse_max(ops.Dxf - 0.5 * plain), 1e-12)

    def test_pack_unpack_roundtrip(self):
        shape = (2, 3, 4)
        rng = np.random.default_rng(5)
        parts = tuple(rng.normal(size=shape) for _ in range(3))
        again = unpack_field(pack_field(*parts), shape)
        for a, b in zip(parts, again):
            np.testing.assert_array_equal(a, b)

    def test_from_yee_grid(self):
        grid = YeeGrid(
            x=np.linspace(0, 1, 5),
            y=np.linspace(0, 1, 6),
            dx=0.2e-6,
            dy=0.2e-6,
            eps_r=np.ones((5, 6)),
            npml=2,
            n_max=1.0,
            conformal=False,
        )
        ops = from_yee_grid(grid, bc="pec")
        self.assertEqual(ops.shape, (5, 6, 1))
        self.assertEqual(ops.n_dof, 3 * 5 * 6)
        self.assertEqual(ops.Dzf.nnz, 0)


if __name__ == "__main__":
    unittest.main()
