"""Yee derivative matrices with stretched-coordinate PML, plus FDTD curls.
Grid arrays have shape ``(Nx, Ny)`` and are flattened C-order (``x`` slow, ``y`` fast),
matching ceviche. Boundaries are periodic; a Bloch phase is the phase jump across that axis.
PML enters as ``D <- diag(1/s) D`` (stretched coordinates).
"""

from __future__ import annotations
import numpy as np
from scipy import sparse
from ceviche.constants import EPSILON_0, ETA_0

def compute_derivative_matrices(omega, shape, npml, dL, bloch_x=0.0, bloch_y=0.0):
    """``(Dxf, Dxb, Dyf, Dyb)`` including PML. ``npml`` is ``(Nx_pml, Ny_pml)``."""
    nx, ny = (int(shape[0]), int(shape[1]))
    dx, dy = _spacing2(dL)
    npml_x, npml_y = _npml2(npml)

    dxf = _kron_deriv(nx, ny, 0, dx, "f", bloch_x)
    dxb = _kron_deriv(nx, ny, 0, dx, "b", bloch_x)
    dyf = _kron_deriv(nx, ny, 1, dy, "f", bloch_y)
    dyb = _kron_deriv(nx, ny, 1, dy, "b", bloch_y)

    sx_f, sx_b, sy_f, sy_b = _s_matrices(omega, (nx, ny), (npml_x, npml_y), (dx, dy))
    return sx_f @ dxf, sx_b @ dxb, sy_f @ dyf, sy_b @ dyb

def curl_e(axis: int, ex, ey, ez, dL):
    """Forward-difference ``(∇ × E)`` component. Arrays are ``(Nx, Ny, Nz)``."""
    dx, dy, dz = _spacing3(dL)
    if axis == 0:
        return (np.roll(ez, -1, axis=1) - ez) / dy - (np.roll(ey, -1, axis=2) - ey) / dz
    if axis == 1:
        return (np.roll(ex, -1, axis=2) - ex) / dz - (np.roll(ez, -1, axis=0) - ez) / dx
    if axis == 2:
        return (np.roll(ey, -1, axis=0) - ey) / dx - (np.roll(ex, -1, axis=1) - ex) / dy
    raise ValueError("axis must be 0, 1, or 2")

def curl_h(axis: int, hx, hy, hz, dL):
    """Backward-difference ``(∇ × H)`` component. Arrays are ``(Nx, Ny, Nz)``."""
    dx, dy, dz = _spacing3(dL)
    if axis == 0:
        return (hz - np.roll(hz, 1, axis=1)) / dy - (hy - np.roll(hy, 1, axis=2)) / dz
    if axis == 1:
        return (hx - np.roll(hx, 1, axis=2)) / dz - (hz - np.roll(hz, 1, axis=0)) / dx
    if axis == 2:
        return (hy - np.roll(hy, 1, axis=0)) / dx - (hx - np.roll(hx, 1, axis=1)) / dy
    raise ValueError("axis must be 0, 1, or 2")

def s_factor_1d(kind: str, omega: float, spacing: float, n: int, n_pml: int) -> np.ndarray:
    """Complex stretch ``s`` along one axis. Ones when ``n_pml == 0``."""
    if n_pml <= 0:
        return np.ones(n, dtype=np.complex128)
    dw = n_pml * spacing
    idx = np.arange(n)
    out = np.ones(n, dtype=np.complex128)
    if kind == "f":
        left = idx <= n_pml
        right = idx > n - n_pml
        dist_left = spacing * (n_pml - idx + 0.5)
        dist_right = spacing * (idx - (n - n_pml) - 0.5)
    elif kind == "b":
        left = idx <= n_pml
        right = idx > n - n_pml
        dist_left = spacing * (n_pml - idx + 1.0)
        dist_right = spacing * (idx - (n - n_pml) - 1.0)
    else:
        raise ValueError("kind must be 'f' or 'b'")
    # A cell that sits in both sponges keeps the left-hand value.
    out[left] = _s_value(dist_left[left], dw, omega)
    right_only = right & ~left
    out[right_only] = _s_value(np.maximum(dist_right[right_only], 0.0), dw, omega)
    return out

def _s_value(distance, thickness, omega, m=3, ln_r=-30.0):
    sigma_max = -(m + 1) * ln_r / (2.0 * ETA_0 * thickness)
    sigma = sigma_max * (distance / thickness) ** m
    return 1.0 - 1j * sigma / (omega * EPSILON_0)

def _s_matrices(omega, shape, npml, spacing):
    nx, ny = shape
    sx_f = s_factor_1d("f", omega, spacing[0], nx, npml[0])
    sx_b = s_factor_1d("b", omega, spacing[0], nx, npml[0])
    sy_f = s_factor_1d("f", omega, spacing[1], ny, npml[1])
    sy_b = s_factor_1d("b", omega, spacing[1], ny, npml[1])
    n = nx * ny
    grids = []
    for vec, along_x in ((sx_f, True), (sx_b, True), (sy_f, False), (sy_b, False)):
        if along_x:
            grid = np.broadcast_to(vec[:, None], (nx, ny))
        else:
            grid = np.broadcast_to(vec[None, :], (nx, ny))
        grids.append(sparse.diags(1.0 / np.ascontiguousarray(grid).ravel(), format="csr", shape=(n, n)))
    return tuple(grids)

def _kron_deriv(nx, ny, axis, spacing, kind, phase) -> sparse.csr_matrix:
    if axis == 0:
        return sparse.kron(
            _deriv_1d(nx, spacing, kind, phase),
            sparse.eye(ny, dtype=np.complex128, format="csr"),
            format="csr",
        )
    return sparse.kron(
        sparse.eye(nx, dtype=np.complex128, format="csr"),
        _deriv_1d(ny, spacing, kind, phase),
        format="csr",
    )

def _deriv_1d(n: int, spacing: float, kind: str, phase: float) -> sparse.csr_matrix:
    """Periodic first difference. ``n == 1`` is identically zero (no extent on that axis)."""
    inv = 1.0 / spacing
    phasor = np.exp(1j * float(phase))
    dtype = np.complex128
    idx = np.arange(n)
    if kind == "f":
        rows = [idx, idx[:-1]]
        cols = [idx, idx[:-1] + 1]
        data = [np.full(n, -inv, dtype=dtype), np.full(max(n - 1, 0), inv, dtype=dtype)]
        rows.append(np.array([n - 1]))
        cols.append(np.array([0]))
        data.append(np.array([inv * phasor], dtype=dtype))
    elif kind == "b":
        rows = [idx, idx[1:]]
        cols = [idx, idx[1:] - 1]
        data = [np.full(n, inv, dtype=dtype), np.full(max(n - 1, 0), -inv, dtype=dtype)]
        rows.append(np.array([0]))
        cols.append(np.array([n - 1]))
        data.append(np.array([-inv * np.conjugate(phasor)], dtype=dtype))
    else:
        raise ValueError("kind must be 'f' or 'b'")
    matrix = sparse.csr_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
        dtype=dtype,
    )
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    return matrix

def _spacing2(dL):
    if np.isscalar(dL):
        value = float(dL)
        if value <= 0.0:
            raise ValueError("dL must be positive")
        return value, value
    dx, dy = float(dL[0]), float(dL[1])
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError("dL must be positive")
    return dx, dy

def _spacing3(dL):
    if np.isscalar(dL):
        value = float(dL)
        return value, value, value
    vals = tuple(float(v) for v in dL)
    if len(vals) == 2:
        return vals[0], vals[1], vals[0]
    if len(vals) == 3:
        return vals
    raise ValueError("dL must be a scalar or length 2/3")

def _npml2(npml) -> tuple[int, int]:
    if isinstance(npml, (int, np.integer)):
        n = int(npml)
        return n, n
    if len(npml) == 1:
        return int(npml[0]), int(npml[0])
    return int(npml[0]), int(npml[1])