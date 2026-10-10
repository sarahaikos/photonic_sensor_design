"""Discrete differential and curl operators on a uniform Yee grid.

Staggering is encoded in the difference direction, not in shifted array shapes:
every component is stored on the same ``(nx, ny, nz)`` lattice and flattened
C-order (``array.ravel()``).

    curl E  (Ce)  — forward differences  → H on the dual edges
    curl H  (Ch)  — backward differences → E on the primal edges

Vector fields pack as ``[Fx, Fy, Fz]``, each of length ``N = nx*ny*nz``.

PEC (Dirichlet) walls set the exterior sample to zero. Periodic / Bloch wrap
with phase ``exp(j φ)`` across that axis. A 2D grid ``(nx, ny)`` is ``nz = 1``
with periodic ``z``, so ``Dz = 0``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

_AXES = ("x", "y", "z")
_BC_PEC = frozenset({"pec", "dirichlet"})
_BC_PERIODIC = frozenset({"periodic", "pbc", "bloch"})


def deriv(
    axis: int | str,
    shape: tuple[int, ...],
    spacing: float | tuple[float, ...],
    *,
    kind: str = "forward",
    bc: str | tuple[str, ...] = "pec",
    bloch: float | tuple[float, ...] | None = None,
) -> sparse.csr_matrix:
    """First-order difference along one axis. Returns an ``N×N`` CSR matrix."""
    spec = _grid_spec(shape, spacing, bc, bloch)
    ax = _axis_index(axis)
    return _axis_matrix(ax, spec, kind)


def derivs(
    shape: tuple[int, ...],
    spacing: float | tuple[float, ...],
    *,
    kind: str = "forward",
    bc: str | tuple[str, ...] = "pec",
    bloch: float | tuple[float, ...] | None = None,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
    """``(Dx, Dy, Dz)`` for one difference direction."""
    spec = _grid_spec(shape, spacing, bc, bloch)
    return tuple(_axis_matrix(ax, spec, kind) for ax in range(3))  # type: ignore[return-value]


def curl_e(
    shape: tuple[int, ...],
    spacing: float | tuple[float, ...],
    *,
    bc: str | tuple[str, ...] = "pec",
    bloch: float | tuple[float, ...] | None = None,
    stretch: tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None] | None = None,
) -> sparse.csr_matrix:
    """Discrete ``∇ × E`` (forward). Maps packed E → packed H, size ``3N×3N``."""
    dx, dy, dz = derivs(shape, spacing, kind="forward", bc=bc, bloch=bloch)
    dx, dy, dz = _stretch((dx, dy, dz), stretch)
    return _block_curl(dx, dy, dz)


def curl_h(
    shape: tuple[int, ...],
    spacing: float | tuple[float, ...],
    *,
    bc: str | tuple[str, ...] = "pec",
    bloch: float | tuple[float, ...] | None = None,
    stretch: tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None] | None = None,
) -> sparse.csr_matrix:
    """Discrete ``∇ × H`` (backward). Maps packed H → packed E, size ``3N×3N``."""
    dx, dy, dz = derivs(shape, spacing, kind="backward", bc=bc, bloch=bloch)
    dx, dy, dz = _stretch((dx, dy, dz), stretch)
    return _block_curl(dx, dy, dz)


def grad(
    shape: tuple[int, ...],
    spacing: float | tuple[float, ...],
    *,
    kind: str = "forward",
    bc: str | tuple[str, ...] = "pec",
    bloch: float | tuple[float, ...] | None = None,
) -> sparse.csr_matrix:
    """Stacked ``[Dx; Dy; Dz]``. Maps a scalar (length N) to a packed vector (3N)."""
    dx, dy, dz = derivs(shape, spacing, kind=kind, bc=bc, bloch=bloch)
    return sparse.vstack([dx, dy, dz], format="csr")


def div(
    shape: tuple[int, ...],
    spacing: float | tuple[float, ...],
    *,
    kind: str = "backward",
    bc: str | tuple[str, ...] = "pec",
    bloch: float | tuple[float, ...] | None = None,
) -> sparse.csr_matrix:
    """Row ``[Dx, Dy, Dz]``. Maps a packed vector (3N) to a scalar (N)."""
    dx, dy, dz = derivs(shape, spacing, kind=kind, bc=bc, bloch=bloch)
    return sparse.hstack([dx, dy, dz], format="csr")


def pack_field(fx: np.ndarray, fy: np.ndarray, fz: np.ndarray) -> np.ndarray:
    """Stack Yee components into a single vector, C-order within each component."""
    return np.concatenate([np.asarray(fx).ravel(), np.asarray(fy).ravel(), np.asarray(fz).ravel()])


def unpack_field(vec: np.ndarray, shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inverse of ``pack_field``. ``shape`` is ``(nx, ny)`` or ``(nx, ny, nz)``."""
    nx, ny, nz = _shape3(shape)
    n = nx * ny * nz
    v = np.asarray(vec)
    if v.size != 3 * n:
        raise ValueError(f"expected length {3 * n}, got {v.size}")
    shp = (nx, ny) if len(shape) == 2 else (nx, ny, nz)
    return tuple(v[i * n : (i + 1) * n].reshape(shp) for i in range(3))  # type: ignore[return-value]


@dataclass
class YeeOperators:
    """All first-order Yee operators for one grid."""

    shape: tuple[int, int, int]
    spacing: tuple[float, float, float]
    bc: tuple[str, str, str]
    bloch: tuple[float, float, float]
    Dxf: sparse.csr_matrix
    Dyf: sparse.csr_matrix
    Dzf: sparse.csr_matrix
    Dxb: sparse.csr_matrix
    Dyb: sparse.csr_matrix
    Dzb: sparse.csr_matrix
    Ce: sparse.csr_matrix
    Ch: sparse.csr_matrix

    @property
    def n(self) -> int:
        return int(np.prod(self.shape))

    @property
    def n_dof(self) -> int:
        return 3 * self.n

    def __repr__(self) -> str:
        return (
            f"YeeOperators(shape={self.shape}, spacing={self.spacing}, "
            f"bc={self.bc}, bloch={self.bloch}, n_dof={self.n_dof})"
        )

    def grad_e(self) -> sparse.csr_matrix:
        return sparse.vstack([self.Dxf, self.Dyf, self.Dzf], format="csr")

    def grad_h(self) -> sparse.csr_matrix:
        return sparse.vstack([self.Dxb, self.Dyb, self.Dzb], format="csr")

    def div_e(self) -> sparse.csr_matrix:
        return sparse.hstack([self.Dxb, self.Dyb, self.Dzb], format="csr")

    def div_h(self) -> sparse.csr_matrix:
        return sparse.hstack([self.Dxf, self.Dyf, self.Dzf], format="csr")

    def curl_curl_e(self) -> sparse.csr_matrix:
        """``∇ × ∇ ×`` on E: ``Ch @ Ce``. Explicit product; use on modest grids."""
        return self.Ch @ self.Ce

    def curl_curl_h(self) -> sparse.csr_matrix:
        return self.Ce @ self.Ch


def yee_operators(
    shape: tuple[int, ...],
    spacing: float | tuple[float, ...],
    *,
    bc: str | tuple[str, ...] = "pec",
    bloch: float | tuple[float, ...] | None = None,
    stretch: tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None] | None = None,
) -> YeeOperators:
    """Build forward/backward derivatives and both discrete curls."""
    spec = _grid_spec(shape, spacing, bc, bloch)
    d_f = tuple(_axis_matrix(ax, spec, "forward") for ax in range(3))
    d_b = tuple(_axis_matrix(ax, spec, "backward") for ax in range(3))
    d_f = _stretch(d_f, stretch)
    d_b = _stretch(d_b, stretch)
    return YeeOperators(
        shape=spec.shape,
        spacing=spec.spacing,
        bc=spec.bc,
        bloch=spec.bloch,
        Dxf=d_f[0],
        Dyf=d_f[1],
        Dzf=d_f[2],
        Dxb=d_b[0],
        Dyb=d_b[1],
        Dzb=d_b[2],
        Ce=_block_curl(*d_f),
        Ch=_block_curl(*d_b),
    )


def from_yee_grid(grid, *, bc: str | tuple[str, ...] = "pec") -> YeeOperators:
    """2D operators from an ``fdtd.mesh.YeeGrid`` (``nz = 1``, periodic z)."""
    return yee_operators((grid.nx, grid.ny), (grid.dx, grid.dy), bc=bc)


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _GridSpec:
    shape: tuple[int, int, int]
    spacing: tuple[float, float, float]
    bc: tuple[str, str, str]
    bloch: tuple[float, float, float]


def _shape3(shape: tuple[int, ...]) -> tuple[int, int, int]:
    if len(shape) == 2:
        nx, ny = (int(shape[0]), int(shape[1]))
        return nx, ny, 1
    if len(shape) == 3:
        return int(shape[0]), int(shape[1]), int(shape[2])
    raise ValueError("shape must be (nx, ny) or (nx, ny, nz)")


def _normalize_bc(bc: str) -> str:
    name = str(bc).lower()
    if name in _BC_PEC:
        return "pec"
    if name in _BC_PERIODIC:
        return "periodic"
    raise ValueError(f"unknown BC {bc!r}; use 'pec' or 'periodic'")


def _grid_spec(
    shape: tuple[int, ...],
    spacing: float | tuple[float, ...],
    bc: str | tuple[str, ...],
    bloch: float | tuple[float, ...] | None,
) -> _GridSpec:
    nx, ny, nz = _shape3(shape)
    if min(nx, ny, nz) < 1:
        raise ValueError("grid dimensions must be ≥ 1")
    two_d = len(shape) == 2

    if np.isscalar(spacing):
        dl = (float(spacing), float(spacing), float(spacing))
    else:
        sp = tuple(float(v) for v in spacing)  # type: ignore[union-attr]
        if len(sp) == 2:
            dl = (sp[0], sp[1], sp[0])
        elif len(sp) == 3:
            dl = sp
        else:
            raise ValueError("spacing must be a scalar or length 2/3")
    if any(d <= 0.0 for d in dl):
        raise ValueError("spacing must be positive")

    if isinstance(bc, str):
        bcs = (_normalize_bc(bc), _normalize_bc(bc), "periodic" if two_d else _normalize_bc(bc))
    else:
        raw = tuple(_normalize_bc(b) for b in bc)
        if len(raw) == 2:
            bcs = (raw[0], raw[1], "periodic")
        elif len(raw) == 3:
            bcs = raw
        else:
            raise ValueError("bc must be a string or length 2/3")
        if two_d:
            bcs = (bcs[0], bcs[1], "periodic")

    if bloch is None:
        phases = (0.0, 0.0, 0.0)
    elif np.isscalar(bloch):
        phi = float(bloch)
        phases = (phi, phi, 0.0 if two_d else phi)
    else:
        ph = tuple(float(v) for v in bloch)  # type: ignore[union-attr]
        if len(ph) == 2:
            phases = (ph[0], ph[1], 0.0)
        elif len(ph) == 3:
            phases = ph
        else:
            raise ValueError("bloch must be a scalar phase or length 2/3")
        if two_d:
            phases = (phases[0], phases[1], 0.0)

    for ax, (b, phi) in enumerate(zip(bcs, phases)):
        if abs(phi) > 1e-15 and b != "periodic":
            raise ValueError(f"Bloch phase on {_AXES[ax]} requires periodic BC")
    return _GridSpec((nx, ny, nz), dl, bcs, phases)


def _axis_index(axis: int | str) -> int:
    if isinstance(axis, str):
        key = axis.lower()
        if key not in _AXES:
            raise ValueError(f"axis must be x/y/z, got {axis!r}")
        return _AXES.index(key)
    ax = int(axis)
    if ax not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2")
    return ax


def _deriv_1d(n: int, d: float, kind: str, bc: str, phase_rad: float) -> sparse.csr_matrix:
    """1D difference. Duplicate COO entries at the wrap are summed (n = 1 periodic)."""
    if kind not in ("forward", "backward"):
        raise ValueError("kind must be 'forward' or 'backward'")
    inv = 1.0 / d
    periodic = bc == "periodic"
    use_complex = periodic and abs(phase_rad) > 1e-15
    bloch = np.exp(1j * float(phase_rad)) if use_complex else 1.0
    dtype = np.complex128 if use_complex else np.float64
    wrap = (bloch if kind == "forward" else np.conjugate(bloch)) if periodic else 0.0

    idx = np.arange(n)
    if kind == "forward":
        rows = [idx, idx[:-1]]
        cols = [idx, idx[:-1] + 1]
        data = [np.full(n, -inv, dtype=dtype), np.full(max(n - 1, 0), inv, dtype=dtype)]
        if periodic:
            rows.append(np.array([n - 1]))
            cols.append(np.array([0]))
            data.append(np.array([inv * wrap], dtype=dtype))
    else:
        rows = [idx, idx[1:]]
        cols = [idx, idx[1:] - 1]
        data = [np.full(n, inv, dtype=dtype), np.full(max(n - 1, 0), -inv, dtype=dtype)]
        if periodic:
            rows.append(np.array([0]))
            cols.append(np.array([n - 1]))
            data.append(np.array([-inv * wrap], dtype=dtype))

    r = np.concatenate(rows)
    c = np.concatenate(cols)
    v = np.concatenate(data)
    matrix = sparse.csr_matrix((v, (r, c)), shape=(n, n), dtype=dtype)
    matrix.eliminate_zeros()
    return matrix


def _eye(n: int, like: sparse.spmatrix) -> sparse.csr_matrix:
    dtype = np.complex128 if np.issubdtype(like.dtype, np.complexfloating) else np.float64
    return sparse.eye(n, format="csr", dtype=dtype)


def _axis_matrix(axis: int, spec: _GridSpec, kind: str) -> sparse.csr_matrix:
    nx, ny, nz = spec.shape
    d1 = _deriv_1d(spec.shape[axis], spec.spacing[axis], kind, spec.bc[axis], spec.bloch[axis])
    # C-order: z fastest, x slowest.  D = I_slow ⊗ d1 ⊗ I_fast
    if axis == 0:
        return sparse.kron(d1, _eye(ny * nz, d1), format="csr")
    if axis == 1:
        return sparse.kron(_eye(nx, d1), sparse.kron(d1, _eye(nz, d1), format="csr"), format="csr")
    return sparse.kron(_eye(nx * ny, d1), d1, format="csr")


def _block_curl(
    dx: sparse.csr_matrix, dy: sparse.csr_matrix, dz: sparse.csr_matrix
) -> sparse.csr_matrix:
    """
    [  0  -Dz   Dy ]
    [ Dz   0   -Dx ]
    [-Dy   Dx   0  ]
    """
    return sparse.bmat(
        [
            [None, -dz, dy],
            [dz, None, -dx],
            [-dy, dx, None],
        ],
        format="csr",
    )


def _stretch(
    derivs_xyz: tuple[sparse.csr_matrix, ...],
    stretch: tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None] | None,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
    if stretch is None:
        return derivs_xyz  # type: ignore[return-value]
    out = []
    for d, s in zip(derivs_xyz, stretch):
        if s is None:
            out.append(d)
            continue
        arr = np.asarray(s).ravel()
        if arr.size != d.shape[0]:
            raise ValueError(f"stretch length {arr.size} != {d.shape[0]}")
        if np.any(arr == 0):
            raise ValueError("stretch s must be nonzero")
        out.append(sparse.diags(1.0 / arr) @ d)
    return tuple(out)  # type: ignore[return-value]