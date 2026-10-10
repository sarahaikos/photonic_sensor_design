"""Leapfrog FDTD on the spatial Yee grid from ``fdfd.operators``.

The grid is the same one used for the discrete curls: forward ``Ce`` updates H
and backward ``Ch`` updates E. A 2D permittivity ``(nx, ny)`` is one cell
thick, with no z-derivative. Spacing is in meters.

    H^{n+1/2} = H^{n-1/2} − (Δt/μ₀) ∇×E^n
    E^{n+1}   = E^n       + (Δt/ε)  ∇×H^{n+1/2}

``E`` is sampled at integer steps and ``H`` at half-steps. For a uniform
medium, ``H^{-1/2} = (Δt / 2μ₀) ∇×E^0`` puts every curl-curl eigenmode at a
cosine peak.

Electric conductivity uses the centered update of ``ε ∂E/∂t + σ E = ∇ × H``.
A constant loss tangent ``σ = α ε₀ εᵣ`` then decays every mode at ``e^{-α t / 2}``,
the same rate as the FDFD pole ``ω² + i α ω − ω₀² = 0``.
"""

from __future__ import annotations

import numpy as np

from fdfd.operators import YeeOperators, pack_field, unpack_field, yee_operators
from fdtd.materials import EPS0, MU0

C_LIGHT = 1.0 / float(np.sqrt(EPS0 * MU0))

class YeeFdtd:
    """2D or 3D Yee stepper. Spatial derivatives are ``YeeOperators``."""

    def __init__(
        self,
        eps_r,
        spacing: float | tuple[float, ...],
        *,
        bc: str | tuple[str, ...] = "periodic",
        cfl: float = 0.95,
        dt: float | None = None,
        sigma=0.0,
    ):
        eps = np.asarray(eps_r, dtype=np.float64)
        if eps.ndim == 2:
            eps = eps[:, :, None]
        elif eps.ndim != 3:
            raise ValueError("eps_r must have shape (nx, ny) or (nx, ny, nz)")
        if eps.size == 0 or not np.isfinite(eps).all() or np.any(eps <= 0.0):
            raise ValueError("eps_r must be finite and positive")

        nx, ny, nz = (int(eps.shape[0]), int(eps.shape[1]), int(eps.shape[2]))
        if nz == 1:
            spacing_ops: float | tuple[float, ...] = _spacing_2d(spacing)
            self._ops = yee_operators((nx, ny), spacing_ops, bc=bc)
        else:
            self._ops = yee_operators((nx, ny, nz), spacing, bc=bc)
        if self._ops.shape != (nx, ny, nz):
            raise ValueError(f"operator shape {self._ops.shape} != {(nx, ny, nz)}")

        self.eps_r = np.ascontiguousarray(eps)
        self.bc = self._ops.bc
        self.spacing = self._ops.spacing
        self.dt_cfl = _cfl_dt(self._ops.shape, self._ops.spacing, float(np.min(eps)))
        if dt is None:
            if not np.isfinite(cfl) or cfl <= 0.0 or cfl > 1.0:
                raise ValueError("cfl must be in (0, 1]")
            self.dt = float(cfl) * self.dt_cfl
            self.cfl = float(cfl)
        else:
            if not np.isfinite(dt) or dt <= 0.0:
                raise ValueError("dt must be positive")
            self.dt = float(dt)
            self.cfl = self.dt / self.dt_cfl
        z = np.zeros(self.shape, dtype=np.float64)
        self.Ex = z.copy()
        self.Ey = z.copy()
        self.Ez = z.copy()
        self.Hx = z.copy()
        self.Hy = z.copy()
        self.Hz = z.copy()
        self.t_index = 0
        self.sigma = self._prepare_sigma(sigma)

    def __repr__(self) -> str:
        return (
            f"YeeFdtd(shape={self.shape}, spacing={self.spacing}, "
            f"bc={self.bc}, dt={self.dt:.6e})"
        )

    @classmethod
    def from_yee_grid(
        cls,
        grid,
        *,
        bc: str | tuple[str, ...] = "pec",
        cfl: float = 0.95,
        dt: float | None = None,
    ):
        """Build a stepper from an ``fdtd.mesh.YeeGrid`` (dx, dy in meters)."""
        eps = np.asarray(grid.eps_r)
        if eps.ndim == 3:
            spacing = (float(grid.dx), float(grid.dy), float(grid.dz))
        else:
            spacing = (float(grid.dx), float(grid.dy))
        return cls(eps, spacing, bc=bc, cfl=cfl, dt=dt)

    @property
    def operators(self) -> YeeOperators:
        return self._ops

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._ops.shape

    @property
    def ndim(self) -> int:
        return 2 if self.shape[2] == 1 else 3

    @property
    def time(self) -> float:
        """Time of ``E``. ``H`` is at ``time - dt/2``."""
        return self.t_index * self.dt

    @property
    def courant(self) -> float:
        return self.dt / self.dt_cfl

    def pack_e(self) -> np.ndarray:
        return pack_field(self.Ex, self.Ey, self.Ez)

    def pack_h(self) -> np.ndarray:
        return pack_field(self.Hx, self.Hy, self.Hz)

    def set_state(self, ex=0.0, ey=0.0, ez=0.0, hx=0.0, hy=0.0, hz=0.0) -> None:
        """Replace ``E`` and ``H^{-1/2}`` and reset the clock."""
        self.Ex = self._coerce(ex)
        self.Ey = self._coerce(ey)
        self.Ez = self._coerce(ez)
        self.Hx = self._coerce(hx)
        self.Hy = self._coerce(hy)
        self.Hz = self._coerce(hz)
        self.t_index = 0

    def set_electric(self, ex=0.0, ey=0.0, ez=0.0, *, cosine_peak: bool = False) -> None:
        """Load ``E^0`` and reset the clock.

        ``cosine_peak=True`` sets ``H^{-1/2} = (Δt / 2μ₀) ∇×E`` so each
        curl-curl eigenmode of a uniform medium starts at a cosine peak.
        """
        self.set_state(ex, ey, ez, 0.0, 0.0, 0.0)
        if cosine_peak:
            half = (self.dt / (2.0 * MU0)) * (self._ops.Ce @ self.pack_e())
            self.Hx, self.Hy, self.Hz = _as_fields(unpack_field(half, self.shape))

    def step(self, jz=None) -> None:
        """Advance one leapfrog step. ``jz`` is an electric current density on ``Ez``."""
        e = self.pack_e()
        h = self.pack_h() - (self.dt / MU0) * (self._ops.Ce @ e)
        self.Hx, self.Hy, self.Hz = _as_fields(unpack_field(h, self.shape))
        curl = (self._ops.Ch @ h) / self._permittivity_vector()
        advance = (self.dt / EPS0) * curl
        if self.sigma is None:
            e = e + advance
        else:
            beta = self._sigma_vector() * self.dt / (2.0 * EPS0 * self._permittivity_vector())
            e = ((1.0 - beta) * e + advance) / (1.0 + beta)
        self.Ex, self.Ey, self.Ez = _as_fields(unpack_field(e, self.shape))
        if jz is not None:
            current = self.dt * self._coerce(jz) / (EPS0 * self.eps_r)
            if self.sigma is not None:
                beta_z = self.sigma * self.dt / (2.0 * EPS0 * self.eps_r)
                current = current / (1.0 + beta_z)
            self.Ez = self.Ez - current
        self.t_index += 1

    def run(self, steps: int, jz=None) -> None:
        n = int(steps)
        if n < 0:
            raise ValueError("steps must be ≥ 0")
        for _ in range(n):
            self.step(jz=jz)

    def field_energy(self) -> float:
        """Discrete field energy. Axes of length 1 are left out of the cell volume."""
        volume = 1.0
        for length, delta in zip(self.shape, self.spacing):
            if length > 1:
                volume *= float(delta)
        electric = self.eps_r * (self.Ex**2 + self.Ey**2 + self.Ez**2)
        magnetic = self.Hx**2 + self.Hy**2 + self.Hz**2
        return 0.5 * volume * float(np.sum(EPS0 * electric + MU0 * magnetic))

    def _prepare_sigma(self, sigma) -> np.ndarray | None:
        """Cell-wise conductivity (S/m), or ``None`` when the grid is lossless."""
        if sigma is None:
            return None
        arr = np.asarray(sigma, dtype=np.float64)
        if arr.shape == ():
            if float(arr) == 0.0:
                return None
            arr = np.full(self.shape, float(arr), dtype=np.float64)
        else:
            arr = self._coerce(arr)
        if not np.isfinite(arr).all() or np.any(arr < 0.0):
            raise ValueError("sigma must be finite and ≥ 0")
        if not np.any(arr > 0.0):
            return None
        return np.ascontiguousarray(arr)

    def _sigma_vector(self) -> np.ndarray:
        flat = np.asarray(self.sigma, dtype=np.float64).ravel()
        if flat.size != self._ops.n:
            raise ValueError(f"sigma has {flat.size} cells, expected {self._ops.n}")
        return np.concatenate((flat, flat, flat))

    def _permittivity_vector(self) -> np.ndarray:
        flat = np.asarray(self.eps_r, dtype=np.float64).ravel()
        if flat.size != self._ops.n:
            raise ValueError(f"eps_r has {flat.size} cells, expected {self._ops.n}")
        return np.concatenate((flat, flat, flat))

    def _coerce(self, field) -> np.ndarray:
        arr = np.asarray(field, dtype=np.float64)
        if arr.shape == ():
            return np.full(self.shape, float(arr), dtype=np.float64)
        if arr.ndim == 2 and self.shape[2] == 1 and arr.shape == self.shape[:2]:
            arr = arr[:, :, None]
        if arr.shape != self.shape:
            raise ValueError(f"field shape {arr.shape} != {self.shape}")
        return np.ascontiguousarray(arr, dtype=np.float64)

def _as_fields(parts: tuple[np.ndarray, np.ndarray, np.ndarray]):
    return tuple(np.ascontiguousarray(part, dtype=np.float64) for part in parts)

def _spacing_2d(spacing: float | tuple[float, ...]) -> float | tuple[float, ...]:
    if np.isscalar(spacing):
        return float(spacing)
    vals = tuple(float(v) for v in spacing)  # type: ignore[union-attr]
    if len(vals) not in (2, 3):
        raise ValueError("spacing must be a scalar or length 2/3")
    return vals[0], vals[1]

def _cfl_dt(shape: tuple[int, int, int], spacing: tuple[float, float, float], eps_min: float) -> float:
    """Largest stable ``Δt``: ``c_max Δt √(Σ 1/Δᵢ²) = 1`` over axes longer than one cell."""
    terms = [1.0 / spacing[ax] ** 2 for ax in range(3) if shape[ax] > 1]
    if not terms:
        raise ValueError("grid must have an axis longer than 1")
    fastest = C_LIGHT / np.sqrt(eps_min)
    return 1.0 / (fastest * float(np.sqrt(sum(terms))))