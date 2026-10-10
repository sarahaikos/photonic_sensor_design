"""Yee FDTD with a polynomial absorber on the periodic boundaries.

``eps_r`` may be ``(Nx, Ny)`` or ``(Nx, Ny, Nz)``. A 2D grid is stored as ``Nz = 1``,
so ``∂/∂z = 0``. One call to ``forward`` advances ``H`` then ``E`` by ``dt``.
"""

from __future__ import annotations

import numpy as np

from ceviche.constants import C_0, EPSILON_0, MU_0
from ceviche.derivatives import curl_e, curl_h


class fdtd:
    """Real-valued Yee stepper. Fields live on one grid; staggering is in the curls."""

    def __init__(self, eps_r, dL, npml, stability=0.5):
        raw = np.asarray(eps_r, dtype=float)
        if raw.ndim == 2:
            raw = raw[:, :, None]
            self._flat = True
        elif raw.ndim == 3:
            self._flat = False
        else:
            raise ValueError("eps_r must have shape (Nx, Ny) or (Nx, Ny, Nz)")
        if np.any(raw <= 0.0):
            raise ValueError("eps_r must be positive")
        self.grid_shape = tuple(int(n) for n in raw.shape)
        self.Nx, self.Ny, self.Nz = self.grid_shape
        self._stability = float(stability)
        self.dL = dL
        self.npml = npml
        self.eps_r = raw

    def __repr__(self) -> str:
        shown = self.grid_shape[:2] if self._flat else self.grid_shape
        return f"fdtd(eps_r.shape={shown}, dL={self.dL}, npml={self.npml})"

    @property
    def dL(self):
        return self._dL

    @dL.setter
    def dL(self, new_dL):
        self._dL = new_dL
        self._set_time_step()
        if hasattr(self, "_npml"):
            self._sigma = _pml_sigma(self.grid_shape, self._npml, self.dt)

    @property
    def npml(self):
        return self._npml

    @npml.setter
    def npml(self, new_npml):
        self._npml = _npml3(new_npml, self._flat)
        if hasattr(self, "dt"):
            self._sigma = _pml_sigma(self.grid_shape, self._npml, self.dt)

    @property
    def eps_r(self):
        grid = self._eps
        return grid[:, :, 0] if self._flat else grid

    @eps_r.setter
    def eps_r(self, new_eps):
        grid = np.asarray(new_eps, dtype=float)
        if grid.ndim == 2:
            grid = grid[:, :, None]
        if grid.shape != self.grid_shape:
            raise ValueError(f"eps_r shape {grid.shape} != {self.grid_shape}")
        self._eps = grid
        self._sigma = _pml_sigma(self.grid_shape, self._npml, self.dt)
        self.initialize_fields()

    def initialize_fields(self):
        z = np.zeros(self.grid_shape)
        self.t_index = 0
        self.Ex = z.copy()
        self.Ey = z.copy()
        self.Ez = z.copy()
        self.Hx = z.copy()
        self.Hy = z.copy()
        self.Hz = z.copy()
        self.fields = {name: getattr(self, name) for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}

    def forward(self, Jx=None, Jy=None, Jz=None):
        """Advance one time step. Electric currents are added to ``E`` after the curl update."""
        self.t_index += 1
        sigma = self._sigma[0] + self._sigma[1] + self._sigma[2]
        curl_x = curl_e(0, self.Ex, self.Ey, self.Ez, self.dL)
        curl_y = curl_e(1, self.Ex, self.Ey, self.Ez, self.dL)
        curl_z = curl_e(2, self.Ex, self.Ey, self.Ez, self.dL)
        self.Hx = _damp(self.Hx, -curl_x / MU_0, sigma, self.dt)
        self.Hy = _damp(self.Hy, -curl_y / MU_0, sigma, self.dt)
        self.Hz = _damp(self.Hz, -curl_z / MU_0, sigma, self.dt)

        curl_x = curl_h(0, self.Hx, self.Hy, self.Hz, self.dL)
        curl_y = curl_h(1, self.Hx, self.Hy, self.Hz, self.dL)
        curl_z = curl_h(2, self.Hx, self.Hy, self.Hz, self.dL)
        eps = EPSILON_0 * self._eps
        self.Ex = _damp(self.Ex, curl_x / eps, sigma, self.dt)
        self.Ey = _damp(self.Ey, curl_y / eps, sigma, self.dt)
        self.Ez = _damp(self.Ez, curl_z / eps, sigma, self.dt)
        self.Ex = self.Ex + self._source(Jx)
        self.Ey = self.Ey + self._source(Jy)
        self.Ez = self.Ez + self._source(Jz)
        self.fields = {name: getattr(self, name) for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}
        return self._view(self.fields)

    def run(self, steps: int, jz=None):
        """Take ``steps`` updates. ``jz`` is a constant array or ``jz(step)``."""
        last = None
        for step in range(int(steps)):
            source = jz(step) if callable(jz) else jz
            last = self.forward(Jz=source)
        return last

    def _source(self, current):
        if current is None:
            return 0.0
        array = np.asarray(current, dtype=float)
        if array.ndim == 2 and self.grid_shape[2] == 1:
            array = array[:, :, None]
        return array

    def _set_time_step(self):
        dx, dy, dz = _spacing3(self._dL)
        terms = []
        for n, spacing in zip(self.grid_shape, (dx, dy, dz)):
            if n > 1:
                terms.append(1.0 / spacing**2)
        if not terms:
            raise ValueError("grid must have at least one axis longer than 1")
        self.dt = self._stability / (C_0 * np.sqrt(sum(terms)))

    def _view(self, fields):
        if not self._flat:
            return fields
        return {name: value[:, :, 0] for name, value in fields.items()}


def _damp(field, rhs, sigma, dt):
    """Leapfrog step with conductivity averaged over the step: ``∂f/∂t = rhs − (σ/ε₀) f``."""
    alpha = sigma * dt / (2.0 * EPSILON_0)
    return ((1.0 - alpha) * field + dt * rhs) / (1.0 + alpha)


def _pml_sigma(shape, npml, dt):
    """One conductivity array per axis, broadcast to the full grid. Zero outside the sponge."""
    peak = 0.5 * EPSILON_0 / dt
    axes = []
    for axis, (n, n_pml) in enumerate(zip(shape, npml)):
        prof = np.zeros(n)
        if n_pml > 0 and n > 1:
            ramp = ((np.arange(n_pml) + 1) / n_pml) ** 3
            prof[:n_pml] = peak * ramp[::-1]
            prof[-n_pml:] = peak * ramp
        shape_i = [1, 1, 1]
        shape_i[axis] = n
        axes.append(prof.reshape(shape_i))
    return tuple(axes)


def _npml3(npml, flat: bool):
    if isinstance(npml, (int, np.integer)):
        n = int(npml)
        return (n, n, 0 if flat else n)
    vals = tuple(int(v) for v in npml)
    if len(vals) == 2:
        return (vals[0], vals[1], 0)
    if len(vals) == 3:
        return vals
    raise ValueError("npml must be an int or length 2/3")


def _spacing3(dL):
    if np.isscalar(dL):
        value = float(dL)
        return value, value, value
    vals = tuple(float(v) for v in dL)
    if len(vals) == 2:
        return vals[0], vals[1], min(vals)
    return vals[0], vals[1], vals[2]
