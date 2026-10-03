"""Grid helpers shared by the solvers."""

from __future__ import annotations
import numpy as np
from ceviche.constants import C_0

def omega_from_wavelength(wavelength: float) -> float:
    """Angular frequency (rad/s) for a free-space wavelength in meters."""
    if wavelength <= 0.0:
        raise ValueError("wavelength must be positive")
    return 2.0 * np.pi * C_0 / float(wavelength)

def grid_xy(shape, dL, origin=(0.0, 0.0)):
    """Cell-center coordinates ``(x, y)`` in meters, each shaped ``(Nx, Ny)``."""
    nx, ny = int(shape[0]), int(shape[1])
    if np.isscalar(dL):
        dx = dy = float(dL)
    else:
        dx, dy = float(dL[0]), float(dL[1])
    x = origin[0] + (np.arange(nx) + 0.5) * dx
    y = origin[1] + (np.arange(ny) + 0.5) * dy
    return np.meshgrid(x, y, indexing="ij")

def as_grid(array, shape):
    """Ravel-compatible reshape to ``shape``."""
    grid = np.asarray(array)
    if grid.shape == tuple(shape):
        return grid
    if grid.size != int(np.prod(shape)):
        raise ValueError(f"cannot reshape size {grid.size} into {shape}")
    return grid.reshape(shape)