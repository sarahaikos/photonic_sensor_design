"""Current distributions on the FDFD grid."""

from __future__ import annotations
import numpy as np
from ceviche.modes import get_modes


def gaussian_source(shape, center, waist, amplitude=1.0):
    """Separable Gaussian current, ``center`` and ``waist`` in cell units."""
    nx, ny = int(shape[0]), int(shape[1])
    x = np.arange(nx)[:, None]
    y = np.arange(ny)[None, :]
    cx, cy = float(center[0]), float(center[1])
    wx, wy = float(waist[0]), float(waist[1])
    return amplitude * np.exp(-((x - cx) / wx) ** 2 - ((y - cy) / wy) ** 2).astype(np.complex128)

def modal_source(eps_r, omega, dL, y, npml=0, m=1):
    """Mode ``m`` of the vertical slice at column ``y``, placed on that column. Propagates in ``y``."""
    cross = np.asarray(eps_r)[:, y]
    vals, vecs = get_modes(cross, omega, dL, npml, m=max(int(m), 1))
    pick = min(int(m), vecs.shape[1]) - 1
    source = np.zeros(np.shape(eps_r), dtype=np.complex128)
    source[:, y] = vecs[:, pick]
    return source, vals