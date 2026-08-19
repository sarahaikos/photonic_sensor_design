"""2D finite-difference eigenmode solver for the waveguide cross-section."""

from __future__ import annotations

import numpy as np

from fdtd.materials import MATERIALS
from fdtd.mesh import cross_section_eps


def solve_strip_mode(
    width_nm: float,
    height_nm: float,
    wavelength_nm: float,
    n_clad: float,
    polarization: str,
) -> dict:
    n_core = MATERIALS["si"].n
    n_box = MATERIALS["sio2"].n
    x, y, eps = cross_section_eps(width_nm, height_nm, n_core, n_box, n_clad, dx_nm=50.0)
    n_eff, psi = _eigh_neff(eps, x, y, wavelength_nm, n_clad, n_box, n_core, polarization)
    intensity = (psi * psi).T
    return {
        "n_eff": n_eff,
        "polarization": polarization,
        "method": "2d-fd-eigenmode",
        "grid": {"nx": int(eps.shape[0]), "ny": int(eps.shape[1]), "dx_nm": 50.0},
        "field": {
            "x_nm": x.tolist(),
            "y_nm": y.tolist(),
            "intensity": np.abs(intensity).tolist(),
            "quantity": "|ψ|²",
            "core": {
                "x0": -width_nm / 2,
                "y0": 0.0,
                "width": width_nm,
                "height": height_nm,
            },
        },
    }


def _eigh_neff(
    eps: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    wavelength_nm: float,
    n_clad: float,
    n_box: float,
    n_core: float,
    polarization: str,
) -> tuple[float, np.ndarray]:
    dx = (x[1] - x[0]) * 1e-9
    k0 = 2 * np.pi / (wavelength_nm * 1e-9)
    nx, ny = eps.shape
    ncells = nx * ny
    invp = np.ones_like(eps) if polarization == "TE" else 1.0 / np.maximum(eps, 1e-6)
    a = np.zeros((ncells, ncells))
    n2 = eps.ravel()
    scale = 1.0 / (k0 * dx) ** 2
    for i in range(nx):
        for j in range(ny):
            p = i * ny + j
            sx = 0.5 * (invp[i, j] + invp[min(i + 1, nx - 1), j]) * scale
            sy = 0.5 * (invp[i, j] + invp[i, min(j + 1, ny - 1)]) * scale
            a[p, p] = n2[p] - 2 * sx - 2 * sy
            if i + 1 < nx:
                a[p, p + ny] = sx
                a[p + ny, p] = sx
            if j + 1 < ny:
                a[p, p + 1] = sy
                a[p + 1, p] = sy
    vals, vecs = np.linalg.eigh(a)
    n_effs = np.sqrt(np.clip(vals, 0.0, None))
    guided = np.where((n_effs > max(n_clad, n_box) + 0.02) & (n_effs < n_core - 0.02))[0]
    k = int(guided[-1]) if guided.size else int(np.argmax(n_effs))
    psi = vecs[:, k].reshape(nx, ny)
    peak = float(np.max(np.abs(psi))) or 1.0
    return float(n_effs[k]), psi / peak
