"""1D waveguide cross-section modes for an Ez source profile.

The eigenvalue problem on a transverse line is

    (ε + ∂²/k₀²) v = n_eff² v

``get_modes`` returns ``(n_eff**2, profiles)`` with profiles normalized to
``sum(|v|²) = 1`` and sorted by descending real effective index.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs

from ceviche.constants import C_0
from ceviche.derivatives import compute_derivative_matrices


def get_modes(eps_cross, omega, dL, npml, m=1, filtering=True):
    """Guided modes of a 1D permittivity slice.

    ``eps_cross`` is 1D. ``npml`` is the number of PML cells on each end of that slice.
    ``m`` is how many eigenvalues to request before filtering.
    """
    eps = np.asarray(eps_cross, dtype=np.complex128).ravel()
    n = eps.size
    if n < 4:
        raise ValueError("cross section needs at least 4 cells")
    k_req = int(m)
    if k_req < 1 or k_req >= n - 1:
        raise ValueError(f"m must be in 1..{n - 2}")

    dxf, dxb, _, _ = compute_derivative_matrices(omega, (n, 1), (int(npml), 0), dL)
    k0 = float(omega) / C_0
    A = sparse.diags(eps, format="csc") + (dxf @ dxb) * (1.0 / k0**2)
    n_max = np.sqrt(np.max(np.real(eps)))
    vals, vecs = eigs(A, k=k_req, sigma=n_max**2, which="LM")
    if filtering:
        keep = np.real(vals) > 0.0
        vals, vecs = vals[keep], vecs[:, keep]
    if vals.size == 0:
        raise RuntimeError("no guided modes for this cross section")
    order = np.argsort(-np.real(vals))
    vals, vecs = vals[order], vecs[:, order]
    return vals, _normalize(vecs)


def insert_mode(omega, dL, eps_r, axis, index, m=1, npml=0):
    """Place mode ``m`` (1-based) of the slice ``eps_r`` at ``index`` along ``axis`` ('x' or 'y')."""
    grid = np.asarray(eps_r)
    if axis == "y":
        cross = grid[:, index]
        slot = (slice(None), index)
    elif axis == "x":
        cross = grid[index, :]
        slot = (index, slice(None))
    else:
        raise ValueError("axis must be 'x' or 'y'")
    vals, vecs = get_modes(cross, omega, dL, npml, m=max(int(m), 1), filtering=True)
    pick = min(int(m), vecs.shape[1]) - 1
    target = np.zeros(grid.shape, dtype=np.complex128)
    target[slot] = vecs[:, pick]
    return target, vals


def _normalize(vectors):
    power = np.sum(np.abs(vectors) ** 2, axis=0)
    return vectors / np.sqrt(power)
