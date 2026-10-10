"""Sparse solves for the FDFD system matrix. Direct SuperLU by default."""

from __future__ import annotations
from scipy.sparse.linalg import bicgstab, gmres, spsolve

_ITERATIVE = {
    "bicgstab": bicgstab,
    "gmres": gmres,
}

def solve_linear(A, b, iterative=False, method="bicgstab", atol=1e-8):
    """Solve ``A x = b``. ``iterative=True`` selects ``method`` (``bicgstab`` or ``gmres``)."""
    if not iterative:
        return spsolve(A, b)
    try:
        solver = _ITERATIVE[method]
    except KeyError as exc:
        raise ValueError(f"unknown iterative method {method!r}; use {sorted(_ITERATIVE)}") from exc
    x, info = solver(A, b, rtol=atol, atol=atol)
    if info != 0:
        raise RuntimeError(f"{method} failed to converge (info={info})")
    return x