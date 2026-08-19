"""CPML absorbing boundaries, plus periodic / symmetry flags."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fdtd.materials import EPS0, MU0


@dataclass
class BoundarySpec:
    kind: str  # cpml | pbc | bloch | pec | pmc
    npml: int
    m: int = 3
    r0: float = 1e-12
    kappa_max: float = 7.0
    symmetry_x: str | None = None  # even | odd
    symmetry_y: str | None = None
    bloch_kx: float = 0.0
    bloch_ky: float = 0.0

    def to_dict(self) -> dict:
        return {
            "absorbing": "CPML" if self.kind == "cpml" else self.kind,
            "npml": self.npml,
            "polynomial_m": self.m,
            "kappa_max": self.kappa_max,
            "symmetry_x": self.symmetry_x,
            "symmetry_y": self.symmetry_y,
            "periodic": self.kind == "pbc",
            "bloch": self.kind == "bloch",
            "domain_reduction": _reduction(self.symmetry_x, self.symmetry_y),
        }


def _reduction(sx: str | None, sy: str | None) -> str:
    n = (2 if sx else 1) * (2 if sy else 1)
    return f"{n}x" if n > 1 else "none"


def cpml_profile(npml: int, dx: float, dt: float, n_max: float, m: int = 3, kappa_max: float = 7.0):
    """1D CPML σ, κ, b, c along one axis (Gedney)."""
    sigma_max = 0.8 * (m + 1) / (MU0 / EPS0) ** 0.5 / dx / n_max * 0.5
    # σ_max ≈ (m+1)/(150 π n dx) in SI-ish; use Taflove form:
    eta = (MU0 / EPS0) ** 0.5
    sigma_max = (m + 1) / (150.0 * np.pi * n_max * dx)
    rho = (np.arange(npml) + 0.5) / npml
    sigma = sigma_max * rho**m
    kappa = 1.0 + (kappa_max - 1.0) * rho**m
    alpha = 0.05 * (1.0 - rho)
    denom = kappa * EPS0 + dt * (sigma + kappa * alpha * EPS0)
    b = np.exp(-(sigma / kappa + alpha) * dt / EPS0)
    c = (sigma * (b - 1.0)) / (sigma * kappa + kappa**2 * alpha * EPS0 + 1e-30)
    return sigma, kappa, b, c, eta, sigma_max


class CPML:
    def __init__(self, nx: int, ny: int, npml: int, dx: float, dy: float, dt: float, n_max: float):
        self.npml = npml
        _, _, bx, cx, _, _ = cpml_profile(npml, dx, dt, n_max)
        _, _, by, cy, _, _ = cpml_profile(npml, dy, dt, n_max)
        self.bx_lo, self.cx_lo = bx[::-1], cx[::-1]
        self.bx_hi, self.cx_hi = bx, cx
        self.by_lo, self.cy_lo = by[::-1], cy[::-1]
        self.by_hi, self.cy_hi = by, cy
        self.psi_ezx = np.zeros((nx, ny))
        self.psi_ezy = np.zeros((nx, ny))
        self.psi_hxy = np.zeros((nx, ny))
        self.psi_hyx = np.zeros((nx, ny))
