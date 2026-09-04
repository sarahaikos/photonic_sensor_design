"""Compact thermo-optic corners for SOI strips and directional couplers.

dn/dT from Cocorullo / Komma (Si ~1550 nm) and typical SiO2 / water values.
This is a uniform ΔT on the compact models — not a heat PDE or inverse design.
"""

from __future__ import annotations

import math

# /K at ~1550 nm.
DN_DT_SI = 1.86e-4
DN_DT_SIO2 = 1.0e-5
DN_DT_WATER = -1.0e-4
DN_DT_AIR = 0.0

# How much extra core–clad contrast shrinks evanescent κ₀.
_KAPPA_CONTRAST = 6.0

CORNERS_K = (-20.0, -10.0, 0.0, 10.0, 20.0, 40.0)


def dn_clad_dt(n_clad: float) -> float:
    if n_clad < 1.15:
        return DN_DT_AIR
    if n_clad < 1.40:
        return DN_DT_WATER
    return DN_DT_SIO2


def dn_eff_dt(gamma_core: float, gamma_clad: float, n_clad: float) -> float:
    """First-order overlap: Γ_si dn_si/dT + Γ_clad dn_clad/dT."""
    return float(gamma_core) * DN_DT_SI + float(gamma_clad) * dn_clad_dt(n_clad)


def n_eff_at(n_eff: float, dndt: float, delta_t_k: float) -> float:
    return float(n_eff) + float(dndt) * float(delta_t_k)


def kappa0_at(kappa0: float, dndt: float, n_clad: float, delta_t_k: float) -> float:
    """Si heats faster than clad → tighter mode → weaker coupling."""
    d_contrast = (dndt - dn_clad_dt(n_clad)) * float(delta_t_k)
    return max(1e-6, float(kappa0) * math.exp(-_KAPPA_CONTRAST * d_contrast))


def kappa_from_kappa0(kappa0: float, length_um: float) -> float:
    return math.sin(kappa0 * length_um) ** 2


def resonance_shift_nm(resonance_nm: float, n_g: float, dndt: float, delta_t_k: float) -> float:
    """dλ/dT = (λ / n_g) dn_eff/dT."""
    if n_g <= 0:
        return 0.0
    return float(resonance_nm) / float(n_g) * float(dndt) * float(delta_t_k)
