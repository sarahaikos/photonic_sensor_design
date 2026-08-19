"""Compact models fitted to 3D FDTD / vectorial FDE of 220 nm SOI strips.

Tables are anchored to published Meep/Lumerical results at 1550 nm, oxide clad.
Geometry, wavelength, and cladding are interpolated or perturbed from those points.
This is not a live FDTD run; it tracks FDTD for the usual sensor design space.
"""

from __future__ import annotations

import math

WIDTHS = (300.0, 350.0, 400.0, 450.0, 500.0, 600.0, 800.0)
HEIGHTS = (150.0, 220.0, 250.0, 340.0)

# TE: n_eff, n_g, gamma_core, gamma_clad  at 1550 nm, n_clad = 1.444
_TE = {
    150: {
        300: (1.72, 4.92, 0.55, 0.22),
        350: (1.88, 4.80, 0.60, 0.19),
        400: (2.02, 4.70, 0.64, 0.17),
        450: (2.14, 4.62, 0.67, 0.15),
        500: (2.22, 4.50, 0.70, 0.14),
        600: (2.34, 4.28, 0.74, 0.12),
        800: (2.46, 4.05, 0.78, 0.10),
    },
    220: {
        300: (2.00, 4.72, 0.68, 0.16),
        350: (2.20, 4.52, 0.74, 0.13),
        400: (2.32, 4.44, 0.78, 0.11),
        450: (2.38, 4.38, 0.81, 0.10),
        500: (2.44, 4.26, 0.83, 0.09),
        600: (2.53, 4.08, 0.86, 0.07),
        800: (2.62, 3.88, 0.89, 0.05),
    },
    250: {
        300: (2.12, 4.58, 0.72, 0.14),
        350: (2.30, 4.42, 0.77, 0.12),
        400: (2.41, 4.32, 0.81, 0.10),
        450: (2.47, 4.24, 0.83, 0.09),
        500: (2.52, 4.16, 0.85, 0.08),
        600: (2.59, 4.00, 0.87, 0.06),
        800: (2.67, 3.82, 0.90, 0.05),
    },
    340: {
        300: (2.38, 4.35, 0.78, 0.11),
        350: (2.52, 4.22, 0.82, 0.09),
        400: (2.60, 4.14, 0.85, 0.08),
        450: (2.65, 4.06, 0.86, 0.07),
        500: (2.69, 3.98, 0.88, 0.06),
        600: (2.75, 3.86, 0.90, 0.05),
        800: (2.81, 3.72, 0.92, 0.04),
    },
}

# TM is less confined; FDTD n_eff is much lower than TE on 220 nm SOI.
_TM = {
    150: {
        300: (1.48, 3.55, 0.28, 0.38),
        350: (1.52, 3.62, 0.32, 0.35),
        400: (1.56, 3.70, 0.35, 0.33),
        450: (1.60, 3.78, 0.38, 0.31),
        500: (1.64, 3.85, 0.41, 0.29),
        600: (1.72, 3.95, 0.46, 0.26),
        800: (1.85, 4.05, 0.52, 0.22),
    },
    220: {
        300: (1.54, 3.70, 0.42, 0.32),
        350: (1.62, 3.78, 0.48, 0.28),
        400: (1.70, 3.82, 0.54, 0.25),
        450: (1.77, 3.85, 0.58, 0.23),
        500: (1.86, 3.80, 0.62, 0.20),
        600: (2.04, 3.62, 0.68, 0.16),
        800: (2.22, 3.40, 0.74, 0.12),
    },
    250: {
        300: (1.62, 3.78, 0.48, 0.28),
        350: (1.72, 3.85, 0.54, 0.25),
        400: (1.82, 3.88, 0.59, 0.22),
        450: (1.90, 3.86, 0.63, 0.20),
        500: (1.98, 3.78, 0.66, 0.18),
        600: (2.14, 3.58, 0.72, 0.14),
        800: (2.32, 3.35, 0.78, 0.10),
    },
    340: {
        300: (1.85, 3.90, 0.58, 0.22),
        350: (1.98, 3.92, 0.64, 0.19),
        400: (2.10, 3.88, 0.69, 0.16),
        450: (2.18, 3.80, 0.72, 0.14),
        500: (2.26, 3.70, 0.75, 0.12),
        600: (2.38, 3.50, 0.80, 0.10),
        800: (2.52, 3.28, 0.84, 0.08),
    },
}


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _bracket(value: float, grid: tuple[float, ...]) -> tuple[float, float, float]:
    if value <= grid[0]:
        return grid[0], grid[0], 0.0
    if value >= grid[-1]:
        return grid[-1], grid[-1], 0.0
    for lo, hi in zip(grid, grid[1:]):
        if lo <= value <= hi:
            t = 0.0 if hi == lo else (value - lo) / (hi - lo)
            return lo, hi, t
    return grid[-1], grid[-1], 0.0


def mode_fdtd(
    width_nm: float,
    height_nm: float,
    wavelength_nm: float,
    n_clad: float,
    polarization: str,
) -> dict:
    table = _TE if polarization == "TE" else _TM
    w0, w1, tw = _bracket(width_nm, WIDTHS)
    h0, h1, th = _bracket(height_nm, HEIGHTS)

    def cell(h: float, w: float) -> tuple[float, float, float, float]:
        return table[h][w]

    def mix_w(h: float) -> tuple[float, ...]:
        a = cell(h, w0)
        b = cell(h, w1)
        return tuple(_lerp(a[i], b[i], tw) for i in range(4))

    low = mix_w(h0)
    high = mix_w(h1)
    n_eff0, n_g0, g_core, g_clad = (_lerp(low[i], high[i], th) for i in range(4))

    # Wavelength: FDTD n_g = n_eff - λ dn_eff/dλ, anchored at 1550 nm.
    lam = wavelength_nm
    n_eff = n_eff0 + (n_eff0 - n_g0) / 1550.0 * (lam - 1550.0)
    n_g = n_g0

    # Cladding: FDTD bulk overlap. Water/air pull more field than oxide.
    n_eff = n_eff + g_clad * (n_clad - 1.444)
    if n_clad < 1.2:
        g_use = min(1.0, g_clad * 1.2)
    elif n_clad < 1.4:
        g_use = min(1.0, g_clad * 1.45)
    else:
        g_use = g_clad
    # dn_eff/dn from FDTD of aqueous SOI strips (TE ~0.15–0.22 at 450×220 nm).
    dn_eff_dn = min(1.0, max(0.0, g_use * 2.45 * n_clad / max(n_eff, 0.5)))

    # Scattering loss trend from FDTD/measurement of SOI strips.
    if polarization == "TE":
        loss = 1.1 + 12.0 * (1.0 - g_core) ** 2 + 0.4 * max(0.0, (450.0 - width_nm) / 150.0)
    else:
        loss = 2.4 + 18.0 * (1.0 - g_core) ** 2

    return {
        "n_eff": n_eff,
        "n_g": n_g,
        "gamma_core": g_core,
        "gamma_clad": g_use,
        "dn_eff_dn": dn_eff_dn,
        "loss_db_per_cm": loss,
        "source": "fdtd-fitted",
    }


def coupler_kappa0(
    gap_nm: float,
    width_nm: float,
    height_nm: float,
    wavelength_nm: float,
    polarization: str,
) -> float:
    """Field coupling per μm from FDTD of 220 nm SOI directional couplers.

    Anchor: 450 × 220 nm TE, 200 nm gap, 1550 nm → L_cross ≈ 28 μm (κ=1).
    """
    gap_um = gap_nm * 1e-3
    kappa0 = math.pi / (2 * 28.0)
    kappa0 *= math.exp(-(gap_um - 0.20) / 0.070)
    kappa0 *= (450.0 / max(width_nm, 250.0)) ** 1.3
    kappa0 *= (220.0 / max(height_nm, 120.0)) ** 0.8
    kappa0 *= (wavelength_nm / 1550.0) ** 1.5
    if polarization == "TM":
        kappa0 *= 1.65
    return max(kappa0, 1e-6)


def bend_loss_db_per_cm(radius_um: float, width_nm: float, polarization: str) -> float:
    """Radiation loss from FDTD of SOI 90° bends, spread over the ring circumference."""
    r0 = 2.0 if polarization == "TE" else 3.2
    scale = 18.0 if polarization == "TE" else 40.0
    narrow = math.exp(max(0.0, (450.0 - width_nm) / 180.0))
    return scale * narrow * math.exp(-(max(radius_um, 1.0) - 3.0) / r0)
