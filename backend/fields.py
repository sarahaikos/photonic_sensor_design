"""FDTD-style field monitors: strip mode and coupler |E|²."""

from __future__ import annotations

import math

import numpy as np

from fdtd_soi import coupler_kappa0


def waveguide_field(
    width_nm: float,
    height_nm: float,
    gamma_core: float,
    polarization: str,
) -> dict:
    """Quasi-TE / quasi-TM |E|² like a 2D FDTD mode monitor on the cross-section."""
    w = float(width_nm)
    h = float(height_nm)
    g = min(max(gamma_core, 0.35), 0.95)
    nx, ny = 200, 160
    x = np.linspace(-1.6 * w, 1.6 * w, nx)
    y = np.linspace(-0.9 * h, h + 1.4 * h, ny)
    X, Y = np.meshgrid(x, y)

    gx = (2.4 * g) / max(w / 2, 1.0)
    gy_clad = (2.6 * g) / max(h, 1.0)
    gy_box = (3.0 * g) / max(h, 1.0)
    if polarization == "TM":
        gx *= 0.62
        gy_clad *= 0.42
        gy_box *= 0.5

    kx = 0.90 * math.pi / w
    ky = 0.86 * math.pi / h

    in_x = np.abs(X) <= w / 2
    Ax = np.cos(kx * np.clip(X, -w / 2, w / 2))
    Ax = np.where(in_x, Ax, np.cos(kx * w / 2) * np.exp(-gx * (np.abs(X) - w / 2)))

    in_core = (Y >= 0.0) & (Y <= h)
    Ay = np.cos(ky * (np.clip(Y, 0.0, h) - h / 2))
    e_bot = math.cos(ky * (-h / 2))
    e_top = math.cos(ky * (h / 2))
    Ay = np.where(Y < 0.0, e_bot * np.exp(gy_box * Y), Ay)
    Ay = np.where(Y > h, e_top * np.exp(-gy_clad * (Y - h)), Ay)

    e = Ax * Ay
    if polarization == "TM":
        # Ex discontinuity: |E| peaks in the low-index at the Si surface.
        surf = np.exp(-(((Y - h) / (0.10 * h)) ** 2))
        n_ratio = 3.48 / 1.44
        clad = np.where(Y > h, n_ratio**0.55, 1.0)
        e = e * clad + 0.7 * Ax * surf * np.max(np.abs(e))

    intensity = np.abs(e) ** 2
    peak = float(intensity.max()) or 1.0
    intensity = intensity / peak

    return {
        "x_nm": x.tolist(),
        "y_nm": y.tolist(),
        "intensity": intensity.tolist(),
        "quantity": "|E|^2",
        "core": {"x0": -w / 2, "y0": 0.0, "width": w, "height": h},
    }


def coupler_monitors(
    gap_nm: float,
    length_um: float,
    width_nm: float,
    height_nm: float,
    wavelength_nm: float,
    polarization: str,
    kappa0: float,
) -> dict:
    beat = (math.pi / 2) / kappa0 if kappa0 > 1e-9 else length_um
    z_max = max(length_um * 1.8, beat * 1.15, 20.0)
    z = np.linspace(0.0, z_max, 160)
    through_z = np.cos(kappa0 * z) ** 2
    cross_z = np.sin(kappa0 * z) ** 2

    lams = np.linspace(wavelength_nm * 0.94, wavelength_nm * 1.06, 81)
    through_l = []
    cross_l = []
    for lam in lams:
        k0 = coupler_kappa0(gap_nm, width_nm, height_nm, float(lam), polarization)
        ph = k0 * length_um
        c = math.sin(ph) ** 2
        through_l.append(1.0 - c)
        cross_l.append(c)

    w = width_nm * 1e-3
    gap = gap_nm * 1e-3
    x1 = -(w + gap) / 2
    x2 = (w + gap) / 2
    xs = np.linspace(x1 - 1.8 * w, x2 + 1.8 * w, 160)
    zs = np.linspace(0.0, max(length_um, 8.0), 180)
    X, Z = np.meshgrid(xs, zs)
    a1 = np.cos(kappa0 * Z)
    a2 = np.sin(kappa0 * Z)
    sig = w * 0.28
    e = a1 * np.exp(-((X - x1) / sig) ** 2) + a2 * np.exp(-((X - x2) / sig) ** 2)
    intensity = e**2
    intensity /= intensity.max() if intensity.max() else 1.0

    return {
        "length_um": z.tolist(),
        "through_vs_length": through_z.tolist(),
        "cross_vs_length": cross_z.tolist(),
        "wavelength_nm": lams.tolist(),
        "through_vs_wavelength": through_l,
        "cross_vs_wavelength": cross_l,
        "field": {
            "x_um": xs.tolist(),
            "z_um": zs.tolist(),
            "intensity": intensity.tolist(),
            "quantity": "|E|^2",
            "guides": [
                {"x0": x1 - w / 2, "y0": 0.0, "width": w, "height": float(zs[-1])},
                {"x0": x2 - w / 2, "y0": 0.0, "width": w, "height": float(zs[-1])},
            ],
        },
    }
