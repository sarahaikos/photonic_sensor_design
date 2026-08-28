"""2D Yee FDTD (Ez, Hx, Hy) with graded CPML-profile absorbers, mode injection, DFT.

Propagation is +y. Transverse is x. 2.5D in-plane engine for a directional coupler.
"""

from __future__ import annotations

import numpy as np

from fdtd.materials import C0, EPS0, MU0
from fdtd.mesh import YeeGrid
from fdtd.monitors import overlap, poynting_power
from fdtd.sources import gaussian_mode


def run_yee(
    grid: YeeGrid,
    wavelength_nm: float,
    src_y: float,
    mon_y: float,
    through_x: float,
    drop_x: float | None,
    wg_width_um: float,
    steps: int | None = None,
) -> dict:
    eps = grid.eps_r
    nx, ny = grid.nx, grid.ny
    dx, dy = grid.dx, grid.dy
    dt = 0.49 * min(dx, dy) / C0 / np.sqrt(2.0)
    freq = C0 / (wavelength_nm * 1e-9)
    omega = 2 * np.pi * freq
    n_steps = steps or min(4000, max(1000, int(2.2 * ny * max(grid.n_max, 1.5))))

    Ez = np.zeros((nx, ny))
    Hx = np.zeros((nx, ny))
    Hy = np.zeros((nx, ny))
    damp = _cpml_damp(nx, ny, grid.npml)

    db = dt / MU0 / dx
    ce = dt / EPS0 / dx

    j_src = _index(grid.y, src_y)
    j_mon = _index(grid.y, mon_y)
    profile = gaussian_mode(nx, through_x, wg_width_um * 0.5, grid.x)
    drop_profile = (
        gaussian_mode(nx, drop_x, wg_width_um * 0.5, grid.x) if drop_x is not None else None
    )
    mask_th = np.abs(grid.x - through_x) <= wg_width_um * 0.9
    mask_dr = (
        np.abs(grid.x - drop_x) <= wg_width_um * 0.9 if drop_x is not None else np.zeros(nx, bool)
    )

    t0 = 3.0 / freq
    tau = 0.6 / freq

    dft_ez_src = np.zeros(nx, dtype=complex)
    dft_hx_src = np.zeros(nx, dtype=complex)
    dft_ez_mon = np.zeros(nx, dtype=complex)
    dft_hx_mon = np.zeros(nx, dtype=complex)

    snap = None
    for n in range(n_steps):
        t = n * dt
        pulse = np.exp(-((t - t0) / tau) ** 2) * np.sin(omega * t)
        Ez[:, j_src] += ce * profile * pulse

        Hx[:, :-1] -= db * (Ez[:, 1:] - Ez[:, :-1])
        Hy[:-1, :] += db * (Ez[1:, :] - Ez[:-1, :])
        Hx *= damp
        Hy *= damp

        curl = np.zeros_like(Ez)
        curl[1:, :] += Hy[1:, :] - Hy[:-1, :]
        curl[:, 1:] -= Hx[:, 1:] - Hx[:, :-1]
        Ez += ce * curl / np.maximum(eps, 1.0)
        Ez *= damp

        if n > n_steps // 5:
            phase = np.exp(-1j * omega * t)
            dft_ez_src += Ez[:, j_src + 4] * phase
            dft_hx_src += Hx[:, j_src + 4] * phase
            dft_ez_mon += Ez[:, j_mon] * phase
            dft_hx_mon += Hx[:, j_mon] * phase

        if n == n_steps * 2 // 3:
            snap = np.abs(Ez).copy()

    eth = float(np.sum(np.abs(dft_ez_mon[mask_th]) ** 2))
    edr = float(np.sum(np.abs(dft_ez_mon[mask_dr]) ** 2)) if drop_x is not None else 0.0
    tot = eth + edr + 1e-30
    t_through = eth / tot
    t_drop = edr / tot if drop_x is not None else None
    pin = abs(poynting_power(dft_ez_src * mask_th, dft_hx_src * mask_th, dx))
    pth = abs(poynting_power(dft_ez_mon * mask_th, dft_hx_mon * mask_th, dx))
    field = None
    if snap is not None:
        # snap is |Ez|(nx, ny). Plot propagation along x so the two rails read as a coupler.
        peak = float(np.nanmax(snap)) or 1.0
        if np.isfinite(peak) and peak > 0:
            y0 = float(grid.y[0])
            yspan = float(grid.y[-1] - grid.y[0])
            guides = [
                {
                    "x0": y0,
                    "y0": float(through_x - wg_width_um * 0.5),
                    "width": yspan,
                    "height": float(wg_width_um),
                }
            ]
            if drop_x is not None:
                guides.append(
                    {
                        "x0": y0,
                        "y0": float(drop_x - wg_width_um * 0.5),
                        "width": yspan,
                        "height": float(wg_width_um),
                    }
                )
            field = {
                "x_um": grid.y.tolist(),
                "z_um": grid.x.tolist(),
                "intensity": np.clip(snap / peak, 0, 1).tolist(),
                "quantity": "|Ez| snapshot",
                "layout": "strip",
                "guides": guides,
            }
    return {
        "steps": n_steps,
        "dt_fs": dt * 1e15,
        "courant": float(C0 * dt / dx * np.sqrt(2.0)),
        "t_through": t_through,
        "t_drop": t_drop,
        "kappa": t_drop,
        "poynting_in": pin,
        "poynting_through": pth,
        "overlap_through": overlap(dft_ez_mon * mask_th, profile),
        "overlap_drop": overlap(dft_ez_mon * mask_dr, drop_profile) if drop_profile is not None else None,
        "field": field,
        "engine": "yee-2d-ez",
        "parallel": "numpy-vectorized",
        "dft": True,
        "poynting": True,
        "absorbing": "CPML-profiled conductivity PML",
    }


def _index(coords: np.ndarray, value: float) -> int:
    return int(np.clip(np.searchsorted(coords, value), 4, coords.size - 5))


def _cpml_damp(nx: int, ny: int, npml: int, m: int = 3) -> np.ndarray:
    """Polynomial conductivity profile (same grading as CPML) applied as a field damper."""
    damp = np.ones((nx, ny))
    if npml < 2:
        return damp
    for i in range(npml):
        r = ((npml - i) / npml) ** m
        factor = np.exp(-2.2 * r)
        damp[i, :] *= factor
        damp[-1 - i, :] *= factor
    for j in range(npml):
        r = ((npml - j) / npml) ** m
        factor = np.exp(-2.2 * r)
        damp[:, j] *= factor
        damp[:, -1 - j] *= factor
    return damp
