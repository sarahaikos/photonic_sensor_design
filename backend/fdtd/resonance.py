"""Resonant wavelength and quality factor on one Yee grid.

FDFD is the curl-curl eigenproblem for TMz, continuous in time:

    ∇ × ∇ × Ez = ω₀² μ₀ ε₀ εᵣ Ez

FDTD is the leapfrog on those same curls. A resolved step maps the
oscillation by ``sin(ω Δt / 2) = ω₀ Δt / 2``; undoing that map is how the
two wavelengths are compared.

Electric loss with a constant loss tangent ``σ = α ε₀ εᵣ`` leaves the mode
shape alone and moves the pole to

    ω² + i α ω − ω₀² = 0,    Q = Re(ω) / α.

The field amplitude decays as ``e^{-α t / 2}``.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

from fdfd.operators import yee_operators
from fdtd.materials import EPS0, MU0
from fdtd.yee import C_LIGHT, YeeFdtd


def block_permittivity(
    shape: tuple[int, int],
    eps_core: float,
    eps_bg: float = 1.0,
) -> np.ndarray:
    """Dielectric block on a uniform background. The block is the cavity defect."""
    if float(eps_core) <= 0.0 or float(eps_bg) <= 0.0:
        raise ValueError("permittivities must be positive")
    nx, ny = (int(shape[0]), int(shape[1]))
    if nx < 8 or ny < 8:
        raise ValueError("cavity grid must be at least 8 cells on a side")
    eps = np.full((nx, ny), float(eps_bg), dtype=np.float64)
    eps[nx // 5 : 4 * nx // 5, ny // 4 : 3 * ny // 4] = float(eps_core)
    return eps


def fdfd_resonance(
    eps,
    spacing: float | tuple[float, ...],
    *,
    bc: str | tuple[str, ...] = "pec",
    alpha: float = 0.0,
) -> dict:
    """Lowest TMz resonance of ``eps`` from the discrete curl-curl operator."""
    grid = np.asarray(eps, dtype=np.float64)
    if grid.ndim != 2:
        raise ValueError("eps must have shape (nx, ny)")
    omega0, ez = _fundamental(grid, spacing, bc)
    pole = _pole(omega0, alpha)
    return {
        "omega0": omega0,
        "omega": float(pole.real),
        "gamma": float(pole.imag),
        "wavelength_nm": _wavelength_nm(float(pole.real)),
        "q": _quality(pole, alpha),
        "ez": ez,
    }


def fdtd_resonance(
    eps,
    spacing: float | tuple[float, ...],
    ez,
    *,
    bc: str | tuple[str, ...] = "pec",
    alpha: float = 0.0,
    cfl: float = 0.5,
    steps: int = 600,
) -> dict:
    """Ringdown of ``ez`` on ``YeeFdtd``. ``alpha`` is the loss tangent rate."""
    grid = np.asarray(eps, dtype=np.float64)
    sigma = 0.0 if float(alpha) == 0.0 else float(alpha) * EPS0 * grid
    sim = YeeFdtd(grid, spacing, bc=bc, cfl=cfl, sigma=sigma)
    sim.set_electric(ez=ez, cosine_peak=True)
    series = _project(sim, ez, int(steps))
    omega, gamma, residual = _fit_damped_cosine(series, sim.dt)
    corrected = _leapfrog_to_spatial(omega, sim.dt)
    return {
        "omega": omega,
        "omega_corrected": corrected,
        "gamma": gamma,
        "wavelength_nm": _wavelength_nm(omega),
        "wavelength_corrected_nm": _wavelength_nm(corrected),
        "q": None if float(alpha) == 0.0 else omega / (2.0 * abs(gamma)),
        "q_corrected": None if float(alpha) == 0.0 else corrected / (2.0 * abs(gamma)),
        "fit_residual": residual,
        "dt": sim.dt,
        "steps": int(steps),
    }


def compare_resonances(
    permittivities,
    spacing: float | tuple[float, ...],
    *,
    alpha: float = 0.0,
    cfl: float = 0.5,
    steps: int = 600,
    bc: str | tuple[str, ...] = "pec",
) -> dict:
    """FDTD against FDFD for each permittivity, plus the wavelength shift.

    The shift is the second resonance minus the first. A higher defect index
    moves it to the red.
    """
    grids = [np.asarray(eps, dtype=np.float64) for eps in permittivities]
    if len(grids) != 2 or grids[0].shape != grids[1].shape:
        raise ValueError("pass two permittivity grids of the same shape")
    rows = []
    for grid in grids:
        freq = fdfd_resonance(grid, spacing, bc=bc, alpha=alpha)
        time = fdtd_resonance(
            grid, spacing, freq["ez"], bc=bc, alpha=alpha, cfl=cfl, steps=steps
        )
        rows.append({"fdfd": freq, "fdtd": time})
    return {
        "rows": rows,
        "shift_fdfd_nm": rows[1]["fdfd"]["wavelength_nm"] - rows[0]["fdfd"]["wavelength_nm"],
        "shift_fdtd_nm": rows[1]["fdtd"]["wavelength_nm"] - rows[0]["fdtd"]["wavelength_nm"],
        "shift_corrected_nm": (
            rows[1]["fdtd"]["wavelength_corrected_nm"] - rows[0]["fdtd"]["wavelength_corrected_nm"]
        ),
    }


def _fundamental(eps: np.ndarray, spacing, bc) -> tuple[float, np.ndarray]:
    ops = yee_operators(eps.shape, spacing, bc=bc)
    # TMz: (∇ × ∇ × Ez) = -(∂xb ∂xf + ∂yb ∂yf) Ez = ω₀² μ₀ ε₀ εᵣ Ez.
    laplace = -(ops.Dxb @ ops.Dxf + ops.Dyb @ ops.Dyf).astype(np.float64)
    mass = (MU0 * EPS0) * sparse.diags(eps.ravel())
    dx, dy = ops.spacing[0], ops.spacing[1]
    n_avg = float(np.sqrt(np.mean(eps)))
    guess = (
        np.pi
        * C_LIGHT
        / n_avg
        * np.sqrt(1.0 / (eps.shape[0] * dx) ** 2 + 1.0 / (eps.shape[1] * dy) ** 2)
    )
    vals, vecs = eigsh(laplace, k=3, M=mass, sigma=(0.35 * guess) ** 2, which="LM")
    pick = int(np.argmin(vals.real))
    omega0 = float(np.sqrt(vals[pick].real))
    ez = np.real(vecs[:, pick]).reshape(eps.shape)
    residual = np.linalg.norm(laplace @ ez.ravel() - (omega0**2) * (mass @ ez.ravel()))
    residual /= np.linalg.norm(laplace @ ez.ravel())
    if residual > 1e-8 or not np.isfinite(omega0) or omega0 <= 0.0:
        raise RuntimeError(f"FDFD mode solve failed (residual {residual})")
    peak = int(np.argmax(np.abs(ez)))
    ez = ez * np.sign(ez.ravel()[peak])
    return omega0, np.ascontiguousarray(ez, dtype=np.float64)


def _pole(omega0: float, alpha: float) -> complex:
    rate = float(alpha)
    if rate < 0.0 or not np.isfinite(rate):
        raise ValueError("alpha must be ≥ 0")
    if rate == 0.0:
        return complex(omega0)
    half = 0.5 * rate
    under = omega0**2 - half**2
    if under <= 0.0:
        raise ValueError("conductivity overdamps the resonance")
    return complex(np.sqrt(under), -half)


def _quality(pole: complex, alpha: float) -> float | None:
    if float(alpha) == 0.0:
        return None
    return float(pole.real / alpha)


def _project(sim: YeeFdtd, ez: np.ndarray, steps: int) -> np.ndarray:
    probe = np.asarray(ez, dtype=np.float64).ravel()
    probe = probe / np.linalg.norm(probe)
    series = np.empty(steps, dtype=np.float64)
    for n in range(steps):
        series[n] = float(np.dot(probe, sim.Ez[:, :, 0].ravel()))
        sim.step()
    return series


def _fit_damped_cosine(series: np.ndarray, dt: float) -> tuple[float, float, float]:
    """One-pole Prony fit: ``a_n = ρ^n cos(n θ)``."""
    s0 = series[:-2]
    s1 = series[1:-1]
    s2 = series[2:]
    coeff, *_ = np.linalg.lstsq(np.column_stack((s1, s0)), s2, rcond=None)
    a, b = float(coeff[0]), float(coeff[1])
    if b >= 0.0:
        raise RuntimeError("ringdown is not a damped oscillation")
    rho = float(np.sqrt(-b))
    theta = float(np.arccos(np.clip(a / (2.0 * rho), -1.0, 1.0)))
    if theta <= 0.0 or rho <= 0.0:
        raise RuntimeError("ringdown pole is degenerate")
    pred = a * s1 + b * s0
    residual = float(np.linalg.norm(pred - s2) / (np.linalg.norm(s2) + 1e-30))
    return theta / float(dt), float(np.log(rho) / dt), residual


def _leapfrog_to_spatial(omega: float, dt: float) -> float:
    return float((2.0 / dt) * np.sin(omega * dt / 2.0))


def _wavelength_nm(omega: float) -> float:
    return float(2.0 * np.pi * C_LIGHT / omega * 1e9)
