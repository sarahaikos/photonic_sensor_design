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

A Lorentz pole makes ε depend on frequency. FDFD rebuilds Re ε(ω) and
resolves the curl-curl mode until ω is consistent with that permittivity.
The decay comes from the energy-weighted Im ε. FDTD steps the same pole
with the auxiliary polarization update.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

from fdfd.operators import yee_operators
from fdtd.materials import EPS0, MU0, _pole_permittivity
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
    xs, ys = _block_slices((nx, ny))
    eps[xs, ys] = float(eps_core)
    return eps


def block_mask(shape: tuple[int, int]) -> np.ndarray:
    """Cells of the dielectric block in ``block_permittivity``."""
    mask = np.zeros((int(shape[0]), int(shape[1])), dtype=np.float64)
    xs, ys = _block_slices(mask.shape)
    mask[xs, ys] = 1.0
    return mask


def permittivity_at(eps_inf, mask, pole, omega: float) -> np.ndarray:
    """ε∞ plus the real Lorentz susceptibility at one frequency, on ``mask``."""
    grid = np.asarray(eps_inf, dtype=np.float64)
    weight = np.asarray(mask, dtype=np.float64)
    if grid.shape != weight.shape:
        raise ValueError("mask shape must match eps_inf")
    chi = _pole_permittivity(pole, float(omega))
    return grid + weight * float(chi.real)


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


def fdfd_dispersive_resonance(
    eps_inf,
    mask,
    pole,
    spacing: float | tuple[float, ...],
    *,
    bc: str | tuple[str, ...] = "pec",
    iterations: int = 24,
) -> dict:
    """TMz resonance whose permittivity is the Lorentz pole at that same ω.

    Each pass rebuilds Re ε(ω) on the masked cells and resolves the real
    curl-curl mode. Iteration stops when that mode's frequency is the ω the
    permittivity was built from. With that mode held fixed, a scalar Newton
    solve then moves ω into the complex plane so ω² ε(ω) stays on the same
    Rayleigh quotient. That complex root is the decay the FDTD ringdown sees.
    """
    grid = np.asarray(eps_inf, dtype=np.float64)
    weight = np.asarray(mask, dtype=np.float64)
    if grid.ndim != 2 or grid.shape != weight.shape:
        raise ValueError("eps_inf and mask must be matching 2D grids")
    if getattr(pole, "kind", None) != "lorentz":
        raise ValueError("dispersive resonance uses a Lorentz pole")

    def _at(query: float):
        built = permittivity_at(grid, weight, pole, query)
        if np.any(built <= 0.0) or not np.isfinite(built).all():
            raise RuntimeError("dispersive permittivity left the positive reals")
        found, mode = _fundamental(built, spacing, bc, sigma_omega=query)
        return found, mode, built

    # Secant on f(ω) = ω_eig(ε(ω)) − ω. The first sample is the nondispersive cavity.
    omega, _bare = _fundamental(grid, spacing, bc)
    eig, ez, eps = _at(omega)
    previous_query, previous_gap = omega, eig - omega
    query = eig
    for _ in range(int(iterations)):
        eig, ez, eps = _at(query)
        gap = eig - query
        if abs(gap) <= 1e-11 * abs(eig):
            omega = eig
            break
        denom = gap - previous_gap
        if abs(denom) <= 1e-30 * abs(gap):
            nxt = 0.5 * (query + eig)
        else:
            nxt = query - gap * (query - previous_query) / denom
        if not np.isfinite(nxt) or nxt <= 0.0 or abs(nxt - query) > 0.5 * query:
            nxt = 0.5 * (query + eig)
        previous_query, previous_gap = query, gap
        query = float(nxt)
    else:
        raise RuntimeError("dispersive resonance did not converge")
    omega, gamma = _complex_pole(grid, weight, pole, eps, ez, omega)
    return {
        "omega": omega,
        "gamma": gamma,
        "wavelength_nm": _wavelength_nm(omega),
        "q": None if abs(gamma) <= 1e-30 * omega else float(omega / (2.0 * abs(gamma))),
        "ez": ez,
        "eps": eps,
    }


def fdtd_dispersive_resonance(
    eps_inf,
    mask,
    pole,
    spacing: float | tuple[float, ...],
    ez,
    *,
    omega_hint: float,
    bc: str | tuple[str, ...] = "pec",
    cfl: float = 0.5,
    steps: int = 1600,
) -> dict:
    """Ringdown of a Lorentz block. ``omega_hint`` only sets the initial polarization."""
    sim = YeeFdtd(eps_inf, spacing, bc=bc, cfl=cfl, poles=(pole,), pole_mask=mask)
    sim.set_electric(ez=ez, cosine_peak=True)
    sim.prime_lorentz(omega_hint)
    series = _project(sim, ez, int(steps))
    omega, gamma, residual = _fit_damped_cosine(series, sim.dt)
    corrected = _leapfrog_to_spatial(omega, sim.dt)
    return {
        "omega": omega,
        "omega_corrected": corrected,
        "gamma": gamma,
        "wavelength_nm": _wavelength_nm(omega),
        "wavelength_corrected_nm": _wavelength_nm(corrected),
        "q": omega / (2.0 * abs(gamma)),
        "q_corrected": corrected / (2.0 * abs(gamma)),
        "fit_residual": residual,
        "dt": sim.dt,
        "steps": int(steps),
    }


def compare_dispersive_resonance(
    eps_inf,
    mask,
    pole,
    spacing: float | tuple[float, ...],
    *,
    frozen_omega: float = 0.0,
    bc: str | tuple[str, ...] = "pec",
    cfl: float = 0.5,
    steps: int = 1600,
) -> dict:
    """Live dispersive resonance against FDTD, and against ε frozen at another ω."""
    live = fdfd_dispersive_resonance(eps_inf, mask, pole, spacing, bc=bc)
    time = fdtd_dispersive_resonance(
        eps_inf,
        mask,
        pole,
        spacing,
        live["ez"],
        omega_hint=live["omega"],
        bc=bc,
        cfl=cfl,
        steps=steps,
    )
    frozen = fdfd_resonance(permittivity_at(eps_inf, mask, pole, frozen_omega), spacing, bc=bc)
    return {"fdfd": live, "fdtd": time, "frozen": frozen}


def _complex_pole(grid, weight, pole, eps, ez, omega: float) -> tuple[float, float]:
    """ω whose complex permittivity keeps the real mode's Rayleigh quotient."""
    energy = float(np.sum(eps * ez**2))
    target = float(omega) ** 2 * energy
    shape = ez**2

    def mismatch(probe: complex) -> complex:
        chi = _pole_permittivity(pole, probe)
        density = np.asarray(grid, dtype=np.float64) + np.asarray(weight, dtype=np.float64) * chi
        return probe**2 * np.sum(density * shape) - target

    chi0 = _pole_permittivity(pole, omega)
    seed = -float(omega) * float(np.sum(np.asarray(weight) * chi0.imag * shape)) / (2.0 * energy)
    probe = complex(float(omega), seed)
    for _ in range(12):
        gap = mismatch(probe)
        if abs(gap) <= 1e-12 * abs(target):
            break
        step = 1e-6 * abs(probe)
        slope = (mismatch(probe + step) - mismatch(probe - step)) / (2.0 * step)
        if slope == 0:
            raise RuntimeError("complex dispersive pole did not converge")
        probe = probe - gap / slope
    else:
        raise RuntimeError("complex dispersive pole did not converge")
    if not np.isfinite(probe.real) or not np.isfinite(probe.imag):
        raise RuntimeError("complex dispersive pole did not converge")
    return float(probe.real), float(probe.imag)


def _block_slices(shape: tuple[int, int]) -> tuple[slice, slice]:
    nx, ny = int(shape[0]), int(shape[1])
    return slice(nx // 5, 4 * nx // 5), slice(ny // 4, 3 * ny // 4)


def _fundamental(eps: np.ndarray, spacing, bc, sigma_omega: float | None = None) -> tuple[float, np.ndarray]:
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
    # A cold start aims below the box resonance. Later passes sit on the last ω.
    if sigma_omega is None:
        shift = (0.35 * guess) ** 2
    else:
        shift = float(sigma_omega) ** 2
    vals, vecs = eigsh(laplace, k=3, M=mass, sigma=shift, which="LM")
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
