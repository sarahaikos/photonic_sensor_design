"""Single-frequency gold nanorods.

At one frequency a linear dispersion model is a complex permittivity. Gans
theory inserts that number into the quasi-static polarizability of a prolate
spheroid. The sphere limit is checked against the Mie series, which is the
same permittivity inside the full-wave frequency-domain solution.

Time convention matches ``permittivity``: e^{-iωt}, outgoing spherical Hankel
function of the first kind.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import spherical_jn, spherical_yn

from fdtd.materials import C0, GOLD, permittivity

# Longitudinal resonance search. Aspect ratios 1–5 in water fall inside it.
_WAVE_NM = np.linspace(400.0, 1200.0, 1601)


def depolarization_prolate(aspect: float) -> float:
    """Longitudinal depolarization L of a prolate spheroid, aspect a/b ≥ 1.

    The transverse factor is ``(1 - L) / 2``. A sphere is L = 1/3.
    """
    ratio = float(aspect)
    if not math.isfinite(ratio) or ratio < 1.0:
        raise ValueError("aspect ratio a/b must be ≥ 1")
    if ratio < 1.0 + 1e-12:
        return 1.0 / 3.0
    ecc_sq = 1.0 - 1.0 / ratio**2
    ecc = math.sqrt(ecc_sq)
    return (1.0 - ecc_sq) / ecc_sq * (-1.0 + math.log((1.0 + ecc) / (1.0 - ecc)) / (2.0 * ecc))


def polarizability(eps: complex, eps_medium: float, depolarization: float, volume: float) -> complex:
    """Quasi-static polarizability volume α, with p = ε0 ε_m α E.

    α = V (ε − ε_m) / (ε_m + L (ε − ε_m)).
    """
    contrast = eps - eps_medium
    return volume * contrast / (eps_medium + depolarization * contrast)


def absorption_cross_section(
    eps: complex,
    eps_medium: float,
    depolarization: float,
    volume: float,
    wavelength_m: float,
) -> float:
    """C_abs = k Im(α) for a particle much smaller than the wavelength."""
    k = 2.0 * math.pi * math.sqrt(eps_medium) / float(wavelength_m)
    alpha = polarizability(eps, eps_medium, depolarization, volume)
    return float(k * np.imag(alpha))


def spheroid_volume(aspect: float, semi_minor_m: float) -> float:
    semi_major = float(aspect) * float(semi_minor_m)
    return (4.0 / 3.0) * math.pi * semi_major * float(semi_minor_m) ** 2


def angular_frequency(wavelength_m: float) -> float:
    return 2.0 * math.pi * C0 / float(wavelength_m)


def lspr(
    aspect: float,
    semi_minor_m: float,
    n_medium: float,
    *,
    longitudinal: bool = True,
) -> dict:
    """Peak of the single-frequency absorption spectrum.

    Each wavelength uses ``permittivity(GOLD, ω)`` at that wavelength only.
    """
    if float(semi_minor_m) <= 0.0 or float(n_medium) <= 0.0:
        raise ValueError("semi-minor axis and medium index must be positive")
    long_l = depolarization_prolate(aspect)
    factor = long_l if longitudinal else 0.5 * (1.0 - long_l)
    medium = float(n_medium) ** 2
    volume = spheroid_volume(aspect, semi_minor_m)
    waves_m = _WAVE_NM * 1e-9
    section = np.empty(_WAVE_NM.size)
    eps = np.empty(_WAVE_NM.size, dtype=np.complex128)
    for i, wave_m in enumerate(waves_m):
        eps[i] = permittivity(GOLD, angular_frequency(wave_m))
        section[i] = absorption_cross_section(eps[i], medium, factor, volume, wave_m)
    peak = int(np.argmax(section))
    if peak == 0 or peak == section.size - 1:
        raise RuntimeError("LSPR is outside 400–1200 nm")
    step = float(_WAVE_NM[1] - _WAVE_NM[0])
    # Parabolic refinement of the sampled maximum.
    y0, y1, y2 = section[peak - 1], section[peak], section[peak + 1]
    shift = 0.5 * (y0 - y2) / (y0 - 2.0 * y1 + y2)
    wave_nm = float(_WAVE_NM[peak] + shift * step)
    eps_peak = permittivity(GOLD, angular_frequency(wave_nm * 1e-9))
    c_abs = absorption_cross_section(
        eps_peak, medium, factor, volume, wave_nm * 1e-9
    )
    return {
        "aspect": float(aspect),
        "semi_minor_m": float(semi_minor_m),
        "n_medium": float(n_medium),
        "longitudinal": bool(longitudinal),
        "depolarization": float(factor),
        "lspr_nm": wave_nm,
        "eps": eps_peak,
        "target_eps": medium * (1.0 - 1.0 / factor),
        "c_abs_m2": c_abs,
    }


def mie_cross_sections(
    eps: complex,
    eps_medium: float,
    radius_m: float,
    wavelength_m: float,
) -> tuple[float, float, float]:
    """Mie (C_ext, C_sca, C_abs) for a sphere of permittivity ``eps``.

    ``eps`` is the single-frequency value. Bohren & Huffman, e^{-iωt}.
    """
    if float(radius_m) <= 0.0 or float(wavelength_m) <= 0.0 or float(eps_medium) <= 0.0:
        raise ValueError("radius, wavelength, and medium permittivity must be positive")
    n_medium = math.sqrt(float(eps_medium))
    k = 2.0 * math.pi * n_medium / float(wavelength_m)
    size = k * float(radius_m)
    index = np.sqrt(complex(eps) / float(eps_medium))
    if np.real(index) < 0.0:
        index = -index
    n_stop = max(2, int(math.ceil(size + 4.0 * size ** (1.0 / 3.0) + 2.0)))
    extinct = 0.0
    scatter = 0.0
    for order in range(1, n_stop + 1):
        electric, magnetic = _mie_coefficients(order, index, size)
        weight = 2.0 * order + 1.0
        extinct += weight * float(np.real(electric + magnetic))
        scatter += weight * (abs(electric) ** 2 + abs(magnetic) ** 2)
    scale = 2.0 * math.pi / k**2
    c_ext = scale * extinct
    c_sca = scale * scatter
    return c_ext, c_sca, c_ext - c_sca


def sphere_comparison(
    radius_m: float,
    wavelength_m: float,
    n_medium: float = 1.333,
) -> dict:
    """Gans absorption against Mie at one frequency, for a gold sphere."""
    medium = float(n_medium) ** 2
    eps = permittivity(GOLD, angular_frequency(wavelength_m))
    volume = spheroid_volume(1.0, radius_m)
    gans = absorption_cross_section(eps, medium, 1.0 / 3.0, volume, wavelength_m)
    _ext, _sca, mie = mie_cross_sections(eps, medium, radius_m, wavelength_m)
    return {
        "wavelength_nm": float(wavelength_m) * 1e9,
        "radius_nm": float(radius_m) * 1e9,
        "eps": eps,
        "gans_m2": gans,
        "mie_m2": mie,
        "relative_error": abs(mie - gans) / abs(gans),
    }


def survey_nanorods(
    aspects: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0),
    semi_minor_m: float = 10e-9,
    n_medium: float = 1.333,
) -> list[dict]:
    """Longitudinal and transverse LSPRs, each from ε(ω) at that wavelength."""
    rows = []
    for aspect in aspects:
        long = lspr(aspect, semi_minor_m, n_medium, longitudinal=True)
        trans = lspr(aspect, semi_minor_m, n_medium, longitudinal=False)
        row = {"longitudinal": long, "transverse": trans}
        if abs(float(aspect) - 1.0) < 1e-12:
            row["mie"] = sphere_comparison(semi_minor_m, long["lspr_nm"] * 1e-9, n_medium)
        rows.append(row)
    return rows


def _mie_coefficients(order: int, index: complex, size: float) -> tuple[complex, complex]:
    psi_x, dpsi_x, xi_x, dxi_x = _riccati(order, size)
    psi_m, dpsi_m, _xi_m, _dxi_m = _riccati(order, index * size)
    electric = (index * psi_m * dpsi_x - psi_x * dpsi_m) / (index * psi_m * dxi_x - xi_x * dpsi_m)
    magnetic = (psi_m * dpsi_x - index * psi_x * dpsi_m) / (psi_m * dxi_x - index * xi_x * dpsi_m)
    return electric, magnetic


def _riccati(order: int, argument: complex) -> tuple[complex, complex, complex, complex]:
    jn = spherical_jn(order, argument)
    yn = spherical_yn(order, argument)
    jn_m = spherical_jn(order - 1, argument)
    yn_m = spherical_yn(order - 1, argument)
    hankel = jn + 1j * yn
    hankel_m = jn_m + 1j * yn_m
    psi = argument * jn
    xi = argument * hankel
    dpsi = argument * jn_m - order * jn
    dxi = argument * hankel_m - order * hankel
    return psi, dpsi, xi, dxi
