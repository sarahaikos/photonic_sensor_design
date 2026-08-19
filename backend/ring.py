"""Ring resonator using FDTD-fitted waveguide/coupler parameters plus bend radiation."""

from __future__ import annotations

import math

import numpy as np

from fdtd_soi import bend_loss_db_per_cm
from sparams import extract_notch, pack_s


def analyze_ring(
    radius_um: float,
    wavelength_nm: float,
    n_eff: float,
    n_g: float,
    kappa: float,
    loss_db_per_cm: float,
    dn_eff_dn: float,
    config: str,
    width_nm: float = 450.0,
    polarization: str = "TE",
) -> dict:
    L_um = 2 * math.pi * radius_um
    L_cm = L_um * 1e-4
    lam_um = wavelength_nm * 1e-3

    loss_total = loss_db_per_cm + bend_loss_db_per_cm(radius_um, width_nm, polarization)
    round_trip_db = loss_total * L_cm
    a = 10 ** (-round_trip_db / 20)
    t = math.sqrt(max(0.0, 1.0 - kappa))

    fsr_nm = (lam_um**2) / (n_g * L_um) * 1e3
    mode_order = round(n_eff * L_um / lam_um)
    resonance_nm = n_eff * L_um / mode_order * 1e3 if mode_order else wavelength_nm

    if config == "add-drop":
        q = _q_factor(n_g, L_um, resonance_nm, a * t * t)
        s11, s21, s31, s41 = _s_add_drop(a, t, 0.0)
        coupling = _add_drop_regime(a, t)
    else:
        q = _q_factor(n_g, L_um, resonance_nm, a * t)
        s11, s21, s31, s41 = _s_all_pass(a, t, 0.0)
        coupling = _all_pass_regime(a, t)

    s_on = pack_s(s11, s21, s31, s41)
    fwhm_nm = resonance_nm / q if q else 0.0
    finesse = fsr_nm / fwhm_nm if fwhm_nm else 0.0
    sensitivity = resonance_nm * dn_eff_dn / n_g if n_g else 0.0
    lod = resonance_nm / (10 * q * sensitivity) if q and sensitivity else None

    ln_a = max(1e-12, -math.log(max(a, 1e-12)))
    q_i = 2 * math.pi * n_g * L_um / (lam_um * 2 * ln_a) if ln_a else None
    # Q_coupling: round-trip power coupling κ_rt.
    # All-pass: one coupler, κ_rt = κ.
    # Add-drop (symmetric): two equal couplers, κ_rt = 2κ = 2(1 - t²).
    k_rt = kappa if config == "all-pass" else min(0.999, 2.0 * (1.0 - t * t))
    q_c = 2 * math.pi * n_g * L_um / (lam_um * max(k_rt, 1e-12))

    spectrum = _spectrum(resonance_nm, fsr_nm, n_eff, n_g, L_um, a, t, config)
    extracted = extract_notch(
        np.array(spectrum["wavelength_nm"]),
        np.array(spectrum["t_through"]),
        np.array(spectrum["t_drop"]) if spectrum.get("t_drop") else None,
    )
    extracted.update(
        {
            "q_intrinsic": q_i,
            "q_coupling": q_c,
            "kappa": kappa,
            "self_coupling_t": t,
            "fsr_nm": fsr_nm,
            "finesse": extracted["q_loaded"] * fsr_nm / resonance_nm if resonance_nm else None,
        }
    )

    return {
        "circumference_um": L_um,
        "round_trip_loss_db": round_trip_db,
        "round_trip_amplitude": a,
        "self_coupling": t,
        "kappa": kappa,
        "mode_order": int(mode_order),
        "resonance_nm": resonance_nm,
        "fsr_nm": fsr_nm,
        "q": q,
        "q_intrinsic": q_i,
        "q_coupling": q_c,
        "fwhm_nm": fwhm_nm,
        "finesse": finesse,
        "extinction_db": extracted["extinction_db"],
        "coupling_regime": coupling,
        "sensitivity_nm_per_riu": sensitivity,
        "lod_riu": lod,
        "t_through": s_on["t_through"],
        "t_drop": s_on["t_drop"] if config == "add-drop" else None,
        "t_through_db": s_on["t_through_db"],
        "t_drop_db": s_on["t_drop_db"] if config == "add-drop" else None,
        "s_parameters": s_on,
        "extracted": extracted,
        "spectrum": spectrum,
        "loss_db_per_cm": loss_total,
        "bend_loss_db_per_cm": loss_total - loss_db_per_cm,
        "source": "fdtd-fitted",
    }


def radius_for_laser(n_eff: float, wavelength_nm: float, radius_guess_um: float) -> float:
    """Radius that puts a ring mode on the laser wavelength."""
    lam_um = wavelength_nm * 1e-3
    m = max(1, round(n_eff * 2 * math.pi * max(radius_guess_um, 1.0) / lam_um))
    return min(80.0, max(1.0, m * lam_um / (2 * math.pi * n_eff)))


def transmission_at_laser(
    radius_um: float,
    wavelength_nm: float,
    n_eff: float,
    kappa: float,
    loss_db_per_cm: float,
    config: str,
    width_nm: float,
    polarization: str,
) -> tuple[float, float]:
    L_um = 2 * math.pi * radius_um
    loss_total = loss_db_per_cm + bend_loss_db_per_cm(radius_um, width_nm, polarization)
    a = 10 ** (-(loss_total * L_um * 1e-4) / 20)
    t = math.sqrt(max(0.0, 1.0 - kappa))
    phase = 2 * math.pi * n_eff * L_um / (wavelength_nm * 1e-3)
    if config == "add-drop":
        _, s21, s31, _ = _s_add_drop(a, t, phase)
    else:
        _, s21, s31, _ = _s_all_pass(a, t, phase)
    return float(abs(s21) ** 2), float(abs(s31) ** 2)


def _q_factor(n_g: float, L_um: float, lam_nm: float, r: float) -> float:
    r = min(max(r, 1e-12), 0.999999)
    lam_um = lam_nm * 1e-3
    return math.pi * n_g * L_um * math.sqrt(r) / (lam_um * (1 - r))


def _s_all_pass(a: float, t: float, phase: float) -> tuple[complex, complex, complex, complex]:
    ej = np.exp(1j * phase)
    s21 = (t - a * ej) / (1 - a * t * ej)
    return 0j, complex(s21), 0j, 0j


def _s_add_drop(a: float, t: float, phase: float) -> tuple[complex, complex, complex, complex]:
    ej = np.exp(1j * phase)
    denom = 1 - a * t * t * ej
    s21 = t * (1 - a * ej) / denom
    s31 = -(1 - t * t) * math.sqrt(a) * np.exp(1j * phase / 2) / denom
    return 0j, complex(s21), complex(s31), 0j


def _all_pass_regime(a: float, t: float) -> str:
    if abs(a - t) < 0.01:
        return "critical"
    return "overcoupled" if t < a else "undercoupled"


def kappa_for_critical(a: float, config: str) -> float:
    """Power coupling that matches round-trip loss (critical coupling)."""
    a = min(0.999999, max(1e-6, a))
    t = math.sqrt(a) if config == "add-drop" else a
    return min(0.95, max(1e-4, 1.0 - t * t))


def _add_drop_regime(a: float, t: float) -> str:
    target = t * t
    if abs(a - target) < 0.01:
        return "critical"
    return "overcoupled" if a > target else "undercoupled"


def _spectrum(
    resonance_nm: float,
    fsr_nm: float,
    n_eff: float,
    n_g: float,
    L_um: float,
    a: float,
    t: float,
    config: str,
) -> dict:
    span = max(fsr_nm * 1.6, 0.4)
    coarse = np.linspace(resonance_nm - span / 2, resonance_nm + span / 2, 400)
    half = min(span / 80.0, 0.08)
    dense = np.linspace(resonance_nm - half, resonance_nm + half, 160)
    wavelengths = np.unique(np.sort(np.concatenate([coarse, dense, np.array([resonance_nm])])))
    t_through = []
    t_drop = []
    s21_db = []
    s31_db = []
    for lam_nm in wavelengths:
        n = n_eff + (n_eff - n_g) / resonance_nm * (lam_nm - resonance_nm)
        phase = 2 * math.pi * n * L_um / (lam_nm * 1e-3)
        if config == "add-drop":
            _, s21, s31, _ = _s_add_drop(a, t, phase)
        else:
            _, s21, s31, _ = _s_all_pass(a, t, phase)
        t_through.append(float(abs(s21) ** 2))
        t_drop.append(float(abs(s31) ** 2))
        s21_db.append(20.0 * math.log10(max(abs(s21), 1e-12)))
        s31_db.append(20.0 * math.log10(max(abs(s31), 1e-12)))
    result = {
        "wavelength_nm": wavelengths.tolist(),
        "through": t_through,
        "t_through": t_through,
        "s21_db": s21_db,
        "s31_db": s31_db,
    }
    if config == "add-drop":
        result["drop"] = t_drop
        result["t_drop"] = t_drop
    return result
