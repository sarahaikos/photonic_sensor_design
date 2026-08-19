"""Directional coupler fitted to FDTD of 220 nm SOI strips."""

from __future__ import annotations

import math

from fdtd_soi import coupler_kappa0
from fields import coupler_monitors
from sparams import pack_s

GAP_MIN_NM = 50.0
GAP_MAX_NM = 500.0
LENGTH_MIN_UM = 1.0
LENGTH_MAX_UM = 200.0


def analyze_coupler(
    gap_nm: float,
    length_um: float,
    width_nm: float,
    wavelength_nm: float,
    polarization: str,
    height_nm: float = 220.0,
) -> dict:
    kappa0 = coupler_kappa0(gap_nm, width_nm, height_nm, wavelength_nm, polarization)
    phase = kappa0 * length_um
    s21 = math.cos(phase)
    s31 = -1j * math.sin(phase)
    s = pack_s(0j, s21, s31, 0j)
    kappa = s["t_drop"]
    beat_um = (math.pi / 2) / kappa0 if kappa0 > 1e-9 else None
    monitors = coupler_monitors(
        gap_nm, length_um, width_nm, height_nm, wavelength_nm, polarization, kappa0
    )

    return {
        "kappa": kappa,
        "through": s["t_through"],
        "cross": s["t_drop"],
        "t_through": s["t_through"],
        "t_drop": s["t_drop"],
        "t_through_db": s["t_through_db"],
        "t_drop_db": s["t_drop_db"],
        "s_parameters": s,
        "extracted": {
            "kappa": kappa,
            "self_coupling_t": abs(s21),
            "cross_coupling": abs(s31),
            "beat_length_um": beat_um,
            "imbalance_db": s["t_through_db"] - (s["t_drop_db"] or 0.0),
        },
        "kappa0_per_um": kappa0,
        "beat_length_um": beat_um,
        "gap_nm": gap_nm,
        "length_um": length_um,
        "source": "fdtd-fitted",
        **monitors,
    }


def geometry_for_kappa(
    target_kappa: float,
    gap_nm: float,
    length_um: float,
    width_nm: float,
    height_nm: float,
    wavelength_nm: float,
    polarization: str,
) -> dict:
    """Gap at fixed length, and length at fixed gap, that give target κ = sin²(κ0 L)."""
    k = min(0.999, max(1e-6, target_kappa))
    phase = math.asin(math.sqrt(k))
    args = (width_nm, height_nm, wavelength_nm, polarization)
    kappa0_now = coupler_kappa0(gap_nm, *args)
    length_um_out = phase / max(kappa0_now, 1e-9)
    length_ok = LENGTH_MIN_UM <= length_um_out <= LENGTH_MAX_UM
    length_um_out = min(LENGTH_MAX_UM, max(LENGTH_MIN_UM, length_um_out))

    need = phase / max(length_um, 0.1)
    k0_lo = coupler_kappa0(GAP_MIN_NM, *args)
    k0_hi = coupler_kappa0(GAP_MAX_NM, *args)
    gap_nm_out = None
    gap_ok = False
    if k0_lo >= need >= k0_hi:
        lo, hi = GAP_MIN_NM, GAP_MAX_NM
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if coupler_kappa0(mid, *args) > need:
                lo = mid
            else:
                hi = mid
        gap_nm_out = 0.5 * (lo + hi)
        gap_ok = True

    if gap_ok:
        preferred = "gap"
    elif length_ok:
        preferred = "length"
    else:
        preferred = "length"

    return {
        "kappa": k,
        "gap_nm": gap_nm_out,
        "length_um": length_um_out,
        "preferred": preferred,
        "reachable": gap_ok or length_ok,
    }
