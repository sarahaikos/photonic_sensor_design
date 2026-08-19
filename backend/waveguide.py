"""SOI strip waveguide using FDTD-fitted compact models."""

from __future__ import annotations

import math

from fdtd_soi import mode_fdtd
from fields import waveguide_field
from sparams import pack_s


def analyze_waveguide(
    width_nm: float,
    height_nm: float,
    wavelength_nm: float,
    n_clad: float,
    polarization: str,
    length_um: float = 100.0,
    include_field: bool = True,
) -> dict | None:
    if width_nm < 250 or height_nm < 120:
        return None
    mode = mode_fdtd(width_nm, height_nm, wavelength_nm, n_clad, polarization)
    n_eff = mode["n_eff"]
    n_g = mode["n_g"]
    if n_eff <= max(n_clad, 1.444):
        return None

    loss = mode["loss_db_per_cm"]
    length_cm = length_um * 1e-4
    loss_db = loss * length_cm
    transmission = 10 ** (-loss_db / 10)
    delay_ps = n_g * length_um / 299.792458
    beta = 2 * math.pi * n_eff / (wavelength_nm * 1e-3)
    s21 = transmission**0.5 * complex(math.cos(beta * length_um), -math.sin(beta * length_um))
    s = pack_s(0j, s21)

    out = {
        "n_eff": n_eff,
        "n_g": n_g,
        "dn_eff_dn": mode["dn_eff_dn"],
        "loss_db_per_cm": loss,
        "gamma_core": mode["gamma_core"],
        "gamma_clad": mode["gamma_clad"],
        "length_um": length_um,
        "loss_db": loss_db,
        "transmission": transmission,
        "t_through": s["t_through"],
        "delay_ps": delay_ps,
        "s_parameters": s,
        "source": "fdtd-fitted",
    }
    if include_field:
        out["field"] = waveguide_field(width_nm, height_nm, mode["gamma_core"], polarization)
    return out
