"""Simple photodetector readout model."""

from __future__ import annotations

import math

Q_E = 1.602176634e-19
H = 6.62607015e-34
C = 2.99792458e8
K_B = 1.380649e-23


def analyze_detector(
    wavelength_nm: float,
    optical_power_uw: float,
    responsivity_a_per_w: float,
    dark_current_na: float,
    bandwidth_mhz: float,
    load_ohm: float,
) -> dict:
    pin_w = optical_power_uw * 1e-6
    id_a = dark_current_na * 1e-9
    bw_hz = bandwidth_mhz * 1e6

    # Cap responsivity by quantum efficiency ≤ 1
    qe_max = (H * C) / (Q_E * wavelength_nm * 1e-9)
    r = min(responsivity_a_per_w, 1.0 / qe_max) if qe_max > 0 else responsivity_a_per_w

    iph = r * pin_w
    isig = iph
    inoise_shot = math.sqrt(2 * Q_E * (iph + id_a) * bw_hz)
    inoise_thermal = math.sqrt(4 * K_B * 300.0 * bw_hz / max(load_ohm, 1.0))
    inoise = math.hypot(inoise_shot, inoise_thermal)

    snr = isig / inoise if inoise > 0 else 0.0
    snr_db = 20 * math.log10(max(snr, 1e-18))
    nep = inoise / r if r > 0 else None  # W
    min_detectable_uw = (nep * 1e6) if nep else None

    return {
        "responsivity_used": r,
        "photocurrent_ua": iph * 1e6,
        "dark_current_na": dark_current_na,
        "noise_current_na": inoise * 1e9,
        "snr": snr,
        "snr_db": snr_db,
        "nep_pw_per_sqrt_hz": (nep / math.sqrt(bw_hz) * 1e12) if nep and bw_hz > 0 else None,
        "min_detectable_uw": min_detectable_uw,
        "quantum_efficiency": r * qe_max if qe_max > 0 else 0.0,
    }
