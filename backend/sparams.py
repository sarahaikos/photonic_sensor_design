"""S-parameters and spectrum parameter extraction (FDTD-style post-process)."""

from __future__ import annotations

import math

import numpy as np


def mag_db(value: complex | float) -> float:
    return 20.0 * math.log10(max(abs(value), 1e-12))


def power_db(value: float) -> float:
    return 10.0 * math.log10(max(abs(value), 1e-18))


def pack_s(s11: complex, s21: complex, s31: complex = 0j, s41: complex = 0j) -> dict:
    t_through = float(abs(s21) ** 2)
    t_drop = float(abs(s31) ** 2)
    t_add = float(abs(s41) ** 2)
    t_refl = float(abs(s11) ** 2)
    return {
        "s11": abs(s11),
        "s21": abs(s21),
        "s31": abs(s31),
        "s41": abs(s41),
        "s11_db": mag_db(s11),
        "s21_db": mag_db(s21),
        "s31_db": mag_db(s31),
        "s41_db": mag_db(s41),
        "s21_phase_rad": float(np.angle(s21)),
        "s31_phase_rad": float(np.angle(s31)),
        "t_through": t_through,
        "t_drop": t_drop,
        "t_add": t_add,
        "t_refl": t_refl,
        "t_through_db": power_db(t_through),
        "t_drop_db": power_db(t_drop) if t_drop > 0 else None,
    }


def extract_notch(wavelength_nm: np.ndarray, t_through: np.ndarray, t_drop: np.ndarray | None) -> dict:
    i0 = int(np.argmin(t_through))
    lam0 = float(wavelength_nm[i0])
    t_min = float(t_through[i0])
    t_max = float(np.max(t_through))
    target = t_min + 0.5 * (t_max - t_min)
    left = _crossing(wavelength_nm, t_through, i0, target, -1)
    right = _crossing(wavelength_nm, t_through, i0, target, 1)
    fwhm = max(right - left, 1e-9)
    q_loaded = lam0 / fwhm
    er_db = power_db(t_max / max(t_min, 1e-18))
    il_db = -power_db(t_max)
    t_drop_on = float(t_drop[i0]) if t_drop is not None else None
    t_drop_off = float(np.min(t_drop)) if t_drop is not None else None
    return {
        "resonance_nm": lam0,
        "fwhm_nm": fwhm,
        "q_loaded": q_loaded,
        "extinction_db": er_db,
        "insertion_loss_db": il_db,
        "t_through": t_min,
        "t_through_off": t_max,
        "t_drop": t_drop_on,
        "t_drop_off": t_drop_off,
        "t_drop_db": power_db(t_drop_on) if t_drop_on else None,
        "t_through_db": power_db(t_min),
    }


def _crossing(x: np.ndarray, y: np.ndarray, i0: int, target: float, direction: int) -> float:
    n = len(y)
    i = i0
    while 0 < i < n - 1:
        j = i + direction
        if j < 0 or j >= n:
            break
        yi, yj = float(y[i]), float(y[j])
        if (yi - target) * (yj - target) <= 0:
            if yj == yi:
                return float(x[j])
            t = (target - yi) / (yj - yi)
            return float(x[i] + t * (x[j] - x[i]))
        i = j
    return float(x[0] if direction < 0 else x[-1])
