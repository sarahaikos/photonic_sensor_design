"""Field excitations: dipole, eigenmode line source, TFSF plane wave."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ModeSource:
    x_index: int
    profile: np.ndarray  # across y (transverse) at a fixed propagation index
    wavelength_nm: float
    t0_steps: int
    width_steps: int
    kind: str = "eigenmode"

    def envelope(self, n: int) -> float:
        w = 2 * np.pi * 299792458.0 / (self.wavelength_nm * 1e-9)
        # dt supplied via call
        return 0.0

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "injection": "eigenmode line",
            "wavelength_nm": self.wavelength_nm,
            "pulse": "gaussian-modulated sinusoid",
            "dipole": False,
            "tfsf": False,
        }


def gaussian_mode(ny: int, y0: float, width: float, y: np.ndarray) -> np.ndarray:
    p = np.exp(-((y - y0) / max(width, 1e-9)) ** 2)
    nrm = np.sqrt(np.sum(p * p)) or 1.0
    return p / nrm


def dipole_j(n: int, dt: float, wavelength_nm: float, t0: float, tau: float) -> float:
    t = n * dt
    w = 2 * np.pi * 299792458.0 / (wavelength_nm * 1e-9)
    return np.exp(-((t - t0) / tau) ** 2) * np.sin(w * t)


def tfsf_plane_wave(n: int, dt: float, wavelength_nm: float, theta_rad: float = 0.0) -> dict:
    """TFSF boundary value; theta is the in-plane incidence (Bloch when used with PBC)."""
    t = n * dt
    w = 2 * np.pi * 299792458.0 / (wavelength_nm * 1e-9)
    return {
        "e": np.sin(w * t),
        "theta_rad": theta_rad,
        "kind": "tfsf",
    }
