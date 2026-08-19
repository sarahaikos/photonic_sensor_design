"""On-the-fly DFT monitors, Poynting flux, and mode overlap."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DFTLine:
    name: str
    index: int
    axis: str  # x | y  (normal to the line)
    frequency_hz: float
    real: np.ndarray
    imag: np.ndarray

    def accumulate(self, field_line: np.ndarray, n: int, dt: float) -> None:
        phase = 2 * np.pi * self.frequency_hz * n * dt
        self.real += field_line * np.cos(phase)
        self.imag += field_line * np.sin(phase)

    def complex(self) -> np.ndarray:
        return self.real + 1j * self.imag


def poynting_power(e: np.ndarray, h: np.ndarray, dl: float) -> float:
    """∫ Re(E × H*) / 2 along a monitor line."""
    return float(0.5 * np.sum(np.real(e * np.conj(h))) * dl)


def overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Mode overlap |⟨a|b⟩|² / (|a|²|b|²)."""
    num = np.abs(np.vdot(a, b)) ** 2
    den = (np.vdot(a, a).real * np.vdot(b, b).real) + 1e-30
    return float(num / den)


def far_field_hint() -> dict:
    return {
        "method": "Stratton-Chu",
        "enabled": False,
        "note": "Near-to-far uses equivalent currents on a closed DFT box; not needed for on-chip S-params.",
    }
