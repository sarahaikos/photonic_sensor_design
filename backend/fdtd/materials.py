"""Linear and dispersive material models (constant, Debye, Lorentz, Drude).

Time-domain updates use ADE (auxiliary differential equation). PLRC coefficients
are stored for the same poles so a convolutional engine can swap in later.
``permittivity`` evaluates those poles at one real frequency (e^{-iωt}).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

C0 = 299792458.0
EPS0 = 8.854187817e-12
MU0 = 1.256637062e-6
# ħ in eV·s. ω (rad/s) = E (eV) / HBAR_EV_S.
HBAR_EV_S = 6.582119569509e-16


@dataclass
class Pole:
    kind: str  # debye | lorentz | drude
    eps_delta: float
    omega0: float
    gamma: float


@dataclass
class Material:
    name: str
    n: float
    model: str
    eps_inf: float
    poles: tuple[Pole, ...] = ()
    anisotropic: bool = False
    nonlinear: bool = False
    gain: bool = False

    @property
    def eps_r(self) -> float:
        return self.n * self.n


# Palik / typical 1550 nm SOI values. Lorentz poles are ADE-ready.
MATERIALS = {
    "si": Material(
        name="si",
        n=3.476,
        model="lorentz",
        eps_inf=7.768,
        poles=(Pole("lorentz", 4.31, 6.02e15, 4.5e13),),
    ),
    "sio2": Material(
        name="sio2",
        n=1.444,
        model="constant",
        eps_inf=2.085,
    ),
    "water": Material(
        name="water",
        n=1.318,
        model="debye",
        eps_inf=1.72,
        poles=(Pole("debye", 0.02, 0.0, 1.0e13),),
    ),
    "air": Material(name="air", n=1.0, model="constant", eps_inf=1.0),
    "ge": Material(
        name="ge",
        n=4.275,
        model="lorentz",
        eps_inf=8.5,
        poles=(Pole("lorentz", 9.8, 5.1e15, 1.2e14),),
        gain=False,
    ),
}


def rad_per_s(energy_ev: float) -> float:
    """Angular frequency for a photon energy in eV."""
    return float(energy_ev) / HBAR_EV_S


def permittivity(material: Material, omega: float) -> complex:
    """Relative permittivity at one real frequency.

    Time convention is e^{-iωt}, so a passive pole has Im ε ≥ 0.
    A monochromatic solve uses this complex number in place of the pole sum.

    Lorentz: ε∞ + Δε ω0² / (ω0² − ω² − iγω)
    Drude:   ε∞ − ωp² / (ω² + iγω), with ωp stored in ``Pole.omega0``
    Debye:   ε∞ + Δε / (1 − iω/γ)
    """
    w = float(omega)
    if not math.isfinite(w) or w < 0.0:
        raise ValueError("omega must be real and ≥ 0")
    eps = complex(material.eps_inf)
    for pole in material.poles:
        eps += _pole_permittivity(pole, w)
    return eps


def _pole_permittivity(pole: Pole, omega: float) -> complex:
    if pole.kind == "lorentz":
        w0, g, de = pole.omega0, pole.gamma, pole.eps_delta
        return de * w0 * w0 / (w0 * w0 - omega * omega - 1j * g * omega)
    if pole.kind == "drude":
        if omega == 0.0:
            raise ValueError("Drude permittivity diverges at ω = 0")
        wp = pole.omega0
        return -(wp * wp) / (omega * omega + 1j * pole.gamma * omega)
    if pole.kind == "debye":
        return pole.eps_delta / (1.0 - 1j * omega / pole.gamma)
    raise ValueError(f"unknown pole kind {pole.kind!r}")


def material_for_clad(n_clad: float) -> Material:
    if n_clad < 1.15:
        return MATERIALS["air"]
    if n_clad < 1.40:
        return MATERIALS["water"]
    return MATERIALS["sio2"]


def ade_lorentz_coeff(pole: Pole, dt: float) -> dict:
    """ADE coefficients for P^{n+1} = C1 P^n + C2 P^{n-1} + C3 E^{n+1}."""
    w0, g, de = pole.omega0, pole.gamma, pole.eps_delta
    den = 1.0 + g * dt + 0.25 * (w0 * dt) ** 2
    return {
        "kind": pole.kind,
        "c1": (2.0 - 0.5 * (w0 * dt) ** 2) / den,
        "c2": -(1.0 - g * dt + 0.25 * (w0 * dt) ** 2) / den,
        "c3": de * EPS0 * (w0 * dt) ** 2 / den,
    }


def catalog() -> list[dict]:
    rows = []
    for m in MATERIALS.values():
        rows.append(
            {
                "name": m.name,
                "n": m.n,
                "model": m.model,
                "eps_inf": m.eps_inf,
                "poles": len(m.poles),
                "anisotropic": m.anisotropic,
                "nonlinear": m.nonlinear,
                "gain": m.gain,
            }
        )
    return rows


def _rakic_gold() -> Material:
    """Lorentz–Drude gold, A. D. Rakić et al., Appl. Opt. 37, 5271 (1998).

    ``n`` is unused: Re ε is negative through the visible and near infrared,
    so the optical response is ``permittivity``, not ``n²``.
    """
    plasma_ev = 9.03
    # (oscillator strength, Γ in eV, ω0 in eV); the Drude weight is f0 below.
    lorentz = (
        (0.024, 0.241, 0.415),
        (0.010, 0.345, 0.830),
        (0.071, 0.870, 2.969),
        (0.601, 2.494, 4.304),
        (4.384, 2.214, 13.32),
    )
    poles = [
        Pole(
            "drude",
            1.0,
            rad_per_s((0.760**0.5) * plasma_ev),
            rad_per_s(0.053),
        )
    ]
    for strength, gamma_ev, omega_ev in lorentz:
        poles.append(
            Pole(
                "lorentz",
                strength * (plasma_ev / omega_ev) ** 2,
                rad_per_s(omega_ev),
                rad_per_s(gamma_ev),
            )
        )
    return Material(name="au", n=0.0, model="drude-lorentz", eps_inf=1.0, poles=tuple(poles))


# Not in MATERIALS: chip rasterization still uses a real n².
GOLD = _rakic_gold()
