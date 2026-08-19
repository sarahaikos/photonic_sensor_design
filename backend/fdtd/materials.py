"""Linear and dispersive material models (constant, Debye, Lorentz, Drude).

Time-domain updates use ADE (auxiliary differential equation). PLRC coefficients
are stored for the same poles so a convolutional engine can swap in later.
"""

from __future__ import annotations

from dataclasses import dataclass

C0 = 299792458.0
EPS0 = 8.854187817e-12
MU0 = 1.256637062e-6


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
