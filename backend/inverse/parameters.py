"""Bounded parameters for the two optimization levels.

Device parameters are continuous geometry on a DAG node (gap, length, radius).
System parameters are the outer objective (targets and weights). They share one
spec — name, level, bounds, scale, frozen — so a tuner can pack only the free
device coordinates and still report the system spec it is aiming at.

Values are stored in physical units. The tuner sees a scaled coordinate in
[0, 1], linear or logarithmic between the bounds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from inverse.graph import PhotonicGraph


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    level: str
    value: float
    lower: float
    upper: float
    scale: str = "linear"
    frozen: bool = False
    node_id: str | None = None
    key: str | None = None

    def __post_init__(self) -> None:
        if self.level not in ("device", "system"):
            raise ValueError("level must be 'device' or 'system'")
        if self.scale not in ("linear", "log"):
            raise ValueError("scale must be 'linear' or 'log'")
        if self.upper <= self.lower:
            raise ValueError(f"{self.name}: upper bound must exceed lower bound")
        if self.scale == "log" and self.lower <= 0.0:
            raise ValueError(f"{self.name}: log scale requires a positive lower bound")

    def to_scaled(self, value: float) -> float:
        clipped = min(self.upper, max(self.lower, float(value)))
        if self.scale == "log":
            span = math.log(self.upper) - math.log(self.lower)
            return (math.log(clipped) - math.log(self.lower)) / span
        return (clipped - self.lower) / (self.upper - self.lower)

    def from_scaled(self, scaled: float) -> float:
        unit = min(1.0, max(0.0, float(scaled)))
        if self.scale == "log":
            log_value = math.log(self.lower) + unit * (math.log(self.upper) - math.log(self.lower))
            return math.exp(log_value)
        return self.lower + unit * (self.upper - self.lower)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "level": self.level,
            "node_id": self.node_id,
            "key": self.key,
            "value": float(self.value),
            "lower": float(self.lower),
            "upper": float(self.upper),
            "scale": self.scale,
            "frozen": bool(self.frozen),
            "scaled": float(self.to_scaled(self.value)),
        }


@dataclass(frozen=True)
class _Template:
    key: str
    lower: float
    upper: float
    scale: str = "linear"
    frozen: bool = False


# Coupler gap/length bounds match coupler.py. Waveguide length and detector
# bias are frozen: they do not set the ring's κ. Unlock them per graph to tune.
_CATALOG: dict[str, tuple[_Template, ...]] = {
    "waveguide": (_Template("length_um", 10.0, 2000.0, frozen=True),),
    "coupler": (
        _Template("gap_nm", 50.0, 500.0),
        _Template("length_um", 1.0, 200.0),
    ),
    "ring": (_Template("radius_um", 5.0, 40.0),),
    "heater": (
        _Template("power_mw", 0.0, 40.0, frozen=True),
        _Template("width_um", 0.5, 6.0, frozen=True),
        _Template("length_um", 2.0, 40.0, frozen=True),
    ),
    "detector": (
        _Template("responsivity_a_per_w", 0.05, 1.2, frozen=True),
        _Template("dark_current_na", 0.0, 100.0, frozen=True),
        _Template("bandwidth_mhz", 1.0, 10_000.0, scale="log", frozen=True),
        _Template("load_ohm", 1.0, 10_000.0, scale="log", frozen=True),
    ),
}


class ParameterSet:
    """Device specs read off a graph, plus the frozen system objective specs."""

    def __init__(self, device: list[ParameterSpec], system: list[ParameterSpec]):
        self.device = list(device)
        self.system = list(system)

    @property
    def free_device(self) -> list[ParameterSpec]:
        return [spec for spec in self.device if not spec.frozen]

    def pack(self) -> np.ndarray:
        """Scaled free-device coordinates in [0, 1], topological then catalog order."""
        return np.array(
            [spec.to_scaled(spec.value) for spec in self.free_device],
            dtype=float,
        )

    def apply(self, graph: PhotonicGraph, scaled: np.ndarray) -> PhotonicGraph:
        """Copy ``graph`` with free device parameters written from scaled coordinates."""
        coords = np.asarray(scaled, dtype=float).reshape(-1)
        free = self.free_device
        if coords.shape != (len(free),):
            raise ValueError(f"expected {len(free)} scaled parameters, got {coords.size}")
        out = graph.copy()
        for spec, unit in zip(free, coords):
            out.nodes[spec.node_id].params[spec.key] = spec.from_scaled(float(unit))
        return out

    def sync(self, graph: PhotonicGraph) -> ParameterSet:
        """Device values taken from ``graph``; frozen flags and system specs kept."""
        updated: list[ParameterSpec] = []
        for spec in self.device:
            raw = graph.nodes[spec.node_id].params.get(spec.key, spec.value)
            updated.append(replace(spec, value=float(raw)))
        return ParameterSet(updated, list(self.system))

    def to_dict(self) -> dict:
        return {
            "device": [spec.to_dict() for spec in self.device],
            "system": [spec.to_dict() for spec in self.system],
            "free": [
                {"node_id": spec.node_id, "key": spec.key, "scaled": spec.to_scaled(spec.value)}
                for spec in self.free_device
            ],
        }


def bind(
    graph: PhotonicGraph,
    system: list[ParameterSpec],
    *,
    free: set[tuple[str, str]] | None = None,
    freeze: set[tuple[str, str]] | None = None,
) -> ParameterSet:
    """Collect tunable parameters in topological order.

    ``free``, when given, is the only device set the inner loop may move.
    ``freeze`` and graph locks always win over ``free``.
    """
    freeze = set(freeze or ())
    freeze |= set(graph.locks)
    device: list[ParameterSpec] = []
    for node_id in graph.topological_order():
        node = graph.nodes[node_id]
        for template in _CATALOG.get(node.kind, ()):
            if template.key not in node.params:
                continue
            identity = (node_id, template.key)
            frozen = template.frozen
            if free is not None:
                frozen = identity not in free
            if identity in freeze:
                frozen = True
            device.append(
                ParameterSpec(
                    name=f"{node_id}.{template.key}",
                    level="device",
                    value=float(node.params[template.key]),
                    lower=template.lower,
                    upper=template.upper,
                    scale=template.scale,
                    frozen=frozen,
                    node_id=node_id,
                    key=template.key,
                )
            )
    return ParameterSet(device, system)


def system_specs(objective: object) -> list[ParameterSpec]:
    """Frozen outer-level weights and targets from a ``SystemObjective``."""
    weights = (
        "sensitivity_weight",
        "q_weight",
        "extinction_weight",
        "thermal_lambda_weight",
        "thermal_kappa_weight",
        "lod_weight",
        "kappa_weight",
    )
    targets = (
        ("q_target", 100.0, 1.0e6),
        ("extinction_target_db", 0.0, 40.0),
        ("kappa_target", 1e-4, 0.999),
    )
    specs: list[ParameterSpec] = []
    for name in weights:
        value = float(getattr(objective, name))
        specs.append(
            ParameterSpec(
                name=name,
                level="system",
                value=value,
                lower=0.0,
                upper=max(1.0, value),
                frozen=True,
            )
        )
    for name, lower, upper in targets:
        value = getattr(objective, name)
        if value is None:
            continue
        specs.append(
            ParameterSpec(
                name=name,
                level="system",
                value=float(value),
                lower=lower,
                upper=upper,
                scale="log" if name == "q_target" else "linear",
                frozen=True,
            )
        )
    return specs
