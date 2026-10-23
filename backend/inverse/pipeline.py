"""Dual-level inverse design on a component DAG.

Outer level: choose a circuit DAG (which components, which ports, which
parameters are locked). Inner level: Adam on the free device parameters.
The forward map is the same compact ring and coupler model as the circuit
endpoint — mode indices, κ(gap, length), Q, sensitivity, and uniform-ΔT
corners — without rasterized monitor fields.

Gradients are forward differences. The compact models are ordinary Python,
not an autograd tape. Grid-level adjoints stay in the topology and ceviche
solvers.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fdtd_soi import bend_loss_db_per_cm, coupler_kappa0, mode_fdtd
from inverse.graph import PhotonicGraph, sensor_graph
from inverse.parameters import ParameterSet, bind, system_specs
from ring import analyze_ring
from thermo import (
    CORNERS_K,
    dn_eff_dt,
    kappa0_at,
    kappa_from_kappa0,
    resonance_shift_nm,
)


def _load_adam():
    """Adam from ceviche, without importing the FDFD package init (needs scipy)."""
    path = Path(__file__).resolve().parents[1] / "ceviche" / "optimizers.py"
    spec = importlib.util.spec_from_file_location("_photonic_adam", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.adam_optimize


adam_optimize = _load_adam()

_FD_EPS = 1e-3
_SENSITIVITY_SCALE = 100.0  # nm/RIU
_THERMAL_LAMBDA_SCALE = 1.0  # nm
_THERMAL_KAPPA_SCALE = 0.05
_LOD_SCALE = 1e-4


@dataclass(frozen=True)
class Stack:
    """Fixed cross-section and wavelength. These are not inner-loop variables."""

    width_nm: float = 450.0
    height_nm: float = 220.0
    wavelength_nm: float = 1550.0
    n_clad: float = 1.33
    polarization: str = "TE"

    def __post_init__(self) -> None:
        if self.polarization not in ("TE", "TM"):
            raise ValueError("polarization must be 'TE' or 'TM'")
        if min(self.width_nm, self.height_nm, self.wavelength_nm, self.n_clad) <= 0:
            raise ValueError("stack dimensions must be positive")


@dataclass(frozen=True)
class SystemObjective:
    """Outer-level figure of merit. Minimized. Sensitivity enters with a minus sign.

    Default is a κ target. Turn on Q, extinction, sensitivity, LOD, or the
    worst ambient |Δλ| / |Δκ| when those are the sensor spec.
    """

    sensitivity_weight: float = 0.0
    q_weight: float = 0.0
    q_target: float | None = None
    extinction_weight: float = 0.0
    extinction_target_db: float = 10.0
    thermal_lambda_weight: float = 0.0
    thermal_kappa_weight: float = 0.0
    lod_weight: float = 0.0
    kappa_weight: float = 1.0
    kappa_target: float | None = 0.1


def forward(graph: PhotonicGraph, stack: Stack) -> dict:
    """Compact metrics for one DAG. ``error`` is set when the mode is cut off."""
    mode = mode_fdtd(
        stack.width_nm,
        stack.height_nm,
        stack.wavelength_nm,
        stack.n_clad,
        stack.polarization,
    )
    if mode["n_eff"] <= max(stack.n_clad, 1.444):
        return {"error": "No guided mode for this cross-section"}

    rings = [node for node in graph.nodes.values() if node.kind == "ring"]
    if len(rings) > 1:
        return {"error": "sensor DAG expects one ring"}
    ring = rings[0] if rings else None
    coupler_id = graph.bus_coupler_id()
    kappa = None
    kappa0 = None
    gap_nm = length_um = None
    if coupler_id is not None:
        params = graph.nodes[coupler_id].params
        gap_nm = float(params["gap_nm"])
        length_um = float(params["length_um"])
        kappa0 = coupler_kappa0(
            gap_nm,
            stack.width_nm,
            stack.height_nm,
            stack.wavelength_nm,
            stack.polarization,
        )
        kappa = min(0.999, max(1e-6, kappa_from_kappa0(kappa0, length_um)))

    metrics: dict = {
        "error": None,
        "kappa": kappa,
        "gap_nm": gap_nm,
        "coupler_length_um": length_um,
        "n_eff": float(mode["n_eff"]),
        "n_g": float(mode["n_g"]),
        "q": None,
        "sensitivity_nm_per_riu": None,
        "extinction_db": None,
        "lod_riu": None,
        "resonance_nm": None,
        "coupling_regime": None,
        "worst_abs_dlambda_nm": None,
        "worst_abs_dkappa": None,
    }
    if ring is None or kappa is None:
        return metrics

    dndt = dn_eff_dt(mode["gamma_core"], mode["gamma_clad"], stack.n_clad)
    ring_result = analyze_ring(
        radius_um=float(ring.params["radius_um"]),
        wavelength_nm=stack.wavelength_nm,
        n_eff=float(mode["n_eff"]),
        n_g=float(mode["n_g"]),
        kappa=kappa,
        loss_db_per_cm=float(mode["loss_db_per_cm"]),
        dn_eff_dn=float(mode["dn_eff_dn"]),
        config=ring.params.get("config", "all-pass"),
        width_nm=stack.width_nm,
        polarization=stack.polarization,
    )
    thermal = _ambient_corners(
        stack,
        kappa0=kappa0,
        length_um=length_um,
        kappa_design=kappa,
        resonance_nm=float(ring_result["resonance_nm"]),
        n_g=float(mode["n_g"]),
        dndt=dndt,
    )
    metrics.update(
        {
            "q": float(ring_result["q"]),
            "q_intrinsic": float(ring_result["q_intrinsic"]),
            "q_coupling": float(ring_result["q_coupling"]),
            "sensitivity_nm_per_riu": float(ring_result["sensitivity_nm_per_riu"]),
            "extinction_db": float(ring_result["extinction_db"]),
            "lod_riu": None if ring_result["lod_riu"] is None else float(ring_result["lod_riu"]),
            "resonance_nm": float(ring_result["resonance_nm"]),
            "coupling_regime": ring_result["coupling_regime"],
            "t_through": float(ring_result["t_through"]),
            "t_drop": None if ring_result["t_drop"] is None else float(ring_result["t_drop"]),
            "worst_abs_dlambda_nm": thermal["worst_abs_dlambda_nm"],
            "worst_abs_dkappa": thermal["worst_abs_dkappa"],
            "bend_loss_db_per_cm": float(
                bend_loss_db_per_cm(
                    float(ring.params["radius_um"]), stack.width_nm, stack.polarization
                )
            ),
        }
    )
    return metrics


def system_loss(metrics: dict, objective: SystemObjective) -> float:
    """Scalar the inner loop minimizes. A failed mode is a large penalty."""
    if metrics.get("error"):
        return 1.0e6
    loss = 0.0
    if objective.kappa_weight and objective.kappa_target is not None and metrics.get("kappa") is not None:
        scale = max(float(objective.kappa_target), 1e-3)
        mismatch = (float(metrics["kappa"]) - float(objective.kappa_target)) / scale
        loss += float(objective.kappa_weight) * mismatch * mismatch
    if objective.q_weight and objective.q_target and metrics.get("q"):
        mismatch = (float(metrics["q"]) - float(objective.q_target)) / float(objective.q_target)
        loss += float(objective.q_weight) * mismatch * mismatch
    if objective.sensitivity_weight and metrics.get("sensitivity_nm_per_riu") is not None:
        loss -= float(objective.sensitivity_weight) * (
            float(metrics["sensitivity_nm_per_riu"]) / _SENSITIVITY_SCALE
        )
    if objective.extinction_weight and metrics.get("extinction_db") is not None:
        shortfall = max(0.0, float(objective.extinction_target_db) - float(metrics["extinction_db"]))
        loss += float(objective.extinction_weight) * shortfall / max(objective.extinction_target_db, 1e-6)
    if objective.thermal_lambda_weight and metrics.get("worst_abs_dlambda_nm") is not None:
        loss += float(objective.thermal_lambda_weight) * (
            float(metrics["worst_abs_dlambda_nm"]) / _THERMAL_LAMBDA_SCALE
        )
    if objective.thermal_kappa_weight and metrics.get("worst_abs_dkappa") is not None:
        loss += float(objective.thermal_kappa_weight) * (
            float(metrics["worst_abs_dkappa"]) / _THERMAL_KAPPA_SCALE
        )
    if objective.lod_weight and metrics.get("lod_riu"):
        loss += float(objective.lod_weight) * float(metrics["lod_riu"]) / _LOD_SCALE
    return float(loss)


def inner_tune(
    graph: PhotonicGraph,
    stack: Stack | None = None,
    objective: SystemObjective | None = None,
    *,
    steps: int = 12,
    step_size: float = 0.03,
    free: set[tuple[str, str]] | None = None,
    freeze: set[tuple[str, str]] | None = None,
) -> dict:
    """Inner level: Adam on free device parameters of one DAG."""
    stack = stack or Stack()
    objective = objective or SystemObjective()
    start = graph.copy()
    bound = bind(start, system_specs(objective), free=free, freeze=freeze)
    initial = bound.pack()
    initial_loss, _initial_metrics = _loss_at(start, bound, initial, stack, objective)

    if initial.size == 0:
        tuned = start
        history = [initial_loss]
        final = initial
    else:
        def objective_and_grad(coords: np.ndarray):
            coords = np.asarray(coords, dtype=float)
            value = _loss_at(start, bound, coords, stack, objective)[0]
            grad = np.zeros_like(coords)
            for index in range(coords.size):
                hi = min(1.0, float(coords[index]) + _FD_EPS)
                lo = max(0.0, float(coords[index]) - _FD_EPS)
                if hi <= lo:
                    continue
                above = coords.copy()
                below = coords.copy()
                above[index] = hi
                below[index] = lo
                # One-sided at a bound, central in the interior.
                if hi > coords[index] and lo < coords[index]:
                    high = _loss_at(start, bound, above, stack, objective)[0]
                    low = _loss_at(start, bound, below, stack, objective)[0]
                    grad[index] = (high - low) / (hi - lo)
                elif hi > coords[index]:
                    high = _loss_at(start, bound, above, stack, objective)[0]
                    grad[index] = (high - value) / (hi - float(coords[index]))
                else:
                    low = _loss_at(start, bound, below, stack, objective)[0]
                    grad[index] = (value - low) / (float(coords[index]) - lo)
            return value, grad

        final, history = adam_optimize(
            objective_and_grad,
            initial,
            jac=True,
            step_size=step_size,
            Nsteps=int(steps),
            bounds=(0.0, 1.0),
            direction="min",
        )
        tuned = bound.apply(start, final)
        final_loss, _metrics = _loss_at(start, bound, final, stack, objective)
        history = [float(item) for item in history]
        history.append(float(final_loss))

    tuned_bound = bound.sync(tuned)
    loss, metrics = _loss_at(start, bound, tuned_bound.pack(), stack, objective)
    return {
        "loss": float(loss),
        "initial_loss": float(initial_loss),
        "history": [float(item) for item in history],
        "metrics": metrics,
        "parameters": _with_initial(bound, tuned_bound),
        "graph": tuned.to_dict(),
        "steps": int(steps) if initial.size else 0,
    }


def run_inverse(
    stack: Stack | None = None,
    config: str = "all-pass",
    *,
    graphs: list[PhotonicGraph] | None = None,
    objective: SystemObjective | None = None,
    steps: int = 12,
    step_size: float = 0.03,
    free: set[tuple[str, str]] | None = None,
    freeze: set[tuple[str, str]] | None = None,
) -> dict:
    """Outer level over candidate DAGs, inner level inside each one.

    ``config`` is ``all-pass``, ``add-drop``, or ``both`` when ``graphs`` is
    omitted. The selected candidate is the one with the lowest inner loss.
    """
    stack = stack or Stack()
    objective = objective or SystemObjective()
    if graphs is None:
        if config == "both":
            graphs = [sensor_graph("all-pass"), sensor_graph("add-drop")]
        elif config in ("all-pass", "add-drop"):
            graphs = [sensor_graph(config)]
        else:
            raise ValueError("config must be 'all-pass', 'add-drop', or 'both'")

    candidates = [
        inner_tune(
            graph,
            stack,
            objective,
            steps=steps,
            step_size=step_size,
            free=free,
            freeze=freeze,
        )
        for graph in graphs
    ]
    best = min(range(len(candidates)), key=lambda index: candidates[index]["loss"])
    for index, candidate in enumerate(candidates):
        candidate["selected"] = index == best
    winner = candidates[best]
    return {
        "selected": winner["graph"]["name"],
        "levels": {
            "outer": "discrete DAG architecture",
            "inner": "continuous device parameters",
        },
        "loss": winner["loss"],
        "initial_loss": winner["initial_loss"],
        "metrics": winner["metrics"],
        "parameters": winner["parameters"],
        "graph": winner["graph"],
        "candidates": candidates,
    }


def _loss_at(
    graph: PhotonicGraph,
    bound: ParameterSet,
    scaled: np.ndarray,
    stack: Stack,
    objective: SystemObjective,
) -> tuple[float, dict]:
    realized = bound.apply(graph, scaled)
    metrics = forward(realized, stack)
    value = system_loss(metrics, objective)
    if not np.isfinite(value):
        value = 1.0e6
    return float(value), metrics


def _with_initial(initial: ParameterSet, tuned: ParameterSet) -> dict:
    device = []
    for before, after in zip(initial.device, tuned.device):
        row = after.to_dict()
        row["initial"] = float(before.value)
        device.append(row)
    return {
        "device": device,
        "system": [spec.to_dict() for spec in tuned.system],
        "free": [
            {"node_id": spec.node_id, "key": spec.key, "scaled": spec.to_scaled(spec.value)}
            for spec in tuned.free_device
        ],
    }


def _ambient_corners(
    stack: Stack,
    *,
    kappa0: float,
    length_um: float,
    kappa_design: float,
    resonance_nm: float,
    n_g: float,
    dndt: float,
) -> dict:
    """Uniform chip ΔT. Same compact dn/dT as the circuit thermal corners, no heater PDE."""
    worst_lambda = 0.0
    worst_kappa = 0.0
    for delta_t in CORNERS_K:
        kappa_t = kappa_from_kappa0(kappa0_at(kappa0, dndt, stack.n_clad, delta_t), length_um)
        shift = resonance_shift_nm(resonance_nm, n_g, dndt, delta_t)
        worst_lambda = max(worst_lambda, abs(shift))
        worst_kappa = max(worst_kappa, abs(kappa_t - kappa_design))
    return {
        "worst_abs_dlambda_nm": float(worst_lambda),
        "worst_abs_dkappa": float(worst_kappa),
    }
