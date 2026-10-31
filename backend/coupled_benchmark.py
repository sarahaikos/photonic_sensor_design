"""Coupled inverse design, foundry brushes, and Lorentz dispersion.

The loop is the inner level of ``inverse.pipeline``: Adam on the free coupler
gap, scored by the compact ring model. Each loss evaluation adds the circular
and notched-square brush penalty on a raster of that gap, a sub-resolution rib,
and a sidewall notch. After the gap step, the rib and notch amplitude takes its
own normalized step on the brush gradient, which is how topology optimization
applies the same penalty.

Every recorded design also calls ``compare_dispersive_resonance`` with the
catalog silicon Lorentz pole, so the FDTD auxiliary update and the FDFD
permittivity see one oscillator.

Run from ``backend`` with the project virtualenv:

    .venv/bin/python coupled_benchmark.py
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import re
import struct
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np

from fdtd.materials import C0, MATERIALS, permittivity
from fdtd.resonance import block_mask, block_permittivity, compare_dispersive_resonance
from inverse.graph import sensor_graph
from inverse.parameters import bind, system_specs
from inverse.pipeline import Stack, SystemObjective, forward, system_loss
from mfs_brush import length_scale_brushes, mfs_penalty


MFS_NM = 150.0
DX_NM = 25.0
DX_UM = DX_NM * 1e-3
WIDTH_NM = 450.0
GUIDE_UM = WIDTH_NM * 1e-3
EDGE_NM = 14.0
KAPPA_TARGET = 0.42
COUPLER_LENGTH_UM = 4.0
GAP_STEP = 0.018
FEATURE_STEP = 0.14
GAP_FD = 0.015
FEATURE_FD = 0.02
STEPS = 20
BRUSH_BETA = 8.0
NX = 96
NY = 112

# Rail A sits on a fixed center. Rail B is one guide width plus the gap away,
# so the void between the inner sidewalls is exactly ``gap_nm``.
_RAIL_A_Y = 800.0
_COUPLER_X = 1300.0
_COUPLER_HALF_X = 550.0
_RIB_X = 280.0
_RIB_HALF_X = 18.0
_RIB_HALF_Y = 280.0
_NOTCH_HALF = 45.0

_CAVITY_SHAPE = (16, 12)
_CAVITY_SPACING = (40e-9, 50e-9)

_ROOT = Path(__file__).resolve().parents[1]
_OUT = _ROOT / "insights"


def _step_adam():
    path = Path(__file__).resolve().parent / "ceviche" / "optimizers.py"
    spec = importlib.util.spec_from_file_location("_coupled_adam", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.step_adam


step_adam = _step_adam()


def _inside(distance_nm: np.ndarray, edge_nm: float = EDGE_NM) -> np.ndarray:
    argument = np.clip(distance_nm / edge_nm, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(argument))


def _bar(x_nm, y_nm, cx: float, cy: float, hx: float, hy: float) -> np.ndarray:
    along = _inside(np.abs(x_nm - cx) - hx)
    across = _inside(np.abs(y_nm - cy) - hy)
    return along[:, None] * across[None, :]


def coupler_density(gap_nm: float, feature_amp: float) -> np.ndarray:
    """Two legal guides, plus a thin rib and a sidewall notch scaled by ``feature_amp``."""
    x_nm = (np.arange(NX) + 0.5) * DX_NM
    y_nm = (np.arange(NY) + 0.5) * DX_NM
    half = 0.5 * WIDTH_NM
    rail_a = _bar(x_nm, y_nm, _COUPLER_X, _RAIL_A_Y, _COUPLER_HALF_X, half)
    rail_b_y = _RAIL_A_Y + WIDTH_NM + float(gap_nm)
    rail_b = _bar(x_nm, y_nm, _COUPLER_X, rail_b_y, _COUPLER_HALF_X, half)
    amp = float(np.clip(feature_amp, 0.0, 1.0))
    rib = amp * _bar(x_nm, y_nm, _RIB_X, _RAIL_A_Y, _RIB_HALF_X, _RIB_HALF_Y)
    notch = _bar(x_nm, y_nm, _COUPLER_X, _RAIL_A_Y + half, _NOTCH_HALF, _NOTCH_HALF)
    solid = np.maximum(np.maximum(rail_a, rail_b), rib)
    return np.clip(solid - amp * notch, 0.0, 1.0)


def design_mask(brushes) -> np.ndarray:
    mask = np.zeros((NX, NY), dtype=bool)
    pad = int(brushes.radius_px) + 1
    mask[pad:-pad, pad:-pad] = True
    return mask


def _gap_nm(bound, graph, gap_scaled: float) -> float:
    tuned = bound.apply(graph, np.array([gap_scaled], dtype=float))
    return float(tuned.nodes["cpl"].params["gap_nm"])


def evaluate(bound, graph, stack, objective, brushes, mask, gap_scaled: float, feature: float, mfs_weight: float) -> dict:
    tuned = bound.apply(graph, np.array([gap_scaled], dtype=float))
    metrics = forward(tuned, stack)
    optical = system_loss(metrics, objective)
    gap = float(tuned.nodes["cpl"].params["gap_nm"])
    rho = coupler_density(gap, feature)
    terms, _grad = mfs_penalty(rho, brushes, mask, BRUSH_BETA)
    brush = float(terms["total"])
    loss = float(optical + mfs_weight * brush)
    return {
        "loss": loss,
        "optical": float(optical),
        "brush": brush,
        "gap_nm": gap,
        "gap_scaled": float(gap_scaled),
        "feature_amp": float(feature),
        "kappa": None if metrics.get("kappa") is None else float(metrics["kappa"]),
        "q": None if metrics.get("q") is None else float(metrics["q"]),
        "extinction_db": None if metrics.get("extinction_db") is None else float(metrics["extinction_db"]),
        "resonance_nm": None if metrics.get("resonance_nm") is None else float(metrics["resonance_nm"]),
        "coupling_regime": metrics.get("coupling_regime"),
        "circular_width": float(terms["circular_width"]),
        "circular_gap": float(terms["circular_gap"]),
        "notch": float(terms["notch"]),
        "notched_width": float(terms["notched_width"]),
        "notched_gap": float(terms["notched_gap"]),
        "gap_meets_mfs": bool(gap + 1e-6 >= MFS_NM),
        "feature_cleared": bool(feature <= 0.2),
    }


def _scalar_grad(coord: float, loss_at, eps: float) -> float:
    value = loss_at(coord)
    hi = min(1.0, coord + eps)
    lo = max(0.0, coord - eps)
    if hi <= lo:
        return 0.0
    if hi > coord and lo < coord:
        return (loss_at(hi) - loss_at(lo)) / (hi - lo)
    if hi > coord:
        return (loss_at(hi) - value) / (hi - coord)
    return (value - loss_at(lo)) / (coord - lo)


def _lorentz_setup():
    silicon = MATERIALS["si"]
    eps_inf = block_permittivity(_CAVITY_SHAPE, eps_core=float(silicon.eps_inf), eps_bg=1.0)
    mask = block_mask(_CAVITY_SHAPE)
    pole = silicon.poles[0]
    if pole.kind != "lorentz":
        raise RuntimeError("silicon catalog entry is not a Lorentz pole")
    return eps_inf, mask, pole


def lorentz_metrics(setup, resonance_nm: float) -> dict:
    """One silicon pole, scored by FDFD and by the FDTD auxiliary update."""
    eps_inf, mask, pole = setup
    started = time.perf_counter()
    report = compare_dispersive_resonance(
        eps_inf,
        mask,
        pole,
        _CAVITY_SPACING,
        frozen_omega=0.0,
        cfl=0.45,
        steps=2000,
    )
    elapsed = time.perf_counter() - started
    fdfd = report["fdfd"]
    fdtd = report["fdtd"]
    frozen = report["frozen"]
    omega = 2.0 * math.pi * C0 / (float(resonance_nm) * 1e-9)
    eps = permittivity(MATERIALS["si"], omega)
    q_fdfd = float(fdfd["q"])
    q_fdtd = float(fdtd["q_corrected"])
    return {
        "lorentz_s": elapsed,
        "q_fdfd": q_fdfd,
        "q_fdtd": q_fdtd,
        "rel_q": abs(q_fdtd - q_fdfd) / abs(q_fdfd),
        "wavelength_fdfd_nm": float(fdfd["wavelength_nm"]),
        "wavelength_fdtd_nm": float(fdtd["wavelength_corrected_nm"]),
        "rel_wavelength": abs(float(fdtd["wavelength_corrected_nm"]) - float(fdfd["wavelength_nm"]))
        / abs(float(fdfd["wavelength_nm"])),
        "fit_residual": float(fdtd["fit_residual"]),
        "frozen_wavelength_nm": float(frozen["wavelength_nm"]),
        "dispersive_shift_nm": float(fdfd["wavelength_nm"]) - float(frozen["wavelength_nm"]),
        "silicon_eps_real": float(eps.real),
        "silicon_eps_imag": float(eps.imag),
    }


def run_arm(
    name: str,
    bound,
    graph,
    stack,
    objective,
    brushes,
    mask,
    lorentz_setup,
    gap_scaled: float,
    feature: float,
    mfs_weight: float,
) -> list[dict]:
    """Adam on the gap, then a normalized brush step on the violation amplitude."""
    moment = np.zeros(1)
    velocity = np.zeros(1)
    history: list[dict] = []
    elapsed = 0.0
    for step in range(STEPS + 1):
        started = time.perf_counter()
        row = evaluate(bound, graph, stack, objective, brushes, mask, gap_scaled, feature, mfs_weight)
        row.update(lorentz_metrics(lorentz_setup, row["resonance_nm"]))
        eval_s = time.perf_counter() - started
        elapsed += eval_s
        row["arm"] = name
        row["step"] = step
        row["mfs_weight"] = float(mfs_weight)
        row["eval_s"] = eval_s
        row["t_s"] = elapsed
        history.append(row)
        print(
            f"  {name:16s} {step:2d}  loss {row['loss']:.4f}  gap {row['gap_nm']:.1f} nm"
            f"  feature {row['feature_amp']:.2f}  κ {row['kappa']:.3f}"
            f"  Q {row['q']:.0f}  ER {row['extinction_db']:.2f} dB"
            f"  brush {row['brush']:.4f}",
            flush=True,
        )
        if step == STEPS:
            break
        started = time.perf_counter()
        grad_gap = _scalar_grad(
            gap_scaled,
            lambda coord: evaluate(
                bound, graph, stack, objective, brushes, mask, coord, feature, mfs_weight
            )["loss"],
            GAP_FD,
        )
        direction, moment, velocity = step_adam(np.array([grad_gap]), moment, velocity, step, 0.9, 0.999)
        # The weighted loss rises between the κ well near 120 nm and the foundry
        # basin near 180 nm. Adam is allowed to cross that ridge.
        gap_scaled = float(np.clip(gap_scaled - GAP_STEP * float(direction[0]), 0.0, 1.0))
        if mfs_weight > 0.0:
            grad_feature = _scalar_grad(
                feature,
                lambda coord: evaluate(
                    bound, graph, stack, objective, brushes, mask, gap_scaled, coord, mfs_weight
                )["loss"],
                FEATURE_FD,
            )
            if abs(grad_feature) > 1e-10:
                trial = float(np.clip(feature - FEATURE_STEP * float(np.sign(grad_feature)), 0.0, 1.0))
                trial_loss = evaluate(
                    bound, graph, stack, objective, brushes, mask, gap_scaled, trial, mfs_weight
                )["loss"]
                current_loss = evaluate(
                    bound, graph, stack, objective, brushes, mask, gap_scaled, feature, mfs_weight
                )["loss"]
                if trial_loss <= current_loss:
                    feature = trial
        elapsed += time.perf_counter() - started
    return history


def gap_sweep(bound, graph, stack, objective, brushes, mask) -> list[dict]:
    rows = []
    for gap in range(80, 321, 10):
        scaled = (float(gap) - 50.0) / (500.0 - 50.0)
        bare = evaluate(bound, graph, stack, objective, brushes, mask, scaled, 0.0, 0.0)
        marked = evaluate(bound, graph, stack, objective, brushes, mask, scaled, 1.0, 0.0)
        rows.append(
            {
                "gap_nm": bare["gap_nm"],
                "optical": bare["optical"],
                "brush": bare["brush"],
                "circular_gap": bare["circular_gap"],
                "circular_width": bare["circular_width"],
                "notch": bare["notch"],
                "kappa": bare["kappa"],
                "q": bare["q"],
                "extinction_db": bare["extinction_db"],
                "marked_brush": marked["brush"],
                "marked_circular_width": marked["circular_width"],
                "marked_notch": marked["notch"],
                "marked_circular_gap": marked["circular_gap"],
            }
        )
    return rows


def assert_brush_sees_violations(samples: list[dict]) -> None:
    illegal = min(samples, key=lambda row: abs(row["gap_nm"] - 80.0))
    legal = min(samples, key=lambda row: abs(row["gap_nm"] - 250.0))
    if not illegal["circular_gap"] > legal["circular_gap"]:
        raise RuntimeError("narrow coupler gap did not raise the circular gap penalty")
    if not illegal["marked_circular_width"] > illegal["circular_width"]:
        raise RuntimeError("sub-resolution rib did not raise the width penalty")
    if not illegal["marked_notch"] > illegal["notch"]:
        raise RuntimeError("sidewall notch did not raise the notch penalty")


def select_weight(samples: list[dict]) -> tuple[float, dict]:
    """Smallest weight that moves the sweep minimum to a gap the foundry allows."""
    optical_best = min(samples, key=lambda row: row["optical"])
    floor = min(row["brush"] for row in samples)
    span = max(row["brush"] for row in samples) - floor
    legal = [row for row in samples if row["gap_nm"] + 1e-6 >= MFS_NM]
    near_floor = [row for row in legal if row["brush"] <= floor + 0.2 * span]
    target = min(near_floor or legal, key=lambda row: row["gap_nm"])
    drop = optical_best["brush"] - target["brush"]
    if drop <= 1e-8:
        raise RuntimeError("brush penalty does not fall once the gap is legal")
    extra = max(0.0, target["optical"] - optical_best["optical"])
    weight = (extra / drop) * 1.35 + 0.02
    for _ in range(8):
        winner = min(samples, key=lambda row: row["optical"] + weight * row["brush"])
        if winner["gap_nm"] + 1e-6 >= MFS_NM:
            return float(weight), winner
        weight *= 1.7
    raise RuntimeError("no brush weight placed the sweep minimum on a legal gap")


def _converged_index(history: list[dict]) -> int:
    losses = [row["loss"] for row in history]
    start = losses[0]
    best = min(losses)
    if start <= best:
        return 0
    limit = best + 0.1 * (start - best)
    for index, loss in enumerate(losses):
        if loss <= limit:
            return index
    return len(history) - 1


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


def arm_summary(history: list[dict]) -> dict:
    final = history[-1]
    best = min(history, key=lambda row: row["loss"])
    reached = _converged_index(history)
    return {
        "steps": STEPS,
        "final": final,
        "best": best,
        "converged_step": reached,
        "converged_s": history[reached]["t_s"],
        "total_s": history[-1]["t_s"],
        "mean_lorentz_s": _mean([row["lorentz_s"] for row in history]),
        "mean_rel_q": _mean([row["rel_q"] for row in history]),
        "max_rel_q": max(row["rel_q"] for row in history),
        "mean_rel_wavelength": _mean([row["rel_wavelength"] for row in history]),
        "max_fit_residual": max(row["fit_residual"] for row in history),
        "q_fdfd": _mean([row["q_fdfd"] for row in history]),
        "q_fdtd": _mean([row["q_fdtd"] for row in history]),
        "dispersive_shift_nm": _mean([row["dispersive_shift_nm"] for row in history]),
        "silicon_eps_real": history[0]["silicon_eps_real"],
        "silicon_eps_imag": history[0]["silicon_eps_imag"],
        "ring_resonance_nm": history[0]["resonance_nm"],
    }


def run_unit_tests() -> dict:
    started = time.perf_counter()
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "test_inverse.py",
            "test_mfs_brush.py",
            "test_dispersive_resonance.py",
        ],
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - started
    text = proc.stdout + proc.stderr
    count = re.search(r"Ran (\d+) tests", text)
    timed = re.search(r"Ran \d+ tests in ([0-9.]+)s", text)
    return {
        "ok": proc.returncode == 0 and "FAILED" not in text,
        "returncode": proc.returncode,
        "tests": int(count.group(1)) if count else None,
        "suite_seconds": float(timed.group(1)) if timed else None,
        "seconds": elapsed,
        "tail": "\n".join(text.strip().splitlines()[-8:]),
    }


def _fmt(value, digits=3) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    number = float(value)
    if abs(number) >= 1000:
        return f"{number:.0f}"
    if abs(number) >= 10:
        return f"{number:.2f}"
    return f"{number:.{digits}g}"


def _svg(path: Path, title: str, xlabel: str, ylabel: str, series: list[tuple], caption: str) -> None:
    width, height = 760, 420
    left, right, top, bottom = 72, 24, 42, 78
    xs = [x for _name, points, _color in series for x, _y in points]
    ys = [y for _name, points, _color in series for _x, y in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmax <= xmin:
        xmax = xmin + 1.0
    if ymax <= ymin:
        pad = abs(ymin) * 0.05 + 1e-6
        ymin, ymax = ymin - pad, ymax + pad
    else:
        pad = 0.06 * (ymax - ymin)
        ymin -= pad
        ymax += pad

    def px(x: float) -> float:
        return left + (x - xmin) / (xmax - xmin) * (width - left - right)

    def py(y: float) -> float:
        return top + (ymax - y) / (ymax - ymin) * (height - top - bottom)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfbfa"/>',
        f'<text x="{left}" y="26" font-family="Georgia, serif" font-size="16" fill="#1c1c1c">{title}</text>',
    ]
    for tick in range(5):
        fraction = tick / 4
        x = xmin + fraction * (xmax - xmin)
        y = ymin + fraction * (ymax - ymin)
        parts.append(
            f'<line x1="{px(x):.1f}" y1="{top}" x2="{px(x):.1f}" y2="{height - bottom}" stroke="#e4e4e0"/>'
        )
        parts.append(
            f'<line x1="{left}" y1="{py(y):.1f}" x2="{width - right}" y2="{py(y):.1f}" stroke="#e4e4e0"/>'
        )
        parts.append(
            f'<text x="{px(x):.1f}" y="{height - bottom + 18}" text-anchor="middle" font-family="ui-sans-serif, sans-serif" font-size="11" fill="#444">{x:.3g}</text>'
        )
        parts.append(
            f'<text x="{left - 8}" y="{py(y) + 4:.1f}" text-anchor="end" font-family="ui-sans-serif, sans-serif" font-size="11" fill="#444">{y:.3g}</text>'
        )
    parts.append(
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#222"/>'
    )
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#222"/>')
    for name, points, color in series:
        coords = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in points)
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{coords}"/>')
    for index, (name, _points, color) in enumerate(series):
        y = height - 28
        x = left + index * 220
        parts.append(f'<line x1="{x}" y1="{y - 4}" x2="{x + 22}" y2="{y - 4}" stroke="{color}" stroke-width="2"/>')
        parts.append(
            f'<text x="{x + 28}" y="{y}" font-family="ui-sans-serif, sans-serif" font-size="12" fill="#222">{name}</text>'
        )
    parts.append(
        f'<text x="{width / 2}" y="{height - bottom + 36}" text-anchor="middle" font-family="ui-sans-serif, sans-serif" font-size="12" fill="#222">{xlabel}</text>'
    )
    parts.append(
        f'<text x="18" y="{height / 2}" text-anchor="middle" font-family="ui-sans-serif, sans-serif" font-size="12" fill="#222" transform="rotate(-90 18 {height / 2})">{ylabel}</text>'
    )
    parts.append(
        f'<text x="{left}" y="{height - 8}" font-family="ui-sans-serif, sans-serif" font-size="11" fill="#555">{caption}</text>'
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def _png(path: Path, rho: np.ndarray) -> None:
    shown = np.clip(rho, 0.0, 1.0).T[::-1, :]
    factor = 6
    shown = np.repeat(np.repeat(shown, factor, axis=0), factor, axis=1)
    void = np.array([18, 24, 38], dtype=np.float64)
    silicon = np.array([214, 186, 122], dtype=np.float64)
    rgb = void * (1.0 - shown[..., None]) + silicon * shown[..., None]
    image = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    height, width, _ = image.shape

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    )


def _csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "arm",
        "step",
        "t_s",
        "loss",
        "optical",
        "brush",
        "gap_nm",
        "feature_amp",
        "kappa",
        "q",
        "extinction_db",
        "circular_width",
        "circular_gap",
        "notch",
        "q_fdfd",
        "q_fdtd",
        "rel_q",
        "lorentz_s",
        "fit_residual",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _markdown(path: Path, context: dict) -> None:
    tests = context["tests"]
    coupled = context["coupled"]
    plain = context["plain"]
    initial = context["initial"]
    brushes = context["brushes"]
    weight = context["weight"]
    winner = context["winner"]
    samples = context["samples"]
    c_best = coupled["best"]
    p_best = plain["best"]
    width_drop = initial["circular_width"] - c_best["circular_width"]
    gap_drop = initial["circular_gap"] - c_best["circular_gap"]
    notch_drop = initial["notch"] - c_best["notch"]
    subres_ok = c_best["feature_amp"] < initial["feature_amp"] and width_drop > 0 and notch_drop > 0
    gap_ok = c_best["gap_nm"] > p_best["gap_nm"] and gap_drop > 0 and c_best["gap_meets_mfs"]
    kappa_err = lambda row: abs(row["kappa"] - KAPPA_TARGET)
    floor = min(samples, key=lambda row: abs(row["gap_nm"] - c_best["gap_nm"]))
    suite_s = tests["suite_seconds"] if tests["suite_seconds"] is not None else tests["seconds"]

    def row(label, key, digits=3):
        return (
            f"| {label} | {_fmt(initial[key], digits)} | {_fmt(c_best[key], digits)} | {_fmt(p_best[key], digits)} |"
        )

    lines = [
        "# Coupled inverse-design benchmark",
        "",
        "Inverse design, foundry brushes, and the silicon Lorentz pole run in one loop.",
        "The compact ring model supplies loaded Q, extinction, and κ. The brush penalty",
        "is inside the gap update. Each recorded design calls the same Lorentz pole from",
        "FDFD and from the FDTD auxiliary update.",
        "",
        "## Unit tests",
        "",
        "Ran `test_inverse.py`, `test_mfs_brush.py`, and `test_dispersive_resonance.py` before the coupled loop.",
        f"{tests['tests']} tests, {'all passed' if tests['ok'] else 'FAILED'}, {suite_s:.3f} s in the unittest timer ({tests['seconds']:.3f} s wall clock).",
        "",
        "## Setup",
        "",
        f"- Foundry minimum feature {MFS_NM:.0f} nm. Grid {DX_NM:.0f} nm. Circular brush radius {brushes.radius_px} px, enforced width {brushes.enforced_feature_nm:.0f} nm. Notch cut {brushes.notch_nm:.0f} nm. Brush smoothing β = {BRUSH_BETA:.0f}.",
        f"- Coupler length locked at {COUPLER_LENGTH_UM:.0f} μm and ring radius locked at 10 μm, so the gap is the coupling knob. κ target {KAPPA_TARGET:.2f}. That target sits near a 120 nm gap, which is below the foundry rule.",
        f"- Start: gap {initial['gap_nm']:.0f} nm, violation amplitude 1 (a 36 nm rib and a 90 nm sidewall notch).",
        f"- Brush weight {weight:.3f}, chosen so the gap sweep's minimum of optical loss plus the weighted penalty is at {winner['gap_nm']:.0f} nm.",
        f"- Adam gap step {GAP_STEP:.3f} in scaled coordinates (about {GAP_STEP * 450:.1f} nm). The weighted loss has a ridge between the κ well near 120 nm and the foundry basin near 180 nm, so the gap step is allowed to climb it. Violation amplitude step {FEATURE_STEP:.2f}, accepted only when it lowers the loss. {STEPS} iterations.",
        "",
        "## Convergence",
        "",
        f"Coupled loop: loss {initial['loss']:.4f} → {coupled['final']['loss']:.4f}.",
        f"Lowest loss is {c_best['loss']:.4f} at iteration {c_best['step']}.",
        f"90% of the drop from the start to that lowest loss is in hand at iteration {coupled['converged_step']} ({coupled['converged_s']:.2f} s).",
        f"Wall time {coupled['total_s']:.2f} s, of which the Lorentz pair averages {coupled['mean_lorentz_s'] * 1000:.0f} ms per call.",
        "",
        f"Unconstrained loop (brush weight 0): lowest loss {p_best['loss']:.4f} at iteration {p_best['step']} ({plain['best']['t_s']:.2f} s), endpoint loss {plain['final']['loss']:.4f}. Wall time {plain['total_s']:.2f} s.",
        "The table below uses the lowest-loss iterate of each arm. Histories are in `coupled_benchmark.json`.",
        "",
        "## Sensor metrics and fabrication",
        "",
        "| Metric | Initial | Coupled selected | Unconstrained selected |",
        "| --- | ---: | ---: | ---: |",
        row("Gap (nm)", "gap_nm", 4),
        row("Violation amplitude", "feature_amp"),
        row("κ", "kappa"),
        f"| κ error | {_fmt(kappa_err(initial))} | {_fmt(kappa_err(c_best))} | {_fmt(kappa_err(p_best))} |",
        row("Loaded Q", "q", 4),
        row("Extinction (dB)", "extinction_db"),
        row("Circular width penalty", "circular_width"),
        row("Circular gap penalty", "circular_gap"),
        row("Notch penalty", "notch"),
        row("Total brush penalty", "brush"),
        "",
        f"Sub-resolution rib and notch penalized: {'yes' if subres_ok else 'no'}.",
        f"Width penalty {initial['circular_width']:.4f} → {c_best['circular_width']:.4f}, notch {initial['notch']:.4f} → {c_best['notch']:.4f}, amplitude {initial['feature_amp']:.2f} → {c_best['feature_amp']:.2f}.",
        f"With the rib and notch removed, the sweep near {floor['gap_nm']:.0f} nm has width penalty {floor['circular_width']:.4f} and notch penalty {floor['notch']:.4f}. That floor is the smooth edge of the legal guides.",
        f"Illegal gap penalized: {'yes' if gap_ok else 'no'}.",
        f"Gap penalty {initial['circular_gap']:.4f} → {c_best['circular_gap']:.4f}. Coupled gap {c_best['gap_nm']:.1f} nm ({'meets' if c_best['gap_meets_mfs'] else 'misses'} {MFS_NM:.0f} nm). Unconstrained gap {p_best['gap_nm']:.1f} nm ({'meets' if p_best['gap_meets_mfs'] else 'misses'} {MFS_NM:.0f} nm).",
        "",
        "Tradeoff against the unconstrained κ match: opening the gap to satisfy the brush",
        f"moves κ from {p_best['kappa']:.3f} to {c_best['kappa']:.3f} (target {KAPPA_TARGET:.2f}).",
        f"Loaded Q goes from {p_best['q']:.0f} to {c_best['q']:.0f} because the ring is less strongly coupled.",
        f"Extinction goes from {p_best['extinction_db']:.2f} dB to {c_best['extinction_db']:.2f} dB.",
        "Both designs stay over-coupled relative to the ring's round-trip loss, so the extinction stays low; critical coupling on this 4 μm coupler wants a still wider gap.",
        "",
        "## Lorentz pole across both solvers",
        "",
        "The cavity is a silicon block on the same 16×12 mesh the dispersion unit test uses.",
        "Its resonance is not the ring resonance. The check is that one catalog pole",
        f"(`si`, Δε = {MATERIALS['si'].poles[0].eps_delta}, ω0 = {MATERIALS['si'].poles[0].omega0:.3e} rad/s)",
        "produces the same complex root in both solvers while the optimizer is moving.",
        "",
        f"- FDFD Q {coupled['q_fdfd']:.1f}, FDTD corrected Q {coupled['q_fdtd']:.1f}, mean relative difference {coupled['mean_rel_q']:.3e} (max {coupled['max_rel_q']:.3e}).",
        f"- Mean relative wavelength difference {coupled['mean_rel_wavelength']:.3e}. Worst FDTD fit residual {coupled['max_fit_residual']:.3e}.",
        f"- Live FDFD wavelength is {abs(coupled['dispersive_shift_nm']):.2f} nm {'longer' if coupled['dispersive_shift_nm'] >= 0 else 'shorter'} than the cavity whose permittivity is frozen at ω = 0.",
        f"- At the ring resonance ({coupled['ring_resonance_nm']:.2f} nm) the same pole gives ε = {coupled['silicon_eps_real']:.4f} + {coupled['silicon_eps_imag']:.4f} i.",
        f"- {STEPS + 1} calls on the coupled arm, mean {coupled['mean_lorentz_s'] * 1000:.0f} ms, sum {coupled['mean_lorentz_s'] * (STEPS + 1):.2f} s.",
        "",
        "## Layouts",
        "",
        "Gold is silicon, navy is void. Propagation is horizontal and the coupler gap is the vertical opening. The rib is the short vertical mark on the left.",
        "",
        "![Initial layout, 80 nm gap and violation amplitude 1](layout_initial.png)",
        "",
        "![Coupled layout at the lowest-loss iteration](layout_coupled.png)",
        "",
        "![Unconstrained layout at the lowest-loss iteration](layout_unconstrained.png)",
        "",
        "![Loss versus iteration](convergence.svg)",
        "",
        "![Gap and κ versus iteration](gap_kappa.svg)",
        "",
        "![Optical loss and brush penalty versus gap, violation amplitude 0](sweep.svg)",
        "",
        "## Files",
        "",
        "- `coupled_benchmark.json` — full histories, the gap sweep, and the Lorentz records",
        "- `history.csv` — one row per iteration of each arm",
        "",
        "Reproduce with `backend/.venv/bin/python coupled_benchmark.py` from `backend`.",
        "",
    ]
    path.write_text("\n".join(lines))


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def main() -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    print("Unit tests", flush=True)
    tests = run_unit_tests()
    print(tests["tail"], flush=True)
    if not tests["ok"]:
        raise SystemExit("unit tests failed; coupled benchmark not started")

    brushes = length_scale_brushes(MFS_NM, DX_UM, GUIDE_UM)
    mask = design_mask(brushes)
    stack = Stack()
    objective = SystemObjective(kappa_weight=1.0, kappa_target=KAPPA_TARGET)
    graph = sensor_graph("all-pass")
    graph.nodes["cpl"].params["gap_nm"] = 80.0
    graph.nodes["cpl"].params["length_um"] = COUPLER_LENGTH_UM
    graph.lock("cpl", "length_um")
    graph.lock("ring", "radius_um")
    bound = bind(graph, system_specs(objective), free={("cpl", "gap_nm")})
    if [spec.key for spec in bound.free_device] != ["gap_nm"]:
        raise RuntimeError("expected the coupler gap to be the only free device parameter")

    print("Gap sweep", flush=True)
    samples = gap_sweep(bound, graph, stack, objective, brushes, mask)
    assert_brush_sees_violations(samples)
    weight, winner = select_weight(samples)
    print(f"Brush weight {weight:.4f}, sweep minimum at {winner['gap_nm']:.1f} nm", flush=True)

    setup = _lorentz_setup()
    gap0 = float(bound.pack()[0])
    print("Coupled arm", flush=True)
    coupled_hist = run_arm(
        "coupled", bound, graph, stack, objective, brushes, mask, setup, gap0, 1.0, weight
    )
    print("Unconstrained arm", flush=True)
    plain_hist = run_arm(
        "unconstrained", bound, graph, stack, objective, brushes, mask, setup, gap0, 1.0, 0.0
    )
    coupled = arm_summary(coupled_hist)
    plain = arm_summary(plain_hist)

    _png(_OUT / "layout_initial.png", coupler_density(coupled_hist[0]["gap_nm"], coupled_hist[0]["feature_amp"]))
    _png(_OUT / "layout_coupled.png", coupler_density(coupled["best"]["gap_nm"], coupled["best"]["feature_amp"]))
    _png(
        _OUT / "layout_unconstrained.png",
        coupler_density(plain["best"]["gap_nm"], plain["best"]["feature_amp"]),
    )

    _svg(
        _OUT / "convergence.svg",
        "Objective versus iteration",
        "Iteration",
        "Loss",
        [
            ("Coupled", [(row["step"], row["loss"]) for row in coupled_hist], "#1f4b99"),
            ("Unconstrained", [(row["step"], row["loss"]) for row in plain_hist], "#b15c2e"),
        ],
        "Source: coupled_benchmark.py · compact ring loss plus weighted brush penalty",
    )
    _svg(
        _OUT / "gap_kappa.svg",
        "Coupler gap and power coupling",
        "Iteration",
        "Gap (nm)  ·  κ × 1000",
        [
            ("Coupled gap", [(row["step"], row["gap_nm"]) for row in coupled_hist], "#1f4b99"),
            ("Unconstrained gap", [(row["step"], row["gap_nm"]) for row in plain_hist], "#b15c2e"),
            ("Coupled κ × 1000", [(row["step"], 1000 * row["kappa"]) for row in coupled_hist], "#2c6e49"),
        ],
        "Source: coupled_benchmark.py · κ scaled by 1000 so it shares the gap axis",
    )
    _svg(
        _OUT / "sweep.svg",
        "κ loss and brush penalty versus gap",
        "Gap (nm)",
        "Loss",
        [
            ("Optical κ loss", [(row["gap_nm"], row["optical"]) for row in samples], "#b15c2e"),
            ("Brush penalty", [(row["gap_nm"], row["brush"]) for row in samples], "#1f4b99"),
            (
                "Weighted sum",
                [(row["gap_nm"], row["optical"] + weight * row["brush"]) for row in samples],
                "#2c6e49",
            ),
        ],
        "Source: coupled_benchmark.py · violation amplitude 0, before the Adam loop",
    )
    _csv(_OUT / "history.csv", coupled_hist + plain_hist)

    payload = {
        "mfs_nm": MFS_NM,
        "dx_nm": DX_NM,
        "brush_beta": BRUSH_BETA,
        "enforced_feature_nm": brushes.enforced_feature_nm,
        "kappa_target": KAPPA_TARGET,
        "coupler_length_um": COUPLER_LENGTH_UM,
        "mfs_weight": weight,
        "sweep_winner_gap_nm": winner["gap_nm"],
        "steps": STEPS,
        "gap_step": GAP_STEP,
        "feature_step": FEATURE_STEP,
        "tests": tests,
        "sweep": samples,
        "coupled": coupled_hist,
        "unconstrained": plain_hist,
    }
    (_OUT / "coupled_benchmark.json").write_text(json.dumps(_jsonable(payload), indent=2))
    _markdown(
        _OUT / "coupled_benchmark.md",
        {
            "tests": tests,
            "coupled": coupled,
            "plain": plain,
            "initial": coupled_hist[0],
            "brushes": brushes,
            "weight": weight,
            "winner": winner,
            "samples": samples,
        },
    )
    print(
        f"Wrote {_OUT} · coupled gap {coupled['best']['gap_nm']:.1f} nm,"
        f" Q {coupled['best']['q']:.0f}, ER {coupled['best']['extinction_db']:.2f} dB,"
        f" feature {coupled['best']['feature_amp']:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
