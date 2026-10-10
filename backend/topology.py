"""Density-based topology optimization → foundry-aware GDS."""

from __future__ import annotations

import base64
import math
import struct
from datetime import datetime, timezone

import numpy as np

from fdtd.geometry import Rect, Scene
from thermo import (
    DN_DT_SI,
    bicgstab_field,
    circuit_heat_grid,
    circuit_masks,
    circuit_si_density,
    coupled_window_heat,
    default_coupler_heater,
    dn_clad_dt,
    dn_eff_dt,
    heat_field,
    heater_layout,
    heater_polygons,
    mean_on_mask,
    neighbor_sum,
    paint_heaters,
    resonance_shift_nm,
    sheet_k,
    solve_heat,
    window_heat_gradient,
)
from thermo_adjoint import (
    helmholtz_apply,
    optical_adjoint_field,
    permittivity_derivatives,
    permittivity_gradient,
    port_field_grad,
    te_apply,
    te_permittivity_gradient,
    thermo_optic_terms,
)
from waveguide import analyze_waveguide

# Dilated / intermediate / eroded thresholds. Wang, Lazarov & Sigmund 2011.
_ROBUST_ETAS = (("dilated", 0.3), ("intermediate", 0.5), ("eroded", 0.7))
# Forward and adjoint share this residual. A nonzero field is not a solution.
_SOLVE_TOL = 1e-5
_SOLVE_ITERS = 4000
_MATERIAL_LOSS = 0.015


def run_topology(
    width_nm: float,
    height_nm: float,
    wavelength_nm: float,
    n_clad: float,
    polarization: str,
    devices: list[dict],
    mfs_nm: float = 150.0,
    min_gap_nm: float = 150.0,
    etch_nm: float = 0.0,
    rounding_nm: float = 20.0,
    beta: float = 8.0,
    steps: int = 8,
    kappa_target: float | None = None,
    thermo_optic: bool = False,
) -> dict:
    scene, w_um, length_um, gap_um = _coupler_scene(width_nm, devices)
    if scene is None:
        return {"error": "Place a waveguide or coupler to run topology optimization."}

    # 2D Helmholtz is reliable on a short window; TO/GDS use that slice of the coupler.
    length_um = min(length_um, 3.6)
    pad_um = 1.0 if thermo_optic else 0.4
    dx_um = _dx_for(w_um, length_um, gap_um)
    xs, ys, rho0, design, guides = _raster(dx_um, w_um, length_um, gap_um, pad_um)
    r_px = float(np.clip(0.5 * mfs_nm / (dx_um * 1e3), 0.8, 2.4))
    beta = float(max(1.0, min(beta, 32.0)))
    steps = int(max(0, min(steps, 20)))

    mode = analyze_waveguide(
        width_nm, height_nm, wavelength_nm, n_clad, polarization, length_um=100.0
    )
    n_core = float(mode["n_eff"]) if mode else 2.45
    n_bg = float(n_clad)
    # Local index shift on the grid. The compact ring model keeps the overlap dn_eff/dT.
    dndt_core = DN_DT_SI
    dndt_clad = dn_clad_dt(n_clad)
    etch = (etch_nm / (dx_um * 1e3), rounding_nm / (dx_um * 1e3))
    heaters = []
    heat_circuit = None
    if thermo_optic:
        laid = heater_layout(devices, width_nm)
        if laid:
            heaters = laid
        else:
            power = next(
                (float(d.get("power_mw", 10.0)) for d in devices if d.get("type") == "heater"),
                10.0,
            )
            heaters = [default_coupler_heater(w_um, length_um, pad_um, power)]
        cxs, cys = circuit_heat_grid(width_nm, devices)
        ring_mask, _coupler_mask = circuit_masks(cxs, cys, width_nm, devices)
        dldt = 0.0
        rings = [d for d in devices if d.get("type") == "ring"]
        if rings and mode:
            radius = float(rings[0].get("radius_um", 10.0))
            length_ring = 2.0 * math.pi * radius
            lam_um = wavelength_nm * 1e-3
            order = round(n_core * length_ring / lam_um)
            resonance_nm = n_core * length_ring / order * 1e3 if order else wavelength_nm
            dldt = resonance_shift_nm(
                resonance_nm,
                float(mode["n_g"]),
                dn_eff_dt(mode["gamma_core"], mode["gamma_clad"], n_clad),
                1.0,
            )
        heat_circuit = {
            "xs": cxs,
            "ys": cys,
            "rho_fixed": circuit_si_density(cxs, cys, width_nm, devices),
            "q": paint_heaters(cxs, cys, heaters),
            "ring": ring_mask,
            "dldt": float(dldt),
        }

    rho = rho0.copy()
    if design.any() and gap_um is not None:
        rho = _seed_swg(rho, design, ys, dx_um)

    target = float(np.clip(kappa_target if kappa_target is not None else 0.12, 0.05, 0.95))
    history: list[dict] = []
    field_ez = None
    t_th = t_dr = None

    if steps > 0 and design.any() and gap_um is not None:
        rho, history, t_th, t_dr, field_ez = _optimize(
            rho,
            design,
            xs,
            ys,
            dx_um,
            r_px,
            beta,
            steps,
            n_core,
            n_bg,
            wavelength_nm,
            w_um,
            gap_um,
            target,
            heaters=heaters,
            dndt_core=dndt_core,
            dndt_clad=dndt_clad,
            polarization=polarization,
            etch=etch,
            heat_circuit=heat_circuit,
        )

    rho_f = helmholtz_filter(rho, r_px)
    variants = {
        name: _project_design(rho_f, design, rho0, beta, eta) for name, eta in _ROBUST_ETAS
    }
    nominal = variants["intermediate"]
    dilated = variants["dilated"]
    eroded = variants["eroded"]
    fabricated = litho_etch_surrogate(nominal, etch_nm / (dx_um * 1e3), rounding_nm / (dx_um * 1e3))
    if design.any():
        fabricated = np.where(design, fabricated, nominal)
    binary = fabricated >= 0.5
    min_pix = max(4, int((mfs_nm / (dx_um * 1e3)) ** 2 * 0.25))
    binary = _drop_speckles(binary, min_pix)

    drc = _drc(binary, dx_um, mfs_nm, min_gap_nm)
    polys = contours_to_polygons(binary, xs, ys, simplify_um=max(dx_um * 0.6, 0.02))
    gds = write_gdsii(polys, cell="TO_COUPLER", layer=1)

    robust = None
    if gap_um is not None:
        robust = _robust_splits(
            {name: _etched(rho_p, design, etch) for name, rho_p in variants.items()},
            xs,
            ys,
            dx_um,
            n_core,
            n_bg,
            wavelength_nm,
            w_um,
            gap_um,
            target,
            polarization=polarization,
        )
        if robust and "intermediate" in robust:
            t_th = robust["intermediate"]["t_through"]
            t_dr = robust["intermediate"]["t_drop"]

    thermal = None
    field_t = None
    if thermo_optic and gap_um is not None:
        thermal = _thermal_report(
            nominal,
            xs,
            ys,
            dx_um,
            n_core,
            n_bg,
            wavelength_nm,
            w_um,
            gap_um,
            target,
            heaters,
            dndt_core,
            dndt_clad,
            guides,
            polarization=polarization,
            heat_circuit=heat_circuit,
        )
        if thermal:
            field_t = thermal.pop("field", None)

    return {
        "filter": "Helmholtz PDE (Lazarov 2011)",
        "projection": "tanh / Heaviside, robust dilated–eroded (Wang 2011)",
        "fabrication": (
            "sum of dilated/intermediate/eroded κ errors in the loop (Piggott 2015; Wang 2011)"
            + (
                "; both heater states, (κ_hot − κ_off)², and the ring resonance shift; "
                "chip-scale heat sampled into the coupler; optical and heat adjoints"
                if thermo_optic
                else ""
            )
            + "; morphological litho-etch inside the sensitivity and on the GDS (not SEM-trained)"
        ),
        "vectorizer": "marching squares + Ramer–Douglas–Peucker",
        "dx_nm": dx_um * 1e3,
        "beta": beta,
        "etas": {name: eta for name, eta in _ROBUST_ETAS},
        "filter_radius_nm": r_px * dx_um * 1e3,
        "steps": steps,
        "kappa_target": target if gap_um is not None else None,
        "t_through": t_th,
        "t_drop": t_dr,
        "kappa": t_dr,
        "grayscale": _grayscale(nominal),
        "drc": drc,
        "history": history,
        "robust": robust,
        "polygons": [[{"x": x, "y": y} for x, y in p] for p in polys[:12]],
        "polygon_count": len(polys),
        "gds_b64": base64.b64encode(gds).decode("ascii"),
        "gds_bytes": len(gds),
        "field": _density_field(xs, ys, fabricated, polys, guides, "projected density"),
        "field_dilated": _density_field(xs, ys, dilated, [], guides, "dilated"),
        "field_eroded": _density_field(xs, ys, eroded, [], guides, "eroded"),
        "field_ez": _with_guides(field_ez, guides),
        "eta_curve": _eta_split_curve(
            rho_f,
            design,
            rho0,
            beta,
            xs,
            ys,
            dx_um,
            n_core,
            n_bg,
            wavelength_nm,
            w_um,
            gap_um,
            target,
            polarization=polarization,
            etch=etch,
        )
        if gap_um is not None
        else None,
        "thermo_optic": bool(thermo_optic),
        "thermal": thermal,
        "field_t": field_t,
    }


_ETA_SWEEP = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80)
_SWEEP_MFS_NM = (100.0, 150.0, 220.0)


def run_robustness_sweep(
    width_nm: float,
    height_nm: float,
    wavelength_nm: float,
    n_clad: float,
    polarization: str,
    devices: list[dict],
    steps: int = 4,
    kappa_target: float | None = None,
    beta: float = 8.0,
) -> dict:
    """Compare in-loop robust TO vs intermediate-only across DUV filter radii."""
    scene, w_um, length_um, gap_um = _coupler_scene(width_nm, devices)
    if scene is None or gap_um is None:
        return {"error": "Place a coupler to sweep dilated/eroded robustness."}

    length_um = min(length_um, 3.6)
    mode = analyze_waveguide(
        width_nm, height_nm, wavelength_nm, n_clad, polarization, length_um=100.0
    )
    n_core = float(mode["n_eff"]) if mode else 2.45
    n_bg = float(n_clad)
    target = float(np.clip(kappa_target if kappa_target is not None else 0.12, 0.05, 0.95))
    steps = int(max(1, min(steps, 8)))
    beta = float(max(1.0, min(beta, 32.0)))

    rows: list[dict] = []
    eta_curve = None
    for mfs_nm in _SWEEP_MFS_NM:
        for robust in (True, False):
            row = _sweep_case(
                width_nm,
                w_um,
                length_um,
                gap_um,
                mfs_nm,
                beta,
                steps,
                n_core,
                n_bg,
                wavelength_nm,
                target,
                robust,
                polarization,
            )
            rows.append(row)
            if robust and abs(mfs_nm - 150.0) < 1e-6:
                eta_curve = row.get("eta_curve")

    return {
        "kind": "dilated-eroded sweep",
        "kappa_target": target,
        "steps": steps,
        "mfs_nm": list(_SWEEP_MFS_NM),
        "modes": ["robust", "intermediate"],
        "rows": rows,
        "eta_curve": eta_curve,
        "note": (
            "Each row is a short TO at one Helmholtz radius (from DUV MFS). "
            "Robust uses worst dilated/intermediate/eroded κ in the loop; "
            "intermediate only scores η = 0.5, then dilated/eroded are evaluated after."
        ),
    }


def helmholtz_filter(rho: np.ndarray, r_px: float, niter: int = 48, clip: bool = True) -> np.ndarray:
    """(I − r²∇²) ρ̃ = ρ with Neumann edges. Lazarov & Sigmund 2011.

    The stencil is self-adjoint, so the same solve maps a ∂J/∂ρ̃ sensitivity
    back to ∂J/∂ρ. Pass ``clip=False`` for that gradient; densities stay in [0, 1].
    """
    rhs = np.asarray(rho, dtype=float)
    if r_px < 0.35:
        return np.clip(rhs, 0.0, 1.0) if clip else rhs.copy()
    r2 = float(r_px) ** 2

    def apply(field):
        return (1.0 + 4.0 * r2) * field - r2 * neighbor_sum(field)

    u, ok = bicgstab_field(apply, rhs, tol=1e-8, maxiter=max(int(niter), 80))
    if not ok or not np.isfinite(u).all():
        den = 1.0 + 4.0 * r2
        u = rhs.copy()
        for _ in range(int(niter)):
            u = (rhs + r2 * neighbor_sum(u)) / den
    if clip:
        u = np.clip(u, 0.0, 1.0)
    return u


def tanh_project(rho: np.ndarray, beta: float, eta: float = 0.5) -> np.ndarray:
    """Hyperbolic-tangent projection. Wang, Lazarov & Sigmund 2011."""
    b = float(beta)
    e = float(eta)
    num = np.tanh(b * e) + np.tanh(b * (rho - e))
    den = np.tanh(b * e) + np.tanh(b * (1.0 - e))
    return np.clip(num / max(den, 1e-12), 0.0, 1.0)


def project_grad(rho: np.ndarray, beta: float, eta: float = 0.5) -> np.ndarray:
    b = float(beta)
    e = float(eta)
    den = np.tanh(b * e) + np.tanh(b * (1.0 - e))
    t = np.tanh(b * (rho - e))
    return b * (1.0 - t * t) / max(den, 1e-12)


def _etch_radius(round_px: float) -> float:
    return max(float(round_px), 0.35)


def _etch_eta(etch_px: float) -> float:
    return float(np.clip(0.5 + 0.25 * np.tanh(float(etch_px) * 0.7), 0.15, 0.85))


def _etched(rho_proj: np.ndarray, design: np.ndarray, etch) -> np.ndarray:
    """Litho-etch inside the design window. ``etch`` is ``(etch_px, round_px)``."""
    if etch is None:
        return rho_proj
    etched = litho_etch_surrogate(rho_proj, etch[0], etch[1])
    return np.where(design, etched, rho_proj) if design.any() else etched


def _etch_vjp(v: np.ndarray, rho: np.ndarray, etch_px: float, round_px: float) -> np.ndarray:
    """Map ∂J/∂ρ_etched back through litho_etch_surrogate."""
    radius = _etch_radius(round_px)
    rounded = helmholtz_filter(rho, radius)
    sens = np.asarray(v, dtype=float) * project_grad(rounded, 10.0, _etch_eta(etch_px))
    return helmholtz_filter(sens, radius, clip=False)


def litho_etch_surrogate(rho: np.ndarray, etch_px: float, round_px: float) -> np.ndarray:
    """Over/under-etch (threshold shift) and corner rounding (Helmholtz + reproject).

    Stand-in for SEM-trained litho/etch nets (Gostimirovic 2022) and FAID etch
    models (Raza 2024). Positive etch_px is over-etch (Si shrinks).
    """
    rounded = helmholtz_filter(rho, _etch_radius(round_px))
    return tanh_project(rounded, beta=10.0, eta=_etch_eta(etch_px))


def contours_to_polygons(
    binary: np.ndarray, xs: np.ndarray, ys: np.ndarray, simplify_um: float
) -> list[list[tuple[float, float]]]:
    segs = _marching_squares(binary.astype(float), xs, ys, iso=0.5)
    loops = _stitch(segs, snap=max(simplify_um * 0.25, 1e-4))
    out: list[list[tuple[float, float]]] = []
    min_area = max(simplify_um * simplify_um * 8.0, 0.05)
    for loop in loops:
        if len(loop) < 4:
            continue
        simp = rdp(loop, simplify_um)
        if len(simp) >= 3:
            if simp[0] != simp[-1]:
                simp.append(simp[0])
            if abs(_area(simp)) >= min_area:
                out.append(simp)
    out.sort(key=lambda p: -abs(_area(p)))
    return out


def rdp(points: list[tuple[float, float]], eps: float) -> list[tuple[float, float]]:
    """Ramer–Douglas–Peucker. Ramer 1972 / Douglas & Peucker 1973."""
    if len(points) < 3 or eps <= 0:
        return list(points)
    closed = math.hypot(points[0][0] - points[-1][0], points[0][1] - points[-1][1]) < max(eps * 0.25, 1e-9)
    body = points[:-1] if closed else points
    simp = _rdp(body, eps)
    if closed and simp and simp[0] != simp[-1]:
        simp.append(simp[0])
    return simp


def _rdp(points: list[tuple[float, float]], eps: float) -> list[tuple[float, float]]:
    if len(points) < 3:
        return list(points)
    a, b = points[0], points[-1]
    dx, dy = b[0] - a[0], b[1] - a[1]
    den = math.hypot(dx, dy) or 1e-30
    best_i = 0
    best_d = 0.0
    for i in range(1, len(points) - 1):
        px, py = points[i]
        d = abs(dx * (a[1] - py) - dy * (a[0] - px)) / den
        if d > best_d:
            best_d = d
            best_i = i
    if best_d <= eps:
        return [a, b]
    left = _rdp(points[: best_i + 1], eps)
    right = _rdp(points[best_i:], eps)
    return left[:-1] + right


def write_gdsii(polygons: list[list[tuple[float, float]]], cell: str, layer: int) -> bytes:
    """Minimal GDSII: one cell of BOUNDARY polygons on `layer` (nm database units)."""
    now = datetime.now(timezone.utc)
    date = _gds_date(now) + _gds_date(now)
    buf = bytearray()
    buf += _rec(0x00, 0x02, struct.pack(">H", 600))
    buf += _rec(0x01, 0x02, date)
    buf += _rec(0x02, 0x06, _gds_str("PHOTONIC_TO"))
    buf += _rec(0x03, 0x05, _gds_real(0.001) + _gds_real(1e-9))
    buf += _rec(0x05, 0x02, date)
    buf += _rec(0x06, 0x06, _gds_str(cell))
    for poly in polygons:
        if len(poly) < 4:
            continue
        xy = []
        for x, y in poly:
            xy.append(int(round(x * 1000)))
            xy.append(int(round(y * 1000)))
        if xy[:2] != xy[-2:]:
            xy += xy[:2]
        if len(xy) // 2 > 8190:
            continue
        buf += _rec(0x08, 0x00, b"")
        buf += _rec(0x0D, 0x02, struct.pack(">H", layer))
        buf += _rec(0x0E, 0x02, struct.pack(">H", 0))
        buf += _rec(0x10, 0x03, struct.pack(">" + "i" * len(xy), *xy))
        buf += _rec(0x11, 0x00, b"")
    buf += _rec(0x07, 0x00, b"")
    buf += _rec(0x04, 0x00, b"")
    return bytes(buf)


def _coupler_scene(width_nm: float, devices: list[dict]):
    wgs = [d for d in devices if d.get("type") == "waveguide"]
    couplers = [d for d in devices if d.get("type") == "coupler"]
    if not couplers and not wgs:
        return None, 0.0, 0.0, None
    w = width_nm * 1e-3
    length = float(couplers[0]["length_um"]) if couplers else min(16.0, float(wgs[0].get("length_um", 12)))
    gap = float(couplers[0]["gap_nm"]) * 1e-3 if couplers else None
    scene = Scene(background="clad")
    scene.add(Rect(-w, 0.0, w, length, "si", "through"))
    if gap is not None:
        scene.add(Rect(gap, 0.0, w, length, "si", "drop"))
    return scene, w, length, gap


def _dx_for(w_um: float, length_um: float, gap_um: float | None) -> float:
    pad = 0.8
    xspan = w_um + (gap_um + w_um if gap_um is not None else 0.0) + pad
    yspan = length_um + pad
    dx = 0.04
    while (xspan / dx) * (yspan / dx) > 10000:
        dx *= 1.08
    return dx


def _raster(dx_um: float, w_um: float, length_um: float, gap_um: float | None, pad: float = 0.4):
    x1 = (gap_um + w_um if gap_um is not None else 0.0) + pad
    xs = np.arange(-w_um - pad, x1 + dx_um * 0.5, dx_um)
    ys = np.arange(-pad, length_um + pad + dx_um * 0.5, dx_um)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    rho = ((xx >= -w_um) & (xx <= 0.0) & (yy >= 0.0) & (yy <= length_um)).astype(float)
    if gap_um is not None:
        rho = np.maximum(
            rho,
            ((xx >= gap_um) & (xx <= gap_um + w_um) & (yy >= 0.0) & (yy <= length_um)).astype(float),
        )
    design = np.zeros_like(rho, dtype=bool)
    if gap_um is not None:
        inner = min(0.22, 0.5 * w_um)
        bite = min(0.08, 0.35 * gap_um)
        y_lo = 0.2
        y_hi = length_um - 0.2
        along = (yy >= y_lo) & (yy <= y_hi)
        through_inner = (xx >= -inner) & (xx <= bite)
        drop_inner = (xx >= gap_um - bite) & (xx <= gap_um + inner)
        design = along & (through_inner | drop_inner)
    guides = [
        {"x0": float(-w_um), "y0": 0.0, "width": float(w_um), "height": float(length_um)},
    ]
    if gap_um is not None:
        guides.append(
            {"x0": float(gap_um), "y0": 0.0, "width": float(w_um), "height": float(length_um)}
        )
    return xs, ys, rho, design, guides


def _seed_swg(rho: np.ndarray, design: np.ndarray, ys: np.ndarray, dx_um: float) -> np.ndarray:
    """Hammood-style SWG: periodic notches on the inner waveguide sidewalls."""
    period = max(4, int(round(0.40 / dx_um)))
    teeth = (np.arange(ys.size) % period) < max(2, int(round(period * 0.4)))
    out = rho.copy()
    out[design & np.broadcast_to(teeth, out.shape)] = 0.0
    return out


def _project_design(rho_f, design, rho_fixed, beta, eta):
    rho_p = tanh_project(rho_f, beta, eta)
    return np.where(design, rho_p, rho_fixed) if design.any() else rho_p


def _eta_split_curve(
    rho_f,
    design,
    rho_fixed,
    beta,
    xs,
    ys,
    dx_um,
    n_core,
    n_bg,
    wavelength_nm,
    w_um,
    gap_um,
    target,
    etas=_ETA_SWEEP,
    polarization="TM",
    etch=None,
):
    if gap_um is None:
        return None
    curve = []
    worst = None
    worst_err = -1.0
    for eta in etas:
        rho_p = _etched(_project_design(rho_f, design, rho_fixed, beta, eta), design, etch)
        split, _, ok = _fdfd_split(
            rho_p,
            xs,
            ys,
            dx_um,
            n_core,
            n_bg,
            wavelength_nm,
            w_um,
            gap_um,
            polarization=polarization,
        )
        if not ok or not split:
            continue
        t_drop = float(split[1])
        err = abs(t_drop - target)
        rec = {
            "eta": float(eta),
            "t_through": float(split[0]),
            "t_drop": t_drop,
            "abs_dkappa": float(err),
        }
        curve.append(rec)
        if err >= worst_err:
            worst_err = err
            worst = rec
    if not curve:
        return None
    return {
        "points": curve,
        "worst_eta": worst["eta"] if worst else None,
        "worst_abs_dkappa": worst["abs_dkappa"] if worst else None,
        "spread": float(max(p["t_drop"] for p in curve) - min(p["t_drop"] for p in curve)),
    }


def _sweep_case(
    width_nm,
    w_um,
    length_um,
    gap_um,
    mfs_nm,
    beta,
    steps,
    n_core,
    n_bg,
    wavelength_nm,
    target,
    robust: bool,
    polarization: str = "TM",
):
    dx_um = _dx_for(w_um, length_um, gap_um)
    xs, ys, rho0, design, _guides = _raster(dx_um, w_um, length_um, gap_um)
    r_px = float(np.clip(0.5 * mfs_nm / (dx_um * 1e3), 0.8, 2.4))
    rho = rho0.copy()
    if design.any():
        rho = _seed_swg(rho, design, ys, dx_um)
    rho, history, t_th, t_dr, _field = _optimize(
        rho,
        design,
        xs,
        ys,
        dx_um,
        r_px,
        beta,
        steps,
        n_core,
        n_bg,
        wavelength_nm,
        w_um,
        gap_um,
        target,
        robust=robust,
        polarization=polarization,
    )
    rho_f = helmholtz_filter(rho, r_px)
    variants = {name: _project_design(rho_f, design, rho0, beta, eta) for name, eta in _ROBUST_ETAS}
    splits = _robust_splits(
        variants,
        xs,
        ys,
        dx_um,
        n_core,
        n_bg,
        wavelength_nm,
        w_um,
        gap_um,
        target,
        polarization=polarization,
    )
    eta_curve = _eta_split_curve(
        rho_f,
        design,
        rho0,
        beta,
        xs,
        ys,
        dx_um,
        n_core,
        n_bg,
        wavelength_nm,
        w_um,
        gap_um,
        target,
        polarization=polarization,
    )
    mid = splits.get("intermediate") if splits else None
    worst_abs = None
    if splits:
        vals = []
        for name, _eta in _ROBUST_ETAS:
            if name in splits:
                vals.append(abs(splits[name]["t_drop"] - target))
        if vals:
            worst_abs = float(max(vals))
    return {
        "mfs_nm": float(mfs_nm),
        "filter_radius_nm": float(r_px * dx_um * 1e3),
        "mode": "robust" if robust else "intermediate",
        "t_through": t_th,
        "t_drop": mid["t_drop"] if mid else t_dr,
        "worst": splits.get("worst") if splits else None,
        "t_drop_worst": splits.get("t_drop_worst") if splits else None,
        "worst_abs_dkappa": worst_abs,
        "dilated": splits.get("dilated") if splits else None,
        "intermediate": mid,
        "eroded": splits.get("eroded") if splits else None,
        "history": history,
        "eta_curve": eta_curve,
        "dx_nm": dx_um * 1e3,
    }


def _coupled_hot(rho_window, xs, ys, heat):
    """Chip heat for one blueprint, and the compact ring shift that heat produces."""
    rho_c, inside, temp_c, temp_w = coupled_window_heat(
        heat["xs"], heat["ys"], heat["rho_fixed"], heat["q"], rho_window, xs, ys
    )
    t_ring = mean_on_mask(temp_c, heat["ring"])
    return {
        "rho_c": rho_c,
        "inside": inside,
        "temp_c": temp_c,
        "temp": temp_w,
        "t_ring": t_ring,
        "shift": float(heat["dldt"]) * t_ring,
    }


def _optimize(
    rho,
    design,
    xs,
    ys,
    dx_um,
    r_px,
    beta_final,
    steps,
    n_core,
    n_bg,
    wavelength_nm,
    w_um,
    gap_um,
    target,
    robust: bool = True,
    heaters: list | None = None,
    dndt_core: float = 0.0,
    dndt_clad: float = 0.0,
    polarization: str = "TM",
    etch=None,
    heat_circuit=None,
):
    """Sum the κ error of every scored blueprint, then step on that gradient.

    With heaters, each blueprint adds heater-off, heater-on, (κ_hot − κ_off)²,
    and the ring resonance shift. Heat is the chip-scale solve; the coupler
    window samples that temperature. The grayscale penalty stays on the
    intermediate projection before etch.
    """
    history = []
    field_ez = None
    t_th = t_dr = None
    npml = max(8, int(round(0.32 / dx_um)))
    etas = _ROBUST_ETAS if robust else (("intermediate", 0.5),)
    heaters = heaters or []
    for k in range(steps):
        beta = min(beta_final, 2.0 * (1.22**k))
        rho_f = helmholtz_filter(rho, r_px)
        q_heat = paint_heaters(xs, ys, heaters) if heaters else None
        candidates = []
        for name, eta in etas:
            rho_proj = _project_design(rho_f, design, rho, beta, eta)
            rho_sim = _etched(rho_proj, design, etch)
            temps = [("off", None)]
            hot_state = None
            if heaters:
                if heat_circuit is not None:
                    hot_state = _coupled_hot(rho_sim, xs, ys, heat_circuit)
                    temps.append(("hot", hot_state["temp"]))
                elif q_heat is not None:
                    temps.append(("hot", solve_heat(xs, ys, q_heat, sheet_k(rho_sim))))
            for tname, temp in temps:
                split, e_fwd, ok = _fdfd_split(
                    rho_sim,
                    xs,
                    ys,
                    dx_um,
                    n_core,
                    n_bg,
                    wavelength_nm,
                    w_um,
                    gap_um,
                    npml,
                    None,
                    temp,
                    dndt_core,
                    dndt_clad,
                    polarization=polarization,
                )
                if not ok or split is None:
                    continue
                t_drop = float(split[1])
                candidates.append(
                    {
                        "name": name,
                        "eta": eta,
                        "thermal": tname,
                        "temp": temp,
                        "rho_p": rho_sim,
                        "rho_proj": rho_proj,
                        "split": split,
                        "e_fwd": e_fwd,
                        "t_drop": t_drop,
                        "err": (t_drop - target) ** 2,
                        "heat": hot_state if tname == "hot" else None,
                        "shift_nm": hot_state["shift"] if tname == "hot" and hot_state else 0.0,
                        "dt_ring_k": hot_state["t_ring"] if tname == "hot" and hot_state else 0.0,
                    }
                )
        if not candidates:
            break
        worst = max(candidates, key=lambda c: c["err"])
        by_key = {(c["name"], c["thermal"]): c for c in candidates}
        mid = by_key.get(("intermediate", "off")) or by_key.get(("intermediate", "hot"), worst)
        t_th, t_dr = mid["split"]
        gray_rho = mid["rho_proj"]
        gray = _grayscale(gray_rho[design]) if design.any() else _grayscale(gray_rho)
        gray_w = 0.02 * (beta / max(beta_final, 1.0))
        scored = 0.0
        drift = 0.0
        pieces = []
        complete = True
        heat_used = False
        for name, _eta in etas:
            off_c = by_key.get((name, "off"))
            hot_c = by_key.get((name, "hot"))
            if off_c is None or (heaters and hot_c is None):
                complete = False
                continue
            scored += _pair_objective(off_c, hot_c, target)
            if hot_c is not None:
                scored += hot_c.get("shift_nm", 0.0) ** 2
            if off_c is not None and hot_c is not None:
                drift += (hot_c["t_drop"] - off_c["t_drop"]) ** 2
            heat_used = heat_used or hot_c is not None
            piece = _desensitization_sensitivity(
                off_c,
                hot_c,
                rho_f,
                design,
                xs,
                ys,
                dx_um,
                r_px,
                beta,
                n_core,
                n_bg,
                wavelength_nm,
                w_um,
                gap_um,
                target,
                npml,
                dndt_core,
                dndt_clad,
                polarization=polarization,
                etch=etch,
                heat_circuit=heat_circuit,
            )
            if piece is None:
                complete = False
            else:
                pieces.append(piece)
        obj = scored + gray_w * gray
        rec = {
            "step": k + 1,
            "objective": float(obj),
            "t_drop": float(worst["t_drop"]),
            "worst": worst["name"],
            "thermal": worst["thermal"],
            "drift": float(drift),
            "beta": float(beta),
        }
        for name, _eta in _ROBUST_ETAS:
            off = by_key.get((name, "off"))
            hot = by_key.get((name, "hot"))
            if off:
                rec[f"t_drop_{name}"] = off["t_drop"]
            if hot:
                rec[f"t_drop_{name}_hot"] = hot["t_drop"]
        if by_key.get(("intermediate", "off")) and by_key.get(("intermediate", "hot")):
            rec["t_drop_off"] = by_key[("intermediate", "off")]["t_drop"]
            rec["t_drop_hot"] = by_key[("intermediate", "hot")]["t_drop"]
            rec["shift_nm"] = by_key[("intermediate", "hot")].get("shift_nm", 0.0)
            rec["dt_ring_k"] = by_key[("intermediate", "hot")].get("dt_ring_k", 0.0)
        history.append(rec)
        if mid["e_fwd"] is not None:
            field_ez = _ez_field(xs, ys, mid["e_fwd"])
        if not complete or not pieces:
            continue
        sens = pieces[0]
        for piece in pieces[1:]:
            sens = sens + piece
        sens = sens + _grayscale_sensitivity(rho_f, design, beta, gray_w, r_px)
        rec["gradient"] = "optical+heat" if heat_used else "optical"
        peak = float(np.max(np.abs(sens[design]))) if design.any() else 1.0
        step = 0.05 * sens / (peak + 1e-12)
        updated = np.clip(rho - step, 0.0, 1.0)
        rho = np.where(design, 0.75 * rho + 0.25 * updated, rho)
    return rho, history, t_th, t_dr, field_ez


def _thermal_report(
    rho,
    xs,
    ys,
    dx_um,
    n_core,
    n_bg,
    wavelength_nm,
    w_um,
    gap_um,
    target,
    heaters,
    dndt_core,
    dndt_clad,
    guides,
    polarization="TM",
    heat_circuit=None,
):
    if not heaters:
        return None
    hot_state = _coupled_hot(rho, xs, ys, heat_circuit) if heat_circuit is not None else None
    t_hot = hot_state["temp"] if hot_state is not None else solve_heat(
        xs, ys, paint_heaters(xs, ys, heaters), sheet_k(rho)
    )
    npml = max(8, int(round(0.32 / dx_um)))
    cold, _, cok = _fdfd_split(
        rho,
        xs,
        ys,
        dx_um,
        n_core,
        n_bg,
        wavelength_nm,
        w_um,
        gap_um,
        npml,
        polarization=polarization,
    )
    hot, _, hok = _fdfd_split(
        rho,
        xs,
        ys,
        dx_um,
        n_core,
        n_bg,
        wavelength_nm,
        w_um,
        gap_um,
        npml,
        None,
        t_hot,
        dndt_core,
        dndt_clad,
        polarization=polarization,
    )
    t_off = float(cold[1]) if cok and cold else None
    t_on = float(hot[1]) if hok and hot else None
    heat_guides = list(guides)
    heat_guides.extend(
        {
            "x0": float(h["x0"]),
            "y0": float(h["y0"]),
            "width": float(h["width"]),
            "height": float(h["height"]),
        }
        for h in heaters
    )
    return {
        "source": "2D BOX-sink heat PDE + FDFD ε(T)",
        "layout": heaters,
        "power_mw": float(sum(h["power_mw"] for h in heaters)),
        "dt_max_k": float(hot_state["temp_c"].max()) if hot_state is not None else float(t_hot.max()),
        "dt_ring_k": hot_state["t_ring"] if hot_state is not None else None,
        "shift_nm": hot_state["shift"] if hot_state is not None else None,
        "t_drop_off": t_off,
        "t_drop_hot": t_on,
        "abs_dkappa": abs((t_on or 0.0) - (t_off or 0.0)) if t_off is not None and t_on is not None else None,
        "kappa_target": float(target),
        "note": (
            "Heat is the chip-scale BOX-sink solve, with this blueprint’s litho-etched "
            "silicon pasted into the coupler window. The coupler field uses that temperature. "
            "The ring shift is (λ / n_g) dn_eff/dT · ΔT_ring from the same heat solution, "
            "and the heat adjoint carries both the coupler and the ring back to the density."
        ),
        "field": heat_field(
            heat_circuit["xs"] if hot_state is not None else xs,
            heat_circuit["ys"] if hot_state is not None else ys,
            hot_state["temp_c"] if hot_state is not None else t_hot,
            heat_guides,
            heater_polygons(heaters),
        ),
    }


def _robust_splits(
    variants,
    xs,
    ys,
    dx_um,
    n_core,
    n_bg,
    wavelength_nm,
    w_um,
    gap_um,
    target,
    polarization="TM",
):
    npml = max(8, int(round(0.32 / dx_um)))
    out = {}
    worst_name = None
    worst_err = -1.0
    for name, rho_p in variants.items():
        split, _, ok = _fdfd_split(
            rho_p,
            xs,
            ys,
            dx_um,
            n_core,
            n_bg,
            wavelength_nm,
            w_um,
            gap_um,
            npml,
            polarization=polarization,
        )
        if not ok or not split:
            continue
        out[name] = {"t_through": split[0], "t_drop": split[1]}
        err = (split[1] - target) ** 2
        if err >= worst_err:
            worst_err = err
            worst_name = name
    if worst_name:
        out["worst"] = worst_name
        out["t_drop_worst"] = out[worst_name]["t_drop"]
    return out or None


def _eps_pml(
    rho, xs, ys, dx_um, n_core, n_bg, wavelength_nm, npml=8, temp=None, dndt_core=0.0, dndt_clad=0.0
):
    """Permittivity on a μm grid. Small background loss + PML keeps FDFD stable."""
    if temp is None:
        n_c, n_b = n_core, n_bg
    else:
        t = np.asarray(temp, dtype=float)
        n_c = n_core + float(dndt_core) * t
        n_b = n_bg + float(dndt_clad) * t
    eps_r = n_b**2 + np.clip(rho, 0, 1) * (n_c**2 - n_b**2)
    lam = wavelength_nm * 1e-3
    k0 = 2 * math.pi / lam
    sigma = np.zeros(eps_r.shape, dtype=float)
    nx, ny = eps_r.shape
    for i in range(npml):
        r = ((npml - i) / npml) ** 2
        s = 1.6 * r
        sigma[i, :] += s
        sigma[-1 - i, :] += s
    for j in range(npml):
        r = ((npml - j) / npml) ** 2
        s = 1.6 * r
        sigma[:, j] += s
        sigma[:, -1 - j] += s
    # Material loss scales with the real permittivity. The PML sponge does not.
    eps = eps_r * (1.0 + 1j * _MATERIAL_LOSS) + 1j * sigma * (float(n_core) ** 2)
    return eps, k0, npml


def _jsrc_jmon(ny: int, npml: int, dx_um: float) -> tuple[int, int]:
    jsrc = npml + 3
    span = min(2.6, max(1.2, (ny - 2 * npml - 8) * dx_um))
    jmon = jsrc + max(8, int(round(span / dx_um)))
    jmon = min(jmon, ny - npml - 4)
    return jsrc, max(jmon, jsrc + 8)


def _fdfd_split(
    rho,
    xs,
    ys,
    dx_um,
    n_core,
    n_bg,
    wavelength_nm,
    w_um,
    gap_um,
    npml=8,
    x0=None,
    temp=None,
    dndt_core=0.0,
    dndt_clad=0.0,
    *,
    polarization="TM",
    tol=_SOLVE_TOL,
    maxiter=_SOLVE_ITERS,
):
    src = _src_profile(xs, w_um)
    e, ok = _fdfd_solve(
        rho,
        xs,
        ys,
        dx_um,
        n_core,
        n_bg,
        wavelength_nm,
        src,
        npml,
        x0,
        temp,
        dndt_core,
        dndt_clad,
        polarization=polarization,
        tol=tol,
        maxiter=maxiter,
    )
    if not ok:
        return None, e, False
    jsrc, jmon = _jsrc_jmon(ys.size, npml, dx_um)
    th = np.abs(xs + w_um / 2) <= w_um * 0.85
    dr = np.abs(xs - (gap_um + w_um / 2)) <= w_um * 0.85
    if np.abs(e[:, jmon]).max() < 1e-3 * (np.abs(e[:, jsrc]).max() + 1e-30):
        return None, e, False
    pth = float(np.sum(np.abs(e[th, jmon]) ** 2))
    pdr = float(np.sum(np.abs(e[dr, jmon]) ** 2))
    tot = pth + pdr + 1e-30
    return (pth / tot, pdr / tot), e, True


def _src_profile(xs, w_um):
    src = np.zeros((xs.size, 1), dtype=complex)
    x0 = -w_um / 2
    src[:, 0] = np.exp(-((xs - x0) / (0.45 * w_um)) ** 2)
    return src


def _fdfd_solve(
    rho,
    xs,
    ys,
    dx_um,
    n_core,
    n_bg,
    wavelength_nm,
    src_x,
    npml=8,
    x0=None,
    temp=None,
    dndt_core=0.0,
    dndt_clad=0.0,
    polarization="TM",
    tol=_SOLVE_TOL,
    maxiter=_SOLVE_ITERS,
):
    eps, k0, npml = _eps_pml(
        rho, xs, ys, dx_um, n_core, n_bg, wavelength_nm, npml, temp, dndt_core, dndt_clad
    )
    dx = float(dx_um)
    jsrc, _ = _jsrc_jmon(eps.shape[1], npml, dx)
    b = np.zeros_like(eps, dtype=complex)
    scale = k0**2
    if src_x.ndim == 1:
        b[:, jsrc] = src_x * scale
    elif src_x.shape[1] == 1:
        b[:, jsrc] = src_x[:, 0] * scale
    else:
        b += src_x * scale

    apply = te_apply if str(polarization).upper() == "TE" else helmholtz_apply
    e, ok = _cocg(
        lambda u: apply(u, eps, k0, dx),
        -b,
        x0=x0,
        tol=tol,
        maxiter=maxiter,
    )
    return e, bool(ok and np.isfinite(e).all())


def _port_masks(xs, w_um, gap_um):
    through = np.abs(xs + w_um / 2) <= w_um * 0.85
    drop = np.abs(xs - (gap_um + w_um / 2)) <= w_um * 0.85
    return through, drop


def _pair_objective(off, hot, target) -> float:
    """(T_off − target)² + (T_hot − target)² + (T_hot − T_off)² for the states that exist."""
    total = 0.0
    if off is not None:
        total += (off["t_drop"] - target) ** 2
    if hot is not None:
        total += (hot["t_drop"] - target) ** 2
    if off is not None and hot is not None:
        total += (hot["t_drop"] - off["t_drop"]) ** 2
    return float(total)


def _state_weight(corner, other, target, track_target):
    """∂J/∂T_drop for one state of _pair_objective, when track_target is set."""
    weight = 2.0 * (corner["t_drop"] - target) if track_target else 0.0
    if other is None:
        return weight
    t_hot = corner["t_drop"] if corner["thermal"] == "hot" else other["t_drop"]
    t_off = other["t_drop"] if corner["thermal"] == "hot" else corner["t_drop"]
    sign = 1.0 if corner["thermal"] == "hot" else -1.0
    return weight + sign * 2.0 * (t_hot - t_off)


def _desensitization_sensitivity(
    off,
    hot,
    rho_f,
    design,
    xs,
    ys,
    dx_um,
    r_px,
    beta,
    n_core,
    n_bg,
    wavelength_nm,
    w_um,
    gap_um,
    target,
    npml,
    dndt_core,
    dndt_clad,
    adj_tol=_SOLVE_TOL,
    polarization="TM",
    etch=None,
    heat_circuit=None,
):
    """Density gradient of one blueprint: both temperatures, including the drift.

    Each existing state is differentiated with track_target, which is the
    derivative of _pair_objective. A missing adjoint drops the whole blueprint.
    """
    acc = None
    for corner, other in ((off, hot), (hot, off)):
        if corner is None:
            continue
        piece = _density_sensitivity(
            corner,
            rho_f,
            design,
            xs,
            ys,
            dx_um,
            r_px,
            beta,
            n_core,
            n_bg,
            wavelength_nm,
            w_um,
            gap_um,
            target,
            npml,
            dndt_core,
            dndt_clad,
            dJ_dt=_state_weight(corner, other, target, True),
            adj_tol=adj_tol,
            polarization=polarization,
            etch=etch,
            heat_circuit=heat_circuit,
        )
        if piece is None:
            return None
        acc = piece if acc is None else acc + piece
    return acc


def _density_sensitivity(
    corner,
    rho_f,
    design,
    xs,
    ys,
    dx_um,
    r_px,
    beta,
    n_core,
    n_bg,
    wavelength_nm,
    w_um,
    gap_um,
    target,
    npml,
    dndt_core,
    dndt_clad,
    dJ_dt=None,
    adj_tol=_SOLVE_TOL,
    polarization="TM",
    etch=None,
    heat_circuit=None,
):
    """∂J/∂ρ for one heater state, with ∂J/∂T_drop = ``dJ_dt``.

    Chains the optical adjoint through ε(ρ, T) of the litho-etched blueprint.
    When the heater is on, the adjoint heat equation differentiates that
    conductivity. The etch map, the projection, and the Helmholtz filter then
    carry the sensitivity back to the design density.
    """
    e_fwd = corner["e_fwd"]
    rho_sim = corner["rho_p"]
    rho_proj = corner.get("rho_proj", rho_sim)
    temp = corner.get("temp")
    eps, k0, _ = _eps_pml(
        rho_sim, xs, ys, dx_um, n_core, n_bg, wavelength_nm, npml, temp, dndt_core, dndt_clad
    )
    _, jmon = _jsrc_jmon(e_fwd.shape[1], npml, dx_um)
    through, drop = _port_masks(xs, w_um, gap_um)
    field_grad = port_field_grad(e_fwd, through, drop, jmon, target, dJ_dt=dJ_dt)
    te = str(polarization).upper() == "TE"
    apply = te_apply if te else helmholtz_apply
    lam, ok = optical_adjoint_field(
        eps,
        k0,
        dx_um,
        field_grad,
        apply=apply,
        solve=lambda op, rhs: _cocg(op, rhs, tol=adj_tol, maxiter=_SOLVE_ITERS),
    )
    if not ok:
        return None
    if te:
        dJ_deps = te_permittivity_gradient(e_fwd, lam, eps, k0, dx_um, loss=_MATERIAL_LOSS)
    else:
        dJ_deps = permittivity_gradient(e_fwd, lam, k0, loss=_MATERIAL_LOSS)
    hot = corner.get("heat")
    if hot is not None and heat_circuit is not None:
        d_rho, d_t = permittivity_derivatives(
            rho_sim, temp, n_core, n_bg, dndt_core, dndt_clad
        )
        direct = np.asarray(dJ_deps, dtype=float) * d_rho
        dJ_dT = np.asarray(dJ_deps, dtype=float) * (0.0 if d_t is None else d_t)
        thermal = window_heat_gradient(
            heat_circuit["xs"],
            heat_circuit["ys"],
            hot["rho_c"],
            hot["inside"],
            hot["temp_c"],
            xs,
            ys,
            dJ_dT,
            heat_circuit["ring"],
            2.0 * float(hot["shift"]) * float(heat_circuit["dldt"]),
        )
    else:
        direct, thermal = thermo_optic_terms(
            dJ_deps,
            rho_sim,
            temp,
            rho_sim,
            xs,
            n_core,
            n_bg,
            dndt_core,
            dndt_clad,
        )
    sens_sim = direct + thermal
    if etch is None:
        sens_proj = sens_sim
    else:
        sens_proj = _etch_vjp(np.where(design, sens_sim, 0.0), rho_proj, etch[0], etch[1])
    sens_f = np.where(design, sens_proj * project_grad(rho_f, beta, corner["eta"]), 0.0)
    return helmholtz_filter(sens_f, r_px, clip=False) * design


def _cocg(op, b, x0=None, tol=_SOLVE_TOL, maxiter=_SOLVE_ITERS):
    """COCG for the complex-symmetric TM and TE operators. Success means the residual."""
    b = np.asarray(b)
    x = np.zeros_like(b) if x0 is None else np.array(x0, copy=True)
    bnorm = np.linalg.norm(b) + 1e-30
    best = x
    best_rel = np.inf
    used = 0
    for _restart in range(6):
        if used >= maxiter:
            break
        r = b - op(x)
        rel = float(np.linalg.norm(r) / bnorm)
        if np.isfinite(rel) and rel < best_rel:
            best, best_rel = x.copy(), rel
        if best_rel <= tol:
            return best, True
        p = r.copy()
        rho = np.sum(r * r)
        while used < maxiter:
            q = op(p)
            den = np.sum(p * q)
            if abs(den) < 1e-18 or abs(rho) < 1e-30:
                break
            alpha = rho / den
            x = x + alpha * p
            r = r - alpha * q
            used += 1
            rel = float(np.linalg.norm(r) / bnorm)
            if np.isfinite(rel) and rel < best_rel:
                best, best_rel = x.copy(), rel
            if best_rel <= tol:
                return best, True
            rho_n = np.sum(r * r)
            p = r + (rho_n / rho) * p
            rho = rho_n
        x = best.copy()
    return best, bool(np.isfinite(best).all() and best_rel <= tol)


def _drc(binary: np.ndarray, dx_um: float, mfs_nm: float, min_gap_nm: float) -> dict:
    dt_si = _chamfer(binary)
    dt_gap = _chamfer(~binary)
    interior = dt_si >= 1.0
    gaps = dt_gap >= 1.0
    min_feat_nm = float(2.0 * dt_si[interior].min() * dx_um * 1e3) if interior.any() else 0.0
    min_gap_meas = float(2.0 * dt_gap[gaps].min() * dx_um * 1e3) if gaps.any() else 0.0
    return {
        "mfs_nm": mfs_nm,
        "min_gap_nm": min_gap_nm,
        "min_feature_nm": min_feat_nm,
        "min_gap_meas_nm": min_gap_meas,
        "mfs_ok": bool(min_feat_nm + 1e-6 >= mfs_nm or not interior.any()),
        "gap_ok": bool(min_gap_meas + 1e-6 >= min_gap_nm or not gaps.any()),
        "fill": float(binary.mean()),
    }


def _chamfer(mask: np.ndarray) -> np.ndarray:
    inf = 1e5
    d = np.where(mask, 0.0, inf)
    nx, ny = d.shape
    for i in range(nx):
        for j in range(ny):
            if i:
                d[i, j] = min(d[i, j], d[i - 1, j] + 1.0)
                if j:
                    d[i, j] = min(d[i, j], d[i - 1, j - 1] + 1.414)
            if j:
                d[i, j] = min(d[i, j], d[i, j - 1] + 1.0)
    for i in range(nx - 1, -1, -1):
        for j in range(ny - 1, -1, -1):
            if i + 1 < nx:
                d[i, j] = min(d[i, j], d[i + 1, j] + 1.0)
                if j + 1 < ny:
                    d[i, j] = min(d[i, j], d[i + 1, j + 1] + 1.414)
            if j + 1 < ny:
                d[i, j] = min(d[i, j], d[i, j + 1] + 1.0)
    d[d >= inf / 2] = 0.0
    return d


def _drop_speckles(binary: np.ndarray, min_pix: int) -> np.ndarray:
    """Remove Si islands smaller than min_pix (post-projection cleanup)."""
    if min_pix <= 1:
        return binary
    seen = np.zeros_like(binary, dtype=bool)
    out = binary.copy()
    nx, ny = binary.shape
    for i in range(nx):
        for j in range(ny):
            if not binary[i, j] or seen[i, j]:
                continue
            stack = [(i, j)]
            cells = []
            seen[i, j] = True
            while stack:
                a, b = stack.pop()
                cells.append((a, b))
                for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    x, y = a + da, b + db
                    if 0 <= x < nx and 0 <= y < ny and binary[x, y] and not seen[x, y]:
                        seen[x, y] = True
                        stack.append((x, y))
            if len(cells) < min_pix:
                for a, b in cells:
                    out[a, b] = False
    return out


def _grayscale(rho: np.ndarray) -> float:
    return float(np.mean(4.0 * rho * (1.0 - rho)))


def _grayscale_sensitivity(rho_f, design, beta, weight, r_px, eta=0.5):
    """∂/∂ρ of weight * mean(4 ρ_p (1 − ρ_p)) on the intermediate projection."""
    if weight == 0.0 or not np.any(design):
        return np.zeros_like(rho_f, dtype=float)
    rho_p = tanh_project(rho_f, beta, eta)
    n = float(np.count_nonzero(design))
    dJ = np.zeros_like(rho_f, dtype=float)
    dJ[design] = weight * 4.0 * (1.0 - 2.0 * rho_p[design]) / n
    sens_f = np.where(design, dJ * project_grad(rho_f, beta, eta), 0.0)
    return helmholtz_filter(sens_f, r_px, clip=False) * design


def _marching_squares(z, xs, ys, iso=0.5):
    nx, ny = z.shape
    segs: list[tuple[tuple[float, float], tuple[float, float]]] = []

    def lerp(p, q, vp, vq):
        t = 0.0 if abs(vq - vp) < 1e-12 else (iso - vp) / (vq - vp)
        t = min(1.0, max(0.0, t))
        return (round(p[0] + t * (q[0] - p[0]), 6), round(p[1] + t * (q[1] - p[1]), 6))

    for i in range(nx - 1):
        for j in range(ny - 1):
            v = (z[i, j], z[i + 1, j], z[i + 1, j + 1], z[i, j + 1])
            bits = sum(1 << k for k, val in enumerate(v) if val >= iso)
            if bits in (0, 15):
                continue
            p = (
                (xs[i], ys[j]),
                (xs[i + 1], ys[j]),
                (xs[i + 1], ys[j + 1]),
                (xs[i], ys[j + 1]),
            )
            e = (
                lerp(p[0], p[1], v[0], v[1]),
                lerp(p[1], p[2], v[1], v[2]),
                lerp(p[2], p[3], v[2], v[3]),
                lerp(p[3], p[0], v[3], v[0]),
            )
            table = {
                1: [(e[3], e[0])],
                2: [(e[0], e[1])],
                3: [(e[3], e[1])],
                4: [(e[1], e[2])],
                5: [(e[3], e[0]), (e[1], e[2])],
                6: [(e[0], e[2])],
                7: [(e[3], e[2])],
                8: [(e[2], e[3])],
                9: [(e[2], e[0])],
                10: [(e[0], e[1]), (e[2], e[3])],
                11: [(e[2], e[1])],
                12: [(e[1], e[3])],
                13: [(e[1], e[0])],
                14: [(e[0], e[3])],
            }
            segs.extend(table.get(bits, []))
    return segs


def _stitch(segs, snap=1e-4):
    unused = list(segs)
    loops: list[list[tuple[float, float]]] = []

    def close(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) < snap * 4

    while unused:
        a, b = unused.pop()
        loop = [a, b]
        changed = True
        while changed:
            changed = False
            for i, (p, q) in enumerate(unused):
                if close(loop[-1], p):
                    loop.append(q)
                    unused.pop(i)
                    changed = True
                    break
                if close(loop[-1], q):
                    loop.append(p)
                    unused.pop(i)
                    changed = True
                    break
        if close(loop[0], loop[-1]):
            loops.append(loop)
        elif len(loop) > 12:
            loop.append(loop[0])
            loops.append(loop)
    return loops


def _area(poly: list[tuple[float, float]]) -> float:
    a = 0.0
    for i in range(len(poly) - 1):
        a += poly[i][0] * poly[i + 1][1] - poly[i + 1][0] * poly[i][1]
    return 0.5 * a


def _plot_guides(guides):
    """Raster (x transverse, y along) → plot (x along, y transverse)."""
    return [
        {
            "x0": float(g["y0"]),
            "y0": float(g["x0"]),
            "width": float(g["height"]),
            "height": float(g["width"]),
        }
        for g in guides
    ]


def _with_guides(field, guides):
    if not field:
        return field
    return {**field, "guides": _plot_guides(guides)}


def _density_field(xs, ys, rho, polys, guides, quantity):
    """Propagation along the plot x-axis so the two rails read as a coupler, not a sliver."""
    return {
        "x_um": ys.tolist(),
        "z_um": xs.tolist(),
        "intensity": np.clip(rho, 0, 1).tolist(),
        "quantity": quantity,
        "colormap": "density",
        "layout": "strip",
        "guides": _plot_guides(guides),
        "polygons": [
            [{"x": y, "y": x} for x, y in p[:: max(1, len(p) // 200)]]
            for p in polys[:8]
        ],
    }


def _ez_field(xs, ys, ez):
    mag = np.abs(ez)
    peak = float(np.max(mag)) or 1.0
    return {
        "x_um": ys.tolist(),
        "z_um": xs.tolist(),
        "intensity": np.clip(mag / peak, 0, 1).tolist(),
        "quantity": "|Ez| FDFD",
        "layout": "strip",
    }


def _rec(rtype: int, dtype: int, data: bytes) -> bytes:
    if len(data) % 2:
        data += b"\x00"
    return struct.pack(">HBB", 4 + len(data), rtype, dtype) + data


def _gds_str(s: str) -> bytes:
    b = s.encode("ascii")[:32]
    if len(b) % 2:
        b += b"\x00"
    return b


def _gds_date(dt: datetime) -> bytes:
    t = (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
    return struct.pack(">6h", *t)


def _gds_real(val: float) -> bytes:
    if val == 0.0:
        return bytes(8)
    sign = 0x80 if val < 0 else 0
    x = abs(val)
    exp = 0
    while x >= 1.0:
        x /= 16.0
        exp += 1
        if exp > 63:
            x = 0.999999999999999
            exp = 63
            break
    while x < 0.0625 and x > 0.0:
        x *= 16.0
        exp -= 1
        if exp < -64:
            return bytes(8)
    mantissa = int(x * (2**56) + 0.5)
    if mantissa >= 2**56:
        mantissa = 2**52
        exp += 1
    return bytes([(exp + 64) & 0x7F | sign]) + mantissa.to_bytes(7, "big")
