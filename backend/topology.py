"""Density-based topology optimization → foundry-aware GDS."""

from __future__ import annotations

import base64
import math
import struct
from datetime import datetime, timezone

import numpy as np

from fdtd.geometry import Rect, Scene
from waveguide import analyze_waveguide

# Dilated / intermediate / eroded thresholds. Wang, Lazarov & Sigmund 2011.
_ROBUST_ETAS = (("dilated", 0.3), ("intermediate", 0.5), ("eroded", 0.7))


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
) -> dict:
    scene, w_um, length_um, gap_um = _coupler_scene(width_nm, devices)
    if scene is None:
        return {"error": "Place a waveguide or coupler to run topology optimization."}

    dx_um = _dx_for(w_um, length_um, gap_um)
    xs, ys, rho0, design, guides = _raster(scene, dx_um, w_um, gap_um)
    r_px = max(0.6, 0.5 * mfs_nm / (dx_um * 1e3))
    beta = float(max(1.0, min(beta, 32.0)))
    steps = int(max(0, min(steps, 20)))

    mode = analyze_waveguide(
        width_nm, height_nm, wavelength_nm, n_clad, polarization, length_um=100.0
    )
    n_core = float(mode["n_eff"]) if mode else 2.45
    n_bg = float(n_clad)

    rho = rho0.copy()
    if design.any() and gap_um is not None:
        rho = _seed_swg(rho, design, ys, dx_um)

    target = 0.5 if kappa_target is None else float(np.clip(kappa_target, 0.05, 0.95))
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
            variants, xs, ys, dx_um, n_core, n_bg, wavelength_nm, w_um, gap_um, target
        )
        if robust and "intermediate" in robust:
            t_th = robust["intermediate"]["t_through"]
            t_dr = robust["intermediate"]["t_drop"]

    return {
        "filter": "Helmholtz PDE (Lazarov 2011)",
        "projection": "tanh / Heaviside, robust dilated–eroded (Wang 2011)",
        "fabrication": "worst-case dilated/eroded κ in the loop (Piggott 2015; Wang 2011); morphological litho-etch after (not SEM-trained)",
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
        "field_ez": field_ez,
    }


def helmholtz_filter(rho: np.ndarray, r_px: float, niter: int = 48) -> np.ndarray:
    """(I − r²∇²) ρ̃ = ρ with Neumann edges. Lazarov & Sigmund 2011."""
    if r_px < 0.35:
        return rho.copy()
    r2 = float(r_px) ** 2
    u = rho.astype(float).copy()
    rhs = rho.astype(float)
    den = 1.0 + 4.0 * r2
    for _ in range(niter):
        pad = np.pad(u, 1, mode="edge")
        nb = pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:]
        u = (rhs + r2 * nb) / den
    return np.clip(u, 0.0, 1.0)


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


def litho_etch_surrogate(rho: np.ndarray, etch_px: float, round_px: float) -> np.ndarray:
    """Over/under-etch (threshold shift) and corner rounding (Helmholtz + reproject).

    Stand-in for SEM-trained litho/etch nets (Gostimirovic 2022) and FAID etch
    models (Raza 2024). Positive etch_px is over-etch (Si shrinks).
    """
    rounded = helmholtz_filter(rho, max(float(round_px), 0.35))
    eta = float(np.clip(0.5 + 0.25 * np.tanh(etch_px * 0.7), 0.15, 0.85))
    return tanh_project(rounded, beta=10.0, eta=eta)


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
    xspan = w_um + (gap_um + w_um if gap_um is not None else 0.0) + 1.2
    yspan = length_um + 1.2
    dx = 0.06
    while (xspan / dx) * (yspan / dx) > 4800:
        dx *= 1.12
    return dx


def _raster(scene: Scene, dx_um: float, w_um: float, gap_um: float | None):
    x0, x1, y0, y1 = scene.bounds()
    xs = np.arange(x0, x1 + dx_um, dx_um)
    ys = np.arange(y0, y1 + dx_um, dx_um)
    rho = np.zeros((xs.size, ys.size))
    design = np.zeros_like(rho, dtype=bool)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            rho[i, j] = 1.0 if scene.material_at(float(x), float(y)) == "si" else 0.0
    if gap_um is not None:
        inner = min(0.14, w_um * 0.35)
        y_lo = ys.min() + 0.35
        y_hi = ys.max() - 0.35
        along = (yy >= y_lo) & (yy <= y_hi)
        through_inner = (xx >= -inner) & (xx <= 0.0)
        drop_inner = (xx >= gap_um) & (xx <= gap_um + inner)
        design = along & (through_inner | drop_inner)
    guides = [
        {"x0": float(-w_um), "y0": float(ys[0]), "width": float(w_um), "height": float(ys[-1] - ys[0])},
    ]
    if gap_um is not None:
        guides.append(
            {
                "x0": float(gap_um),
                "y0": float(ys[0]),
                "width": float(w_um),
                "height": float(ys[-1] - ys[0]),
            }
        )
    return xs, ys, rho, design, guides


def _seed_swg(rho: np.ndarray, design: np.ndarray, ys: np.ndarray, dx_um: float) -> np.ndarray:
    """Hammood-style SWG: periodic notches on the inner waveguide sidewalls."""
    period = max(2, int(round(0.40 / dx_um)))
    teeth = (np.arange(ys.size) % period) < period * 0.45
    out = rho.copy()
    out[design & np.broadcast_to(teeth, out.shape)] = 0.0
    return out


def _project_design(rho_f, design, rho_fixed, beta, eta):
    rho_p = tanh_project(rho_f, beta, eta)
    return np.where(design, rho_p, rho_fixed) if design.any() else rho_p


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
):
    """Piggott / Wang robust step: worst κ error of dilated, intermediate, eroded."""
    history = []
    field_ez = None
    t_th = t_dr = None
    for k in range(steps):
        beta = min(beta_final, 1.0 * (1.35**k))
        rho_f = helmholtz_filter(rho, r_px)
        candidates = []
        for name, eta in _ROBUST_ETAS:
            rho_p = _project_design(rho_f, design, rho, beta, eta)
            split, e_fwd, ok = _fdfd_split(
                rho_p, xs, ys, dx_um, n_core, n_bg, wavelength_nm, w_um, gap_um
            )
            if not ok or split is None:
                continue
            t_drop = float(split[1])
            candidates.append(
                {
                    "name": name,
                    "eta": eta,
                    "rho_p": rho_p,
                    "split": split,
                    "e_fwd": e_fwd,
                    "t_drop": t_drop,
                    "err": (t_drop - target) ** 2,
                }
            )
        if not candidates:
            break
        worst = max(candidates, key=lambda c: c["err"])
        by_name = {c["name"]: c for c in candidates}
        mid = by_name.get("intermediate", worst)
        t_th, t_dr = mid["split"]
        gray = _grayscale(mid["rho_p"])
        obj = worst["err"] + 0.08 * gray
        rec = {
            "step": k + 1,
            "objective": float(obj),
            "t_drop": float(worst["t_drop"]),
            "worst": worst["name"],
            "beta": float(beta),
        }
        for name, _eta in _ROBUST_ETAS:
            if name in by_name:
                rec[f"t_drop_{name}"] = by_name[name]["t_drop"]
        history.append(rec)
        dL_dE = _adjoint_source(worst["e_fwd"], xs, w_um, gap_um, worst["t_drop"], target)
        e_adj, aok = _fdfd_solve(
            worst["rho_p"], xs, ys, dx_um, n_core, n_bg, wavelength_nm, -dL_dE
        )
        if mid["e_fwd"] is not None:
            field_ez = _ez_field(xs, ys, mid["e_fwd"])
        if not aok or not np.isfinite(e_adj).all():
            continue
        k0 = 2 * math.pi / (wavelength_nm * 1e-3)
        deps = n_core**2 - n_bg**2
        sens = np.real(worst["e_fwd"] * e_adj) * (k0**2) * deps
        sens *= project_grad(rho_f, beta, worst["eta"])
        sens = helmholtz_filter(sens, r_px)
        sens *= design
        scale = float(np.max(np.abs(sens))) or 1.0
        updated = np.clip(rho - 0.08 * sens / scale, 0.0, 1.0)
        rho = np.where(design, updated, rho)
    return rho, history, t_th, t_dr, field_ez


def _robust_splits(
    variants, xs, ys, dx_um, n_core, n_bg, wavelength_nm, w_um, gap_um, target
):
    out = {}
    worst_name = None
    worst_err = -1.0
    for name, rho_p in variants.items():
        split, _, ok = _fdfd_split(
            rho_p, xs, ys, dx_um, n_core, n_bg, wavelength_nm, w_um, gap_um
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


def _eps_pml(rho, xs, ys, dx_um, n_core, n_bg, wavelength_nm, npml=8):
    """Permittivity on a μm grid. Small background loss + PML keeps FDFD stable."""
    eps = n_bg**2 + np.clip(rho, 0, 1) * (n_core**2 - n_bg**2)
    lam = wavelength_nm * 1e-3
    k0 = 2 * math.pi / lam
    sigma = np.zeros_like(eps)
    nx, ny = eps.shape
    for i in range(npml):
        r = ((npml - i) / npml) ** 2
        s = 1.4 * r
        sigma[i, :] += s
        sigma[-1 - i, :] += s
    for j in range(npml):
        r = ((npml - j) / npml) ** 2
        s = 1.4 * r
        sigma[:, j] += s
        sigma[:, -1 - j] += s
    eps = eps + 1j * (0.04 + sigma) * (n_core**2)
    return eps, k0, npml


def _fdfd_split(rho, xs, ys, dx_um, n_core, n_bg, wavelength_nm, w_um, gap_um):
    src = _src_profile(xs, w_um)
    e, ok = _fdfd_solve(rho, xs, ys, dx_um, n_core, n_bg, wavelength_nm, src)
    if not ok:
        return None, e, False
    jmon = max(8, ys.size - 10)
    th = np.abs(xs + w_um / 2) <= w_um * 0.85
    dr = np.abs(xs - (gap_um + w_um / 2)) <= w_um * 0.85
    pth = float(np.sum(np.abs(e[th, jmon]) ** 2))
    pdr = float(np.sum(np.abs(e[dr, jmon]) ** 2))
    tot = pth + pdr + 1e-30
    return (pth / tot, pdr / tot), e, True


def _src_profile(xs, w_um):
    src = np.zeros((xs.size, 1), dtype=complex)
    x0 = -w_um / 2
    src[:, 0] = np.exp(-((xs - x0) / (0.45 * w_um)) ** 2)
    return src


def _fdfd_solve(rho, xs, ys, dx_um, n_core, n_bg, wavelength_nm, src_x):
    eps, k0, npml = _eps_pml(rho, xs, ys, dx_um, n_core, n_bg, wavelength_nm)
    nx, ny = eps.shape
    dx = float(dx_um)
    jsrc = npml + 2
    b = np.zeros((nx, ny), dtype=complex)
    scale = k0**2
    if src_x.ndim == 1:
        b[:, jsrc] = src_x * scale
    elif src_x.shape[1] == 1:
        b[:, jsrc] = src_x[:, 0] * scale
    else:
        b += src_x * scale

    def op(u):
        p = np.pad(u, 1, mode="constant")
        lap = p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] - 4 * p[1:-1, 1:-1]
        return lap / dx**2 + (k0**2) * eps * u

    e, ok = _bicgstab(op, -b, tol=3e-3, maxiter=220)
    usable = bool(np.isfinite(e).all() and np.abs(e).max() > 1e-8)
    return e, ok or usable


def _adjoint_source(e_fwd, xs, w_um, gap_um, t_dr, target):
    jmon = max(8, e_fwd.shape[1] - 10)
    dr = np.abs(xs - (gap_um + w_um / 2)) <= w_um * 0.85
    src = np.zeros_like(e_fwd)
    src[dr, jmon] = (t_dr - target) * e_fwd[dr, jmon]
    return src


def _bicgstab(op, b, tol=2e-3, maxiter=160):
    x = np.zeros_like(b)
    r = b - op(x)
    r0 = r.copy()
    rho = alpha = omega = 1.0 + 0j
    v = np.zeros_like(b)
    p = np.zeros_like(b)
    bnorm = np.linalg.norm(b) + 1e-30
    for _ in range(maxiter):
        rho_n = np.vdot(r0, r)
        if abs(rho_n) < 1e-20:
            r0 = r.copy()
            rho_n = np.vdot(r0, r)
        beta = (rho_n / (rho + 1e-30)) * (alpha / (omega + 1e-30))
        p = r + beta * (p - omega * v)
        v = op(p)
        den = np.vdot(r0, v)
        if abs(den) < 1e-20:
            return x, False
        alpha = rho_n / den
        h = x + alpha * p
        s = r - alpha * v
        if np.linalg.norm(s) / bnorm < tol:
            return h, True
        t = op(s)
        t2 = np.vdot(t, t)
        if abs(t2) < 1e-20:
            return h, False
        omega = np.vdot(t, s) / t2
        x = h + omega * s
        r = s - omega * t
        if np.linalg.norm(r) / bnorm < tol:
            return x, True
        rho = rho_n
    return x, bool(np.isfinite(x).all() and np.linalg.norm(r) / bnorm < 0.12)


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


def _density_field(xs, ys, rho, polys, guides, quantity):
    """Propagation along the plot x-axis so the two rails read as a coupler, not a sliver."""
    _ = guides
    return {
        "x_um": ys.tolist(),
        "z_um": xs.tolist(),
        "intensity": np.clip(rho, 0, 1).tolist(),
        "quantity": quantity,
        "colormap": "density",
        "layout": "strip",
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
