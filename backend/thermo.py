"""Thermo-optic corners, SOI heater layout, and a 2D BOX-sink heat solve.
dn/dT from Cocorullo / Komma (Si ~1550 nm) and typical SiO2 / water values.
Steady heat is (-∇·(κ∇T) + g T = q): in-plane sheet conduction plus vertical
sink through the BOX to the substrate. The same operator supplies the adjoint
heat equation for exact ∂J/∂κ in the thermo-optic topology loop.
"""

from __future__ import annotations

import math
import numpy as np

# /K at ~1550 nm.
DN_DT_SI = 1.86e-4
DN_DT_SIO2 = 1.0e-5
DN_DT_WATER = -1.0e-4
DN_DT_AIR = 0.0

# How much extra core–clad contrast shrinks evanescent κ₀.
_KAPPA_CONTRAST = 6.0

CORNERS_K = (-20.0, -10.0, 0.0, 10.0, 20.0, 40.0)
heater_scales = (0.0, 0.5, 1.0)

# SOI thermal compact model (SI units).
K_SI = 130.0
K_SIO2 = 1.4
T_BOX_M = 2.0e-6
T_SI_M = 0.220e-6
G_SINK = K_SIO2 / T_BOX_M


def dn_clad_dt(n_clad: float) -> float:
    if n_clad < 1.15:
        return DN_DT_AIR
    if n_clad < 1.40:
        return DN_DT_WATER
    return DN_DT_SIO2


def dn_eff_dt(gamma_core: float, gamma_clad: float, n_clad: float) -> float:
    """First-order overlap: Γ_si dn_si/dT + Γ_clad dn_clad/dT."""
    return float(gamma_core) * DN_DT_SI + float(gamma_clad) * dn_clad_dt(n_clad)


def n_eff_at(n_eff: float, dndt: float, delta_t_k: float) -> float:
    return float(n_eff) + float(dndt) * float(delta_t_k)


def kappa0_at(kappa0: float, dndt: float, n_clad: float, delta_t_k: float) -> float:
    """Si heats faster than clad → tighter mode → weaker coupling."""
    d_contrast = (dndt - dn_clad_dt(n_clad)) * float(delta_t_k)
    return max(1e-6, float(kappa0) * math.exp(-_KAPPA_CONTRAST * d_contrast))


def kappa_from_kappa0(kappa0: float, length_um: float) -> float:
    return math.sin(kappa0 * length_um) ** 2


def resonance_shift_nm(resonance_nm: float, n_g: float, dndt: float, delta_t_k: float) -> float:
    """dλ/dT = (λ / n_g) dn_eff/dT."""
    if n_g <= 0:
        return 0.0
    return float(resonance_nm) / float(n_g) * float(dndt) * float(delta_t_k)


def sheet_k(rho: np.ndarray) -> np.ndarray:
    """In-plane sheet conductivity (W/K): BOX plus Si where ρ = 1."""
    return K_SIO2 * T_BOX_M + np.clip(rho, 0.0, 1.0) * K_SI * T_SI_M


def dsheet_k_drho(rho: np.ndarray) -> np.ndarray:
    """∂κ/∂ρ on the feasible interval. ``sheet_k`` is linear for ρ in [0, 1]."""
    density = np.asarray(rho, dtype=float)
    return np.where((density >= 0.0) & (density <= 1.0), K_SI * T_SI_M, 0.0)


def neighbor_sum(field: np.ndarray) -> np.ndarray:
    """Four edge-padded neighbors. A missing neighbor copies the boundary cell."""
    pad = np.pad(np.asarray(field), 1, mode="edge")
    return pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:]


def _dx_m(xs: np.ndarray) -> float:
    return float(abs(xs[1] - xs[0])) * 1e-6


def _kappa(k_sheet: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(k_sheet, dtype=float), 1e-12)


def apply_heat(field: np.ndarray, k_sheet: np.ndarray, dx_m: float, g: float = G_SINK) -> np.ndarray:
    """BOX-sink operator L(κ) applied to a temperature (or test) field.

    (L T)_i = (g + 4 κ_i / Δx²) T_i − (κ_i / Δx²) Σ_neighbors T.
    κ multiplies the whole stencil at cell i, matching the Jacobi fixed point.
    """
    c = 1.0 / (float(dx_m) ** 2)
    kappa = _kappa(k_sheet)
    temp = np.asarray(field, dtype=float)
    return (float(g) + 4.0 * c * kappa) * temp - c * kappa * neighbor_sum(temp)


def apply_heat_transpose(field: np.ndarray, k_sheet: np.ndarray, dx_m: float, g: float = G_SINK) -> np.ndarray:
    """L(κ)ᵀ. Off-diagonal weights sit on the source cell's conductivity."""
    c = 1.0 / (float(dx_m) ** 2)
    kappa = _kappa(k_sheet)
    adj = np.asarray(field, dtype=float)
    n_self = np.zeros(adj.shape, dtype=float)
    n_self[0, :] += 1.0
    n_self[-1, :] += 1.0
    n_self[:, 0] += 1.0
    n_self[:, -1] += 1.0
    out = (float(g) + c * kappa * (4.0 - n_self)) * adj
    out[:-1, :] += -c * kappa[1:, :] * adj[1:, :]
    out[1:, :] += -c * kappa[:-1, :] * adj[:-1, :]
    out[:, :-1] += -c * kappa[:, 1:] * adj[:, 1:]
    out[:, 1:] += -c * kappa[:, :-1] * adj[:, :-1]
    return out


def bicgstab_field(apply, rhs: np.ndarray, tol: float = 1e-8, maxiter: int = 400):
    """Matrix-free BiCGSTAB. ``apply`` maps an array of ``rhs.shape`` to one of the same shape."""
    b = np.asarray(rhs)
    x = np.zeros_like(b)
    r = b - apply(x)
    rhat = np.array(r, copy=True)
    rho = alpha = omega = 1.0 + 0.0j if np.iscomplexobj(b) else 1.0
    v = np.zeros_like(b)
    p = np.zeros_like(b)
    bnorm = np.linalg.norm(b) + 1e-30
    best = x
    best_rel = np.linalg.norm(r) / bnorm
    if best_rel <= tol:
        return x, True
    for _ in range(int(maxiter)):
        rho_n = np.vdot(rhat, r)
        if abs(rho_n) < 1e-30:
            break
        beta = (rho_n / (rho + 1e-30)) * (alpha / (omega + 1e-30))
        p = r + beta * (p - omega * v)
        v = apply(p)
        den = np.vdot(rhat, v)
        if abs(den) < 1e-30:
            break
        alpha = rho_n / den
        s = r - alpha * v
        rel_s = np.linalg.norm(s) / bnorm
        if rel_s < best_rel and np.isfinite(rel_s):
            best, best_rel = x + alpha * p, rel_s
        if rel_s <= tol:
            return x + alpha * p, True
        t = apply(s)
        tt = np.vdot(t, t)
        if abs(tt) < 1e-30:
            break
        omega = np.vdot(t, s) / tt
        if abs(omega) < 1e-30:
            break
        x = x + alpha * p + omega * s
        r = s - omega * t
        rel = np.linalg.norm(r) / bnorm
        if rel < best_rel and np.isfinite(x).all():
            best, best_rel = x, rel
        if rel <= tol:
            return x, True
        rho = rho_n
    return best, bool(np.isfinite(best).all() and best_rel <= tol)


def _heat_diagonal(kappa: np.ndarray, dx_m: float, g: float) -> np.ndarray:
    """Diagonal of L(κ), including boundary cells that neighbor themselves."""
    c = 1.0 / (float(dx_m) ** 2)
    n_self = np.zeros(kappa.shape, dtype=float)
    n_self[0, :] += 1.0
    n_self[-1, :] += 1.0
    n_self[:, 0] += 1.0
    n_self[:, -1] += 1.0
    return float(g) + c * kappa * (4.0 - n_self)


def _preconditioned_solve(apply, rhs, diag, tol, maxiter):
    """Left Jacobi-preconditioned BiCGSTAB. The solution is still apply(x) = rhs."""
    inv = 1.0 / diag
    return bicgstab_field(lambda field: apply(field) * inv, rhs * inv, tol=tol, maxiter=maxiter)


def solve_heat(
    xs: np.ndarray,
    ys: np.ndarray,
    q: np.ndarray,
    k_sheet: np.ndarray,
    g: float = G_SINK,
    niter: int = 90,
) -> np.ndarray:
    """(-∇·(κ∇) + g) T = q. κ is sheet conductivity; g sinks to the substrate."""
    if xs.size < 2 or ys.size < 2:
        return np.zeros_like(q, dtype=float)
    dx = _dx_m(xs)
    kappa = _kappa(k_sheet)
    rhs = np.asarray(q, dtype=float)
    diag = _heat_diagonal(kappa, dx, g)
    temp, ok = _preconditioned_solve(
        lambda field: apply_heat(field, kappa, dx, g),
        rhs,
        diag,
        tol=1e-7,
        maxiter=max(int(niter) * 8, 800),
    )
    if not ok or not np.isfinite(temp).all():
        temp, _ok = _preconditioned_solve(
            lambda field: apply_heat(field, kappa, dx, g),
            rhs,
            diag,
            tol=1e-6,
            maxiter=1600,
        )
    return np.maximum(np.nan_to_num(temp, nan=0.0, posinf=0.0, neginf=0.0), 0.0)


def adjoint_heat(
    xs: np.ndarray,
    k_sheet: np.ndarray,
    dJ_dT: np.ndarray,
    g: float = G_SINK,
    tol: float = 1e-7,
    maxiter: int = 800,
) -> np.ndarray:
    """Adjoint temperature λ solving L(κ)ᵀ λ = ∂J/∂T.

    L is the operator inside ``solve_heat``. κ is sheet conductivity.
    """
    rhs = np.asarray(dJ_dT, dtype=float)
    if xs.size < 2 or rhs.size == 0:
        return np.zeros_like(rhs)
    dx = _dx_m(xs)
    kappa = _kappa(k_sheet)
    diag = _heat_diagonal(kappa, dx, g)
    lam, ok = _preconditioned_solve(
        lambda field: apply_heat_transpose(field, kappa, dx, g),
        rhs,
        diag,
        tol=tol,
        maxiter=maxiter,
    )
    if not ok or not np.isfinite(lam).all():
        lam, _ok = _preconditioned_solve(
            lambda field: apply_heat_transpose(field, kappa, dx, g),
            rhs,
            diag,
            tol=max(tol, 1e-6),
            maxiter=max(maxiter * 2, 800),
        )
    return np.nan_to_num(lam, nan=0.0, posinf=0.0, neginf=0.0)


def heat_kappa_gradient(lam: np.ndarray, temperature: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """∂J/∂κ from λᵀ L(κ) T = λᵀ q, with q independent of κ.

    ∂R_i/∂κ_i = (4 T_i − Σ T_neighbors) / Δx² and dJ = −λ · (∂R/∂κ) dκ.
    """
    c = 1.0 / (_dx_m(xs) ** 2)
    temp = np.asarray(temperature, dtype=float)
    return -np.asarray(lam, dtype=float) * c * (4.0 * temp - neighbor_sum(temp))


def _axis_weights(axis: np.ndarray, points: np.ndarray):
    axis = np.asarray(axis, dtype=float)
    points = np.asarray(points, dtype=float)
    idx = np.clip(np.searchsorted(axis, points, side="right") - 1, 0, max(axis.size - 2, 0))
    span = axis[idx + 1] - axis[idx]
    t = np.clip((points - axis[idx]) / np.where(np.abs(span) < 1e-30, 1.0, span), 0.0, 1.0)
    return idx, t


def bilinear_sample(xs: np.ndarray, ys: np.ndarray, field: np.ndarray, xq, yq) -> np.ndarray:
    """Sample ``field`` on the increasing axes ``xs``, ``ys`` at query coordinates."""
    ix, tx = _axis_weights(xs, xq)
    iy, ty = _axis_weights(ys, yq)
    f = np.asarray(field, dtype=float)
    return (
        (1.0 - tx) * (1.0 - ty) * f[ix, iy]
        + tx * (1.0 - ty) * f[ix + 1, iy]
        + (1.0 - tx) * ty * f[ix, iy + 1]
        + tx * ty * f[ix + 1, iy + 1]
    )


def bilinear_scatter(xs: np.ndarray, ys: np.ndarray, xq, yq, values, shape) -> np.ndarray:
    """Adjoint of ``bilinear_sample``: scatter query values back onto the source grid."""
    ix, tx = _axis_weights(xs, xq)
    iy, ty = _axis_weights(ys, yq)
    out = np.zeros(shape, dtype=float)
    v = np.asarray(values, dtype=float)
    np.add.at(out, (ix, iy), (1.0 - tx) * (1.0 - ty) * v)
    np.add.at(out, (ix + 1, iy), tx * (1.0 - ty) * v)
    np.add.at(out, (ix, iy + 1), (1.0 - tx) * ty * v)
    np.add.at(out, (ix + 1, iy + 1), tx * ty * v)
    return out


def paste_window_density(rho_fixed, xs_c, ys_c, rho_window, xs_w, ys_w):
    """Write the coupler-window density onto circuit cells whose centers lie in the window."""
    rho = np.array(rho_fixed, dtype=float, copy=True)
    xx, yy = np.meshgrid(xs_c, ys_c, indexing="ij")
    inside = (xx >= xs_w[0]) & (xx <= xs_w[-1]) & (yy >= ys_w[0]) & (yy <= ys_w[-1])
    if np.any(inside):
        rho[inside] = bilinear_sample(xs_w, ys_w, rho_window, xx[inside], yy[inside])
    return rho, inside


def sample_window_temperature(xs_c, ys_c, temp_c, xs_w, ys_w) -> np.ndarray:
    xx, yy = np.meshgrid(xs_w, ys_w, indexing="ij")
    return bilinear_sample(xs_c, ys_c, temp_c, xx, yy)


def coupled_window_heat(xs_c, ys_c, rho_fixed, q, rho_window, xs_w, ys_w):
    """Chip-scale heat with the coupler window's silicon pasted in. Returns window ΔT too."""
    rho_c, inside = paste_window_density(rho_fixed, xs_c, ys_c, rho_window, xs_w, ys_w)
    temp_c = solve_heat(xs_c, ys_c, q, sheet_k(rho_c))
    temp_w = sample_window_temperature(xs_c, ys_c, temp_c, xs_w, ys_w)
    return rho_c, inside, temp_c, temp_w


def window_heat_gradient(
    xs_c,
    ys_c,
    rho_c,
    inside,
    temp_c,
    xs_w,
    ys_w,
    dJ_dT_window,
    ring_mask=None,
    dJ_dT_mean: float = 0.0,
) -> np.ndarray:
    """∂J/∂ρ on the coupler window after chip-scale heat and window sampling.

    ``dJ_dT_window`` is the optical derivative on the window grid.
    ``dJ_dT_mean`` is ∂J/∂(mean T on ``ring_mask``), from the compact resonance shift.
    """
    xx, yy = np.meshgrid(xs_w, ys_w, indexing="ij")
    dJ_dT = bilinear_scatter(xs_c, ys_c, xx, yy, dJ_dT_window, np.shape(temp_c))
    if ring_mask is not None and dJ_dT_mean != 0.0 and np.any(ring_mask):
        dJ_dT[ring_mask] = dJ_dT[ring_mask] + float(dJ_dT_mean) / float(np.count_nonzero(ring_mask))
    lam = adjoint_heat(xs_c, sheet_k(rho_c), dJ_dT)
    dJ_drho = np.where(inside, heat_kappa_gradient(lam, temp_c, xs_c) * dsheet_k_drho(rho_c), 0.0)
    if not np.any(inside):
        return np.zeros((xs_w.size, ys_w.size), dtype=float)
    cxx, cyy = np.meshgrid(xs_c, ys_c, indexing="ij")
    return bilinear_scatter(
        xs_w, ys_w, cxx[inside], cyy[inside], dJ_drho[inside], (xs_w.size, ys_w.size)
    )


def paint_heaters(xs: np.ndarray, ys: np.ndarray, heaters: list[dict], scale: float = 1.0) -> np.ndarray:
    q = np.zeros((xs.size, ys.size), dtype=float)
    if not heaters:
        return q
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    for h in heaters:
        x0 = float(h["x0"])
        y0 = float(h["y0"])
        w = float(h["width"])
        ht = float(h["height"])
        area = max(w * ht * 1e-12, 1e-20)
        on = (xx >= x0) & (xx <= x0 + w) & (yy >= y0) & (yy <= y0 + ht)
        q[on] += float(scale) * float(h["power_w"]) / area
    return q


def heater_layout(devices: list[dict], width_nm: float) -> list[dict]:
    """Metal heaters in the same μm frame as the FDTD / TO coupler scene."""
    heaters = [d for d in devices if d.get("type") == "heater"]
    if not heaters:
        return []
    rings = [d for d in devices if d.get("type") == "ring"]
    couplers = [d for d in devices if d.get("type") == "coupler"]
    w = width_nm * 1e-3
    length = float(couplers[0]["length_um"]) if couplers else 12.0
    gap = float(couplers[0]["gap_nm"]) * 1e-3 if couplers else None
    out = []
    for i, h in enumerate(heaters):
        hw = max(0.4, float(h.get("width_um", 2.0)))
        hl = max(1.0, float(h.get("length_um", 40.0)))
        pw = max(0.0, float(h.get("power_mw", 10.0))) * 1e-3
        if rings:
            r = float(rings[0].get("radius_um", 10.0))
            cx, cy = -w / 2.0, length * 0.5
            hx = float(h.get("x", rings[0].get("x", 0.0)))
            rx = float(rings[0].get("x", 0.0))
            side = -1.0 if hx < rx else 1.0
            hl = min(hl, max(2.0, 1.2 * r))
            x0 = cx + side * r - hw / 2.0
            y0 = cy - hl / 2.0
        else:
            hl = min(hl, max(1.2, length - 0.3))
            x0 = (-w - 0.25 - hw) if i % 2 == 0 else ((gap + w + 0.25) if gap is not None else w + 0.25)
            y0 = max(0.1, (length - hl) * 0.5)
        out.append(
            {
                "x0": float(x0),
                "y0": float(y0),
                "width": float(hw),
                "height": float(hl),
                "power_w": float(pw),
                "power_mw": float(pw * 1e3),
            }
        )
    return out


def heater_hits_grid(heater: dict, xs: np.ndarray, ys: np.ndarray) -> bool:
    return (
        float(heater["x0"]) + float(heater["width"]) >= float(xs[0])
        and float(heater["x0"]) <= float(xs[-1])
        and float(heater["y0"]) + float(heater["height"]) >= float(ys[0])
        and float(heater["y0"]) <= float(ys[-1])
    )


def default_coupler_heater(w_um: float, length_um: float, pad_um: float, power_mw: float = 10.0) -> dict:
    """Side heater on the through rail, inside the TO window pad."""
    hw = min(1.2, max(0.45, pad_um - 0.12))
    hl = min(max(1.2, length_um - 0.4), 2.6)
    return {
        "x0": float(-w_um - pad_um + 0.06),
        "y0": float(max(0.12, (length_um - hl) * 0.5)),
        "width": float(hw),
        "height": float(hl),
        "power_w": float(power_mw) * 1e-3,
        "power_mw": float(power_mw),
    }


def mean_on_mask(temp: np.ndarray, mask: np.ndarray) -> float:
    if mask is None or not np.any(mask):
        return 0.0
    return float(temp[mask].mean())


def circuit_heat_grid(width_nm: float, devices: list[dict]):
    w = width_nm * 1e-3
    rings = [d for d in devices if d.get("type") == "ring"]
    couplers = [d for d in devices if d.get("type") == "coupler"]
    r = float(rings[0]["radius_um"]) if rings else 0.0
    length = float(couplers[0]["length_um"]) if couplers else 12.0
    gap = float(couplers[0]["gap_nm"]) * 1e-3 if couplers else 0.0
    x0 = -max(r + 2.2, w + 3.0)
    x1 = max(gap + w + 2.0, r + 2.2, w + 2.0)
    y0 = -2.0
    y1 = max(length, 2.0 * r if r else length) + 2.0
    dx = 0.16
    while ((x1 - x0) / dx) * ((y1 - y0) / dx) > 9000:
        dx *= 1.08
    xs = np.arange(x0, x1 + dx * 0.5, dx)
    ys = np.arange(y0, y1 + dx * 0.5, dx)
    return xs, ys


def circuit_si_density(xs: np.ndarray, ys: np.ndarray, width_nm: float, devices: list[dict]) -> np.ndarray:
    w = width_nm * 1e-3
    rings = [d for d in devices if d.get("type") == "ring"]
    couplers = [d for d in devices if d.get("type") == "coupler"]
    length = float(couplers[0]["length_um"]) if couplers else 12.0
    gap = float(couplers[0]["gap_nm"]) * 1e-3 if couplers else None
    add_drop = any(d.get("config") == "add-drop" for d in rings)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    rho = ((xx >= -w) & (xx <= 0.0) & (yy >= 0.0) & (yy <= length)).astype(float)
    if gap is not None and (add_drop or couplers):
        rho = np.maximum(
            rho,
            ((xx >= gap) & (xx <= gap + w) & (yy >= 0.0) & (yy <= length)).astype(float),
        )
    if rings:
        r = float(rings[0]["radius_um"])
        cx, cy = -w / 2.0, length * 0.5
        rad = np.hypot(xx - cx, yy - cy)
        rho = np.maximum(rho, ((rad <= r + w / 2.0) & (rad >= max(0.15, r - w / 2.0))).astype(float))
    return rho


def circuit_masks(xs: np.ndarray, ys: np.ndarray, width_nm: float, devices: list[dict]):
    w = width_nm * 1e-3
    rings = [d for d in devices if d.get("type") == "ring"]
    couplers = [d for d in devices if d.get("type") == "coupler"]
    length = float(couplers[0]["length_um"]) if couplers else 12.0
    gap = float(couplers[0]["gap_nm"]) * 1e-3 if couplers else None
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    through = (xx >= -w) & (xx <= 0.0) & (yy >= 0.0) & (yy <= length)
    drop = (
        (xx >= gap) & (xx <= gap + w) & (yy >= 0.0) & (yy <= length)
        if gap is not None
        else np.zeros_like(xx, dtype=bool)
    )
    ring = np.zeros_like(xx, dtype=bool)
    if rings:
        r = float(rings[0]["radius_um"])
        cx, cy = -w / 2.0, length * 0.5
        rad = np.hypot(xx - cx, yy - cy)
        ring = (rad <= r + w / 2.0) & (rad >= max(0.15, r - w / 2.0))
    return ring, through | drop


def to_plot_guides(guides: list[dict]) -> list[dict]:
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


def heater_polygons(heaters: list[dict]) -> list[list[dict]]:
    polys = []
    for h in heaters:
        x0, y0 = float(h["x0"]), float(h["y0"])
        w, ht = float(h["width"]), float(h["height"])
        raster = ((x0, y0), (x0 + w, y0), (x0 + w, y0 + ht), (x0, y0 + ht), (x0, y0))
        polys.append([{"x": y, "y": x} for x, y in raster])
    return polys


def heat_field(
    xs: np.ndarray,
    ys: np.ndarray,
    temp: np.ndarray,
    guides: list[dict] | None = None,
    polygons: list[list[dict]] | None = None,
) -> dict:
    peak = float(np.max(temp)) if temp.size else 0.0
    return {
        "x_um": ys.tolist(),
        "z_um": xs.tolist(),
        "intensity": np.clip(temp / (peak + 1e-12), 0.0, 1.0).tolist(),
        "quantity": "ΔT",
        "colormap": "jet",
        "layout": "strip",
        "guides": to_plot_guides(guides or []),
        "polygons": polygons or [],
        "dt_max_k": peak,
    }


def solve_circuit_heat(devices: list[dict], width_nm: float) -> dict | None:
    heaters = heater_layout(devices, width_nm)
    if not heaters:
        return None
    xs, ys = circuit_heat_grid(width_nm, devices)
    rho = circuit_si_density(xs, ys, width_nm, devices)
    q = paint_heaters(xs, ys, heaters)
    temp = solve_heat(xs, ys, q, sheet_k(rho))
    ring_m, cpl_m = circuit_masks(xs, ys, width_nm, devices)
    w = width_nm * 1e-3
    couplers = [d for d in devices if d.get("type") == "coupler"]
    length = float(couplers[0]["length_um"]) if couplers else 12.0
    gap = float(couplers[0]["gap_nm"]) * 1e-3 if couplers else None
    guides = [{"x0": float(-w), "y0": 0.0, "width": float(w), "height": float(length)}]
    if gap is not None:
        guides.append({"x0": float(gap), "y0": 0.0, "width": float(w), "height": float(length)})
    guides.extend(
        {
            "x0": float(h["x0"]),
            "y0": float(h["y0"]),
            "width": float(h["width"]),
            "height": float(h["height"]),
        }
        for h in heaters
    )
    return {
        "source": "2D BOX-sink heat PDE",
        "layout": heaters,
        "power_mw": float(sum(h["power_mw"] for h in heaters)),
        "dt_max_k": float(temp.max()),
        "dt_ring_k": mean_on_mask(temp, ring_m),
        "dt_coupler_k": mean_on_mask(temp, cpl_m),
        "field": heat_field(xs, ys, temp, guides, heater_polygons(heaters)),
        "note": (
            "Steady (−∇·(κ∇T) + gT = q) with vertical sink through 2 μm BOX. "
            "Ring and coupler can see different ΔT when the heater is offset."
        ),
    }
