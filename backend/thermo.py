"""Thermo-optic corners, SOI heater layout, and a 2D BOX-sink heat solve.
dn/dT from Cocorullo / Komma (Si ~1550 nm) and typical SiO2 / water values.
Steady heat is (-∇·(κ∇T) + g T = q): in-plane sheet conduction plus vertical
sink through the BOX to the substrate. Used for heater-driven ΔT and the
thermo-optic inverse-design loop.
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
        return np.zeros_like(q)
    dx = float(abs(xs[1] - xs[0])) * 1e-6
    T = np.zeros_like(q, dtype=float)
    rhs = np.asarray(q, dtype=float)
    coeff = np.maximum(np.asarray(k_sheet, dtype=float), 1e-12) / (dx * dx)
    den = float(g) + 4.0 * coeff
    for _ in range(niter):
        pad = np.pad(T, 1, mode="edge")
        nb = pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:]
        T = (rhs + coeff * nb) / den
    return np.maximum(T, 0.0)


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
