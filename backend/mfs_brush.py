"""Minimum-feature brushes for gradient-based topology optimization.

A circular brush of diameter equal to the foundry minimum linewidth is the
structuring element for morphological opening (solid too thin to contain the
brush) and closing (gap too narrow to contain it). A square of the same
size with its corners cut out is the notched-square brush: it sits on a
legal straight edge, and it does not sit in a sub-resolution notch or a
one-pixel jog where a waveguide sidewall changes width.

The min and max over each brush are the log-sum-exp (Kreisselmeier–Steinhauser)
smoothings, so the mismatch between the density and its opening or closing
has a gradient. That gradient is what the optimizer steps on.

Hammond, Oskooi, Johnson & Ralph, Opt. Express 29, 23916 (2021).
Zhou, Lazarov, Wang & Sigmund, Comput. Methods Appl. Mech. Eng. 293, 266 (2015).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Sharp enough that a brush either fits or it does not, still smooth for a step.
BRUSH_BETA = 24.0

_TERMS = ("circular_width", "circular_gap", "notched_width", "notched_gap")


@dataclass(frozen=True)
class LengthScaleBrushes:
    """Pixel brushes sized from the DUV minimum feature and the grid."""

    radius_px: int
    half_px: int
    notch_px: int
    circular: tuple[tuple[int, int], ...]
    notched: tuple[tuple[int, int], ...]
    mfs_nm: float
    dx_um: float

    @property
    def circular_radius_nm(self) -> float:
        return float(self.radius_px * self.dx_um * 1e3)

    @property
    def notched_side_nm(self) -> float:
        return float((2 * self.half_px + 1) * self.dx_um * 1e3)

    @property
    def notch_nm(self) -> float:
        return float(self.notch_px * self.dx_um * 1e3)

    @property
    def enforced_feature_nm(self) -> float:
        """Width a feature must reach before the circular brush fits inside it."""
        return float(2.0 * self.radius_px * self.dx_um * 1e3)


def circular_offsets(radius: int) -> tuple[tuple[int, int], ...]:
    """Disk of radius ``radius`` pixels, including the center."""
    r = int(radius)
    r2 = r * r
    return tuple(
        (i, j)
        for i in range(-r, r + 1)
        for j in range(-r, r + 1)
        if i * i + j * j <= r2
    )


def notched_square_offsets(half: int, notch: int) -> tuple[tuple[int, int], ...]:
    """Square of half-side ``half`` with a ``notch`` × ``notch`` bite at each corner.

    ``notch`` of 0 leaves the full square. The corner cut is what stops the
    brush fitting into a stair-step or a notch narrower than the design rule.
    """
    h = int(half)
    n = int(min(max(notch, 0), max(h - 1, 0)))
    offsets = []
    for i in range(-h, h + 1):
        for j in range(-h, h + 1):
            corner = n > 0 and abs(i) > h - n and abs(j) > h - n
            if not corner:
                offsets.append((i, j))
    return tuple(offsets)


def length_scale_brushes(mfs_nm: float, dx_um: float, guide_um: float) -> LengthScaleBrushes:
    """Size both brushes to the minimum feature, and keep them inside the guide.

    The circle diameter is the minimum linewidth. The notched square uses the
    same half-side; its corner cut is half of that, and at least one pixel
    once the square is wider than 3×3. At a one-pixel radius the circle is a
    plus and the square keeps its corners, because cutting them would leave
    that same plus.
    """
    dx_nm = float(dx_um) * 1e3
    guide_px = float(guide_um) * 1e3 / max(dx_nm, 1e-9)
    max_half = max(1, int((guide_px - 1.0) // 2))
    radius = int(np.clip(int(round(0.5 * float(mfs_nm) / max(dx_nm, 1e-9))), 1, max_half))
    notch = 0 if radius < 2 else max(1, radius // 2)
    return LengthScaleBrushes(
        radius_px=radius,
        half_px=radius,
        notch_px=notch,
        circular=circular_offsets(radius),
        notched=notched_square_offsets(radius, notch),
        mfs_nm=float(mfs_nm),
        dx_um=float(dx_um),
    )


def _pad_radius(offsets: tuple[tuple[int, int], ...]) -> int:
    return max(max(abs(i), abs(j)) for i, j in offsets)


def _gather(padded: np.ndarray, offsets, rad: int, shape: tuple[int, int]) -> np.ndarray:
    nx, ny = shape
    return np.stack(
        [padded[rad + di : rad + di + nx, rad + dj : rad + dj + ny] for di, dj in offsets],
        axis=0,
    )


def _fold_edge_pad(gpad: np.ndarray, rad: int) -> np.ndarray:
    """Scatter a gradient out of an edge-padded array back onto the original pixels."""
    nx = gpad.shape[0] - 2 * rad
    ny = gpad.shape[1] - 2 * rad
    g = gpad[rad : rad + nx, rad : rad + ny].copy()
    if rad == 0:
        return g
    left = gpad[:, :rad]
    right = gpad[:, rad + ny :]
    g[0, 0] += left[:rad, :].sum()
    g[-1, 0] += left[rad + nx :, :].sum()
    g[:, 0] += left[rad : rad + nx, :].sum(axis=1)
    g[0, -1] += right[:rad, :].sum()
    g[-1, -1] += right[rad + nx :, :].sum()
    g[:, -1] += right[rad : rad + nx, :].sum(axis=1)
    g[0, :] += gpad[:rad, rad : rad + ny].sum(axis=0)
    g[-1, :] += gpad[rad + nx :, rad : rad + ny].sum(axis=0)
    return g


def _scatter(contrib: np.ndarray, offsets, rad: int, shape: tuple[int, int]) -> np.ndarray:
    nx, ny = shape
    gpad = np.zeros((nx + 2 * rad, ny + 2 * rad), dtype=float)
    for k, (di, dj) in enumerate(offsets):
        gpad[rad + di : rad + di + nx, rad + dj : rad + dj + ny] += contrib[k]
    return _fold_edge_pad(gpad, rad)


def _softmax(stacked: np.ndarray, beta: float) -> tuple[np.ndarray, np.ndarray]:
    peak = np.max(stacked, axis=0, keepdims=True)
    weights = np.exp(beta * (stacked - peak))
    weights /= np.sum(weights, axis=0, keepdims=True)
    return weights, peak


def soft_dilate(rho: np.ndarray, offsets, beta: float) -> np.ndarray:
    """Smooth max over the brush. A constant field is unchanged.

    The ``1/K`` inside the log-mean is what keeps ``dilate(c) = c``. The
    gradient does not see it: ∂/∂ρ of this map is the softmax over the brush.
    """
    rad = _pad_radius(offsets)
    padded = np.pad(rho, rad, mode="edge")
    stacked = _gather(padded, offsets, rad, rho.shape)
    peak = np.max(stacked, axis=0)
    k = float(stacked.shape[0])
    z = np.sum(np.exp(beta * (stacked - peak)), axis=0)
    return peak + np.log(z / k) / beta


def dilate_vjp(rho: np.ndarray, offsets, beta: float, upstream: np.ndarray) -> np.ndarray:
    """∂J/∂ρ for J that depends on soft_dilate through ``upstream`` = ∂J/∂dilate."""
    rad = _pad_radius(offsets)
    padded = np.pad(rho, rad, mode="edge")
    stacked = _gather(padded, offsets, rad, rho.shape)
    weights, _peak = _softmax(stacked, beta)
    return _scatter(weights * np.asarray(upstream, dtype=float), offsets, rad, rho.shape)


def soft_erode(rho: np.ndarray, offsets, beta: float) -> np.ndarray:
    """Smooth min. erode(ρ) = 1 − dilate(1 − ρ), so a constant is unchanged."""
    return 1.0 - soft_dilate(1.0 - rho, offsets, beta)


def erode_vjp(rho: np.ndarray, offsets, beta: float, upstream: np.ndarray) -> np.ndarray:
    # erode = 1 - dilate(1 - ρ). The two minus signs cancel in the VJP.
    return dilate_vjp(1.0 - rho, offsets, beta, upstream)


def opening(rho: np.ndarray, offsets, beta: float) -> np.ndarray:
    """Erosion then dilation: deletes solid that cannot contain the brush."""
    return soft_dilate(soft_erode(rho, offsets, beta), offsets, beta)


def opening_vjp(rho: np.ndarray, offsets, beta: float, upstream: np.ndarray) -> np.ndarray:
    eroded = soft_erode(rho, offsets, beta)
    return erode_vjp(rho, offsets, beta, dilate_vjp(eroded, offsets, beta, upstream))


def closing(rho: np.ndarray, offsets, beta: float) -> np.ndarray:
    """Dilation then erosion: fills a gap that cannot contain the brush."""
    return soft_erode(soft_dilate(rho, offsets, beta), offsets, beta)


def closing_vjp(rho: np.ndarray, offsets, beta: float, upstream: np.ndarray) -> np.ndarray:
    dilated = soft_dilate(rho, offsets, beta)
    return dilate_vjp(rho, offsets, beta, erode_vjp(dilated, offsets, beta, upstream))


def _mismatch(diff: np.ndarray, mask: np.ndarray) -> tuple[float, np.ndarray]:
    n = float(np.count_nonzero(mask))
    if n < 1.0:
        n = float(diff.size)
        mask = np.ones(diff.shape, dtype=bool)
    masked = np.where(mask, diff, 0.0)
    value = float(np.sum(masked * masked) / n)
    grad = 2.0 * masked / n
    return value, grad


def feature_penalty(
    rho: np.ndarray,
    offsets,
    beta: float,
    mask: np.ndarray,
    kind: str,
) -> tuple[float, np.ndarray]:
    """Mean squared mismatch of an opening (width) or a closing (gap).

    ``mask`` selects the pixels that count — the free sidewall — while the
    brush still reads the whole density, including the fixed guide it sits on.
    """
    if kind == "width":
        morph = opening(rho, offsets, beta)
        value, d_diff = _mismatch(rho - morph, mask)
        return value, d_diff + opening_vjp(rho, offsets, beta, -d_diff)
    morph = closing(rho, offsets, beta)
    value, d_diff = _mismatch(morph - rho, mask)
    return value, -d_diff + closing_vjp(rho, offsets, beta, d_diff)


def mfs_penalty(
    rho: np.ndarray,
    brushes: LengthScaleBrushes,
    mask: np.ndarray,
    beta: float = BRUSH_BETA,
) -> tuple[dict[str, float], np.ndarray]:
    """Width, gap, and notched-square penalties, and ∂/∂ρ of their sum.

    Circular opening is the linewidth. Circular closing is the gap. The
    notched square applies the same pair to notches and structural transitions.
    """
    pairs = (
        ("circular_width", "width", brushes.circular),
        ("circular_gap", "gap", brushes.circular),
        ("notched_width", "width", brushes.notched),
        ("notched_gap", "gap", brushes.notched),
    )
    terms: dict[str, float] = {}
    grad = np.zeros(rho.shape, dtype=float)
    for name, kind, offsets in pairs:
        value, g = feature_penalty(rho, offsets, beta, mask, kind)
        terms[name] = value
        grad += g
    terms["total"] = float(sum(terms[name] for name in _TERMS))
    terms["width"] = terms["circular_width"]
    terms["gap"] = terms["circular_gap"]
    terms["notch"] = terms["notched_width"] + terms["notched_gap"]
    return terms, grad


def brush_report(
    rho: np.ndarray,
    brushes: LengthScaleBrushes,
    mask: np.ndarray,
    beta: float = BRUSH_BETA,
) -> dict:
    """Penalty on a finished density, for the topology response."""
    terms, _grad = mfs_penalty(rho, brushes, mask, beta)
    return {
        "method": "circular and notched-square brushes",
        "reference": "Hammond, Oskooi, Johnson & Ralph, Opt. Express 29, 23916 (2021)",
        "smoothing": "log-sum-exp opening and closing (Zhou, Lazarov, Wang & Sigmund 2015)",
        "circular_radius_nm": brushes.circular_radius_nm,
        "notched_side_nm": brushes.notched_side_nm,
        "notch_nm": brushes.notch_nm,
        "enforced_feature_nm": brushes.enforced_feature_nm,
        "circular_width": terms["circular_width"],
        "circular_gap": terms["circular_gap"],
        "notched_width": terms["notched_width"],
        "notched_gap": terms["notched_gap"],
        "width": terms["width"],
        "gap": terms["gap"],
        "notch": terms["notch"],
        "total": terms["total"],
    }
