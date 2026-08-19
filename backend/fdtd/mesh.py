"""Yee-grid mesher with volume-averaged (conformal) permittivity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fdtd.geometry import Scene
from fdtd.materials import MATERIALS, material_for_clad


@dataclass
class YeeGrid:
    x: np.ndarray
    y: np.ndarray
    dx: float
    dy: float
    eps_r: np.ndarray
    npml: int
    n_max: float
    conformal: bool

    @property
    def nx(self) -> int:
        return int(self.eps_r.shape[0])

    @property
    def ny(self) -> int:
        return int(self.eps_r.shape[1])

    def to_dict(self) -> dict:
        return {
            "nx": self.nx,
            "ny": self.ny,
            "dx_nm": self.dx * 1e9,
            "dy_nm": self.dy * 1e9,
            "cells": self.nx * self.ny,
            "npml": self.npml,
            "n_max": self.n_max,
            "conformal": self.conformal,
            "subpixel": "volume-average eps",
        }


def rasterize(
    scene: Scene,
    dx_um: float,
    npml: int = 10,
    n_clad: float = 1.33,
    conformal: bool = True,
) -> YeeGrid:
    x0, x1, y0, y1 = scene.bounds()
    dx = dx_um
    xs = np.arange(x0, x1 + dx, dx)
    ys = np.arange(y0, y1 + dx, dx)
    clad = material_for_clad(n_clad)
    bg = clad.eps_r
    eps = np.full((xs.size, ys.size), bg)
    mats = {**MATERIALS, "clad": clad, "box": MATERIALS["sio2"]}
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            if conformal:
                acc = 0.0
                for sx in (-0.25, 0.25):
                    for sy in (-0.25, 0.25):
                        name = scene.material_at(float(x + sx * dx), float(y + sy * dx))
                        mat = mats.get(name, clad)
                        acc += mat.eps_r
                eps[i, j] = acc / 4.0
            else:
                name = scene.material_at(float(x), float(y))
                eps[i, j] = mats.get(name, clad).eps_r
    n_max = float(np.sqrt(eps.max()))
    return YeeGrid(
        x=xs,
        y=ys,
        dx=dx * 1e-6,
        dy=dx * 1e-6,
        eps_r=eps,
        npml=npml,
        n_max=n_max,
        conformal=conformal,
    )


def cross_section_eps(
    width_nm: float,
    height_nm: float,
    n_core: float,
    n_box: float,
    n_clad: float,
    dx_nm: float = 50.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Conformal 2D mesh of a strip on BOX for the eigenmode solver."""
    w = width_nm
    h = height_nm
    x = np.arange(-1.5 * w, 1.5 * w + dx_nm, dx_nm)
    y = np.arange(-700.0, h + 900.0 + dx_nm, dx_nm)
    X, Y = np.meshgrid(x, y, indexing="ij")
    hit = np.zeros_like(X, dtype=float)
    for sx in (-0.25, 0.25):
        for sy in (-0.25, 0.25):
            in_core = (np.abs(X + sx * dx_nm) <= w / 2) & (Y + sy * dx_nm >= 0) & (
                Y + sy * dx_nm <= h
            )
            in_clad = (Y + sy * dx_nm) >= 0
            sample_bg = np.where(in_clad, n_clad**2, n_box**2)
            hit += np.where(in_core, n_core**2, sample_bg)
    return x, y, hit / 4.0
