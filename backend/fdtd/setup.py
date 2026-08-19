"""Build an FDTD scene from the sensor circuit and describe / run the six engines."""

from __future__ import annotations

from fdtd.boundaries import BoundarySpec
from fdtd.engine import run_yee
from fdtd.geometry import Cylinder, Rect, Scene
from fdtd.materials import MATERIALS, catalog, material_for_clad
from fdtd.mesh import rasterize
from fdtd.modes import solve_strip_mode
from fdtd.monitors import far_field_hint


def circuit_scene(
    width_nm: float,
    devices: list[dict],
    n_clad: float,
) -> Scene:
    """Top-view 2.5D geometry in μm. Waveguides along +y, coupler gap along x."""
    scene = Scene(background="clad")
    w = width_nm * 1e-3
    rings = [d for d in devices if d.get("type") == "ring"]
    couplers = [d for d in devices if d.get("type") == "coupler"]
    wgs = [d for d in devices if d.get("type") == "waveguide"]
    dets = [d for d in devices if d.get("type") == "detector"]

    length = 12.0
    gap = 0.20
    if couplers:
        length = float(couplers[0].get("length_um", 12.0))
        gap = float(couplers[0].get("gap_nm", 200.0)) * 1e-3
    elif wgs:
        length = min(40.0, float(wgs[0].get("length_um", 20.0)))

    # Through bus
    scene.add(Rect(-w, 0.0, w, length, "si", "through"))
    add_drop = any(d.get("config") == "add-drop" for d in rings)
    if add_drop:
        scene.add(Rect(gap, 0.0, w, length, "si", "drop"))

    if rings:
        r = float(rings[0].get("radius_um", 10.0))
        # Ring sits beside the through bus (annulus).
        cx = -w / 2
        cy = length * 0.5
        scene.add(Cylinder(cx, cy, r + w / 2, "si", "ring", inner_radius=max(0.2, r - w / 2)))

    if dets:
        scene.add(Rect(-w, length + 0.2, w, 1.2, "ge", "detector"))

    scene.background = "clad"
    _ = n_clad
    return scene


def describe_fdtd(
    width_nm: float,
    height_nm: float,
    wavelength_nm: float,
    n_clad: float,
    polarization: str,
    devices: list[dict],
    run_mode: bool = True,
) -> dict:
    scene = circuit_scene(width_nm, devices, n_clad)
    clad = material_for_clad(n_clad)
    dx_um = 0.05
    npml = 10
    bc = BoundarySpec(kind="cpml", npml=npml, symmetry_x=None, symmetry_y=None)
    bc_dict = bc.to_dict()
    bc_dict["absorbing"] = "graded conductivity PML"
    bc_dict["note"] = "Polynomial profile (same grading as CPML). Convolutional CPML is not in the update yet."
    couplers = [d for d in devices if d.get("type") == "coupler"]
    length = float(couplers[0]["length_um"]) if couplers else 12.0
    gap = float(couplers[0]["gap_nm"]) * 1e-3 if couplers else 0.20
    w = width_nm * 1e-3

    mode = None
    if run_mode:
        try:
            mode = solve_strip_mode(width_nm, height_nm, wavelength_nm, n_clad, polarization)
        except Exception as exc:
            mode = {"error": str(exc), "n_eff": None}

    n_core_2d = (mode or {}).get("n_eff") or MATERIALS["si"].n
    # 2.5D in-plane index uses the FDE n_eff as the strip, cladding as background.
    inplane = Scene(background="clad")
    inplane.add(Rect(-w, 0.0, w, length, "si", "through"))
    if couplers:
        inplane.add(Rect(gap, 0.0, w, length, "si", "drop"))

    x0, x1, y0, y1 = inplane.bounds()
    xspan = (x1 - x0) + 2 * npml * dx_um
    yspan = (y1 - y0) + 2 * npml * dx_um
    nx = int(xspan / dx_um)
    ny = int(yspan / dx_um)
    cfl = 0.49 / (2**0.5)

    return {
        "geometry": {
            "primitives": scene.to_list(),
            "units": "um",
            "gdsii": "polygons_from_layout() maps GDS layers onto Polygon primitives",
            "background": clad.name,
        },
        "mesh": {
            "kind": "Yee",
            "dx_nm": dx_um * 1e3,
            "nx": nx,
            "ny": ny,
            "cells": nx * ny,
            "conformal": True,
            "subpixel": "2×2 volume-average ε",
            "n_core_2d": n_core_2d,
        },
        "materials": {
            "catalog": catalog(),
            "active": [clad.name, "si"] + (["ge"] if any(d.get("type") == "detector" for d in devices) else []),
            "dispersion": "Lorentz/Debye poles stored; time-step uses n²",
            "complex_media": {
                "anisotropic": False,
                "nonlinear": False,
                "gain": False,
            },
        },
        "boundaries": bc_dict,
        "sources": {
            "mode_solver": "2d-fd-eigenmode on the strip cross-section",
            "injection": "gaussian line source (eigenmode on Run)",
            "dipole": False,
            "tfsf": False,
            "polarization": polarization,
            "wavelength_nm": wavelength_nm,
        },
        "engine": {
            "scheme": "Yee interleaved E(n+1) / H(n+1/2)",
            "courant": cfl,
            "parallel": "NumPy vectorized",
            "dimension": "2.5D in-plane coupler + 2D FDE cross-section",
        },
        "monitors": {
            "dft": True,
            "poynting": True,
            "mode_overlap": True,
            "far_field": far_field_hint(),
            "ports": ["through", "drop"] if couplers else ["through"],
        },
        "mode": mode,
        "run": None,
    }


def run_fdtd(
    width_nm: float,
    height_nm: float,
    wavelength_nm: float,
    n_clad: float,
    polarization: str,
    devices: list[dict],
) -> dict:
    report = describe_fdtd(
        width_nm, height_nm, wavelength_nm, n_clad, polarization, devices, run_mode=True
    )
    couplers = [d for d in devices if d.get("type") == "coupler"]
    wgs = [d for d in devices if d.get("type") == "waveguide"]
    if not couplers and not wgs:
        report["run"] = {"error": "Place a waveguide or coupler to run 2D FDTD."}
        return report

    length = float(couplers[0]["length_um"]) if couplers else min(16.0, float(wgs[0].get("length_um", 12)))
    gap = float(couplers[0]["gap_nm"]) * 1e-3 if couplers else None
    w = width_nm * 1e-3
    scene = Scene(background="clad")
    scene.add(Rect(-w, 0.0, w, length, "si", "through"))
    if gap is not None:
        scene.add(Rect(gap, 0.0, w, length, "si", "drop"))
    grid = rasterize(scene, dx_um=0.05, npml=10, n_clad=n_clad, conformal=True)
    # Paint 2.5D n_eff² into Si cells so the in-plane index matches the FDE.
    if report["mode"] and report["mode"].get("n_eff"):
        n2 = report["mode"]["n_eff"] ** 2
        core = grid.eps_r > 4.0
        grid.eps_r[core] = n2
        grid.n_max = float(max(grid.n_max, n2**0.5))

    src_y = grid.y[grid.npml + 4]
    mon_y = grid.y[-(grid.npml + 6)]
    through_x = -w / 2
    drop_x = gap + w / 2 if gap is not None else None
    report["run"] = run_yee(
        grid,
        wavelength_nm,
        src_y,
        mon_y,
        through_x,
        drop_x,
        w,
    )
    report["mesh"].update(grid.to_dict())
    return report
