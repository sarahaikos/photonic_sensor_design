from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from circuit import analyze_circuit
from coupler import analyze_coupler
from detector import analyze_detector
from fdtd.setup import run_fdtd
from ring import analyze_ring
from topology import run_robustness_sweep, run_topology
from waveguide import analyze_waveguide

app = FastAPI(title="Photonic Sensor Design")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RingInput(BaseModel):
    radius_um: float = Field(gt=0)
    wavelength_nm: float = Field(gt=0)
    n_eff: float = Field(gt=0)
    n_g: float = Field(gt=0)
    kappa: float = Field(ge=0, lt=1)
    loss_db_per_cm: float = Field(ge=0)
    dn_eff_dn: float = Field(ge=0, le=1)
    config: str = Field(pattern="^(all-pass|add-drop)$")
    width_nm: float = Field(default=450.0, gt=0)
    polarization: str = Field(default="TE", pattern="^(TE|TM)$")


class WaveguideInput(BaseModel):
    width_nm: float = Field(gt=0)
    height_nm: float = Field(gt=0)
    wavelength_nm: float = Field(gt=0)
    n_clad: float = Field(gt=0)
    polarization: str = Field(pattern="^(TE|TM)$")
    length_um: float = Field(default=100.0, gt=0)


class CouplerInput(BaseModel):
    gap_nm: float = Field(gt=0)
    length_um: float = Field(gt=0)
    width_nm: float = Field(gt=0)
    wavelength_nm: float = Field(gt=0)
    polarization: str = Field(pattern="^(TE|TM)$")
    height_nm: float = Field(default=220.0, gt=0)


class DetectorInput(BaseModel):
    wavelength_nm: float = Field(gt=0)
    optical_power_uw: float = Field(gt=0)
    responsivity_a_per_w: float = Field(gt=0)
    dark_current_na: float = Field(ge=0)
    bandwidth_mhz: float = Field(gt=0)
    load_ohm: float = Field(gt=0)


@app.post("/api/ring")
def ring(payload: RingInput):
    return analyze_ring(**payload.model_dump())


@app.post("/api/waveguide")
def waveguide(payload: WaveguideInput):
    result = analyze_waveguide(**payload.model_dump())
    if result is None:
        raise HTTPException(status_code=400, detail="No guided mode for this cross-section")
    return result


@app.post("/api/coupler")
def coupler(payload: CouplerInput):
    return analyze_coupler(**payload.model_dump())


@app.post("/api/detector")
def detector(payload: DetectorInput):
    return analyze_detector(**payload.model_dump())


class CircuitInput(BaseModel):
    width_nm: float = Field(gt=0)
    height_nm: float = Field(gt=0)
    wavelength_nm: float = Field(gt=0)
    n_clad: float = Field(gt=0)
    polarization: str = Field(pattern="^(TE|TM)$")
    devices: list[dict] = Field(default_factory=list)
    input_power_uw: float = Field(default=10.0, gt=0)


@app.post("/api/circuit")
def circuit(payload: CircuitInput):
    result = analyze_circuit(**payload.model_dump())
    if result.get("error"):
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.post("/api/fdtd")
def fdtd(payload: CircuitInput):
    return run_fdtd(
        payload.width_nm,
        payload.height_nm,
        payload.wavelength_nm,
        payload.n_clad,
        payload.polarization,
        payload.devices,
    )


class TopologyInput(CircuitInput):
    mfs_nm: float = Field(default=150.0, gt=0)
    min_gap_nm: float = Field(default=150.0, gt=0)
    etch_nm: float = Field(default=0.0)
    rounding_nm: float = Field(default=20.0, ge=0)
    beta: float = Field(default=8.0, gt=0)
    steps: int = Field(default=8, ge=0, le=20)
    kappa_target: float | None = Field(default=None, ge=0, le=1)


@app.post("/api/topology")
def topology(payload: TopologyInput):
    result = run_topology(
        payload.width_nm,
        payload.height_nm,
        payload.wavelength_nm,
        payload.n_clad,
        payload.polarization,
        payload.devices,
        mfs_nm=payload.mfs_nm,
        min_gap_nm=payload.min_gap_nm,
        etch_nm=payload.etch_nm,
        rounding_nm=payload.rounding_nm,
        beta=payload.beta,
        steps=payload.steps,
        kappa_target=payload.kappa_target,
    )
    if result.get("error"):
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.post("/api/topology/sweep")
def topology_sweep(payload: TopologyInput):
    result = run_robustness_sweep(
        payload.width_nm,
        payload.height_nm,
        payload.wavelength_nm,
        payload.n_clad,
        payload.polarization,
        payload.devices,
        steps=min(payload.steps, 6) if payload.steps else 4,
        kappa_target=payload.kappa_target,
        beta=payload.beta,
    )
    if result.get("error"):
        raise HTTPException(status_code=400, detail=result["error"])
    return result
