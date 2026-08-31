"""Full-circuit S-parameters: waveguide + coupler + ring + detector together."""

from __future__ import annotations

from coupler import analyze_coupler, geometry_for_kappa
from detector import analyze_detector
from fdtd.setup import describe_fdtd
from ring import analyze_ring, kappa_for_critical, radius_for_laser, transmission_at_laser
from sparams import pack_s, power_db, scale_spectrum_power
from theory import circuit_theory
from waveguide import analyze_waveguide


def analyze_circuit(
    width_nm: float,
    height_nm: float,
    wavelength_nm: float,
    n_clad: float,
    polarization: str,
    devices: list[dict],
    input_power_uw: float = 10.0,
) -> dict:
    mode = analyze_waveguide(
        width_nm, height_nm, wavelength_nm, n_clad, polarization, length_um=100.0
    )
    if mode is None:
        return {"error": "No guided mode for this cross-section"}

    wgs = [d for d in devices if d.get("type") == "waveguide"]
    couplers = [d for d in devices if d.get("type") == "coupler"]
    rings = [d for d in devices if d.get("type") == "ring"]
    dets = [d for d in devices if d.get("type") == "detector"]

    t_wg = 1.0
    loss_db = 0.0
    for w in wgs:
        part = analyze_waveguide(
            width_nm,
            height_nm,
            wavelength_nm,
            n_clad,
            polarization,
            length_um=float(w.get("length_um", 100)),
        )
        if part:
            t_wg *= part["t_through"]
            loss_db += part["loss_db"]

    coupler_results = []
    for c in couplers:
        coupler_results.append(
            analyze_coupler(
                float(c["gap_nm"]),
                float(c["length_um"]),
                width_nm,
                wavelength_nm,
                polarization,
                height_nm,
            )
        )

    ring = rings[0] if rings else None
    kappa = 0.05
    if ring and coupler_results:
        kappa = _kappa_for_ring(ring, rings, couplers, coupler_results)

    ring_result = None
    if ring and mode:
        ring_result = analyze_ring(
            radius_um=float(ring["radius_um"]),
            wavelength_nm=wavelength_nm,
            n_eff=mode["n_eff"],
            n_g=mode["n_g"],
            kappa=kappa,
            loss_db_per_cm=mode["loss_db_per_cm"],
            dn_eff_dn=mode["dn_eff_dn"],
            config=ring.get("config", "all-pass"),
            width_nm=width_nm,
            polarization=polarization,
        )

    if ring_result:
        t_through = ring_result["t_through"] * t_wg
        t_drop = (ring_result["t_drop"] or 0.0) * t_wg
        s21 = (t_through**0.5) * (1 + 0j)
        s31 = (t_drop**0.5) * (1 + 0j) if t_drop > 0 else 0j
        s = pack_s(0j, s21, s31, 0j)
        spectrum = scale_spectrum_power(ring_result["spectrum"], t_wg)
        extracted = ring_result["extracted"]
        extracted = {
            **extracted,
            "t_through": extracted.get("t_through", 0) * t_wg,
            "t_drop": (extracted["t_drop"] * t_wg) if extracted.get("t_drop") is not None else None,
        }
        core = {
            "resonance_nm": ring_result["resonance_nm"],
            "fsr_nm": ring_result["fsr_nm"],
            "q": ring_result["q"],
            "q_intrinsic": ring_result["q_intrinsic"],
            "q_coupling": ring_result["q_coupling"],
            "fwhm_nm": ring_result["fwhm_nm"],
            "coupling_regime": ring_result["coupling_regime"],
            "sensitivity_nm_per_riu": ring_result["sensitivity_nm_per_riu"],
            "lod_riu": ring_result["lod_riu"],
            "extinction_db": ring_result["extinction_db"],
            "extracted": extracted,
            "spectrum": spectrum,
        }
    elif coupler_results:
        c0 = coupler_results[0]
        t_through = c0["t_through"] * t_wg
        t_drop = c0["t_drop"] * t_wg
        s = pack_s(0j, t_through**0.5, t_drop**0.5, 0j)
        through_l = [v * t_wg for v in c0.get("through_vs_wavelength", [])]
        drop_l = [v * t_wg for v in c0.get("cross_vs_wavelength", [])]
        core = {
            "spectrum": {
                "wavelength_nm": c0.get("wavelength_nm", []),
                "through": through_l,
                "drop": drop_l,
                "t_through": through_l,
                "t_drop": drop_l,
                "s21_db": [power_db(v) for v in through_l],
                "s31_db": [power_db(v) for v in drop_l],
            },
            "extracted": c0.get("extracted"),
        }
    else:
        t_through = t_wg
        t_drop = 0.0
        s = pack_s(0j, t_through**0.5)
        core = {"spectrum": None, "extracted": None}

    pin = input_power_uw

    detector_out = []
    for i, d in enumerate(dets):
        port = _detector_port(d, ring, wgs)
        t_port = t_drop if port == "drop" and t_drop > 0 else t_through
        power = pin * t_port
        det = analyze_detector(
            wavelength_nm=wavelength_nm,
            optical_power_uw=max(power, 1e-9),
            responsivity_a_per_w=float(d.get("responsivity_a_per_w", 0.9)),
            dark_current_na=float(d.get("dark_current_na", 5)),
            bandwidth_mhz=float(d.get("bandwidth_mhz", 100)),
            load_ohm=float(d.get("load_ohm", 50)),
        )
        det["port"] = port
        det["optical_power_uw"] = power
        detector_out.append(det)

    sweep = None
    radius_tune = None
    critical = None
    if ring and mode and ring_result:
        radius_tune = radius_for_laser(mode["n_eff"], wavelength_nm, float(ring["radius_um"]))
        sweep = _analyte_sweep(
            width_nm,
            height_nm,
            wavelength_nm,
            n_clad,
            polarization,
            ring,
            kappa,
        )
        k_crit = kappa_for_critical(
            ring_result["round_trip_amplitude"], ring.get("config", "all-pass")
        )
        c0 = couplers[0] if couplers else {"gap_nm": 200.0, "length_um": 12.0}
        geo = geometry_for_kappa(
            k_crit,
            float(c0.get("gap_nm", 200)),
            float(c0.get("length_um", 12)),
            width_nm,
            height_nm,
            wavelength_nm,
            polarization,
        )
        critical = {
            "kappa": k_crit,
            "a": ring_result["round_trip_amplitude"],
            "t": ring_result["self_coupling"],
            **geo,
        }

    theory = None
    if ring and mode and ring_result:
        c0_dev = couplers[0] if couplers else None
        theory = circuit_theory(
            wavelength_nm=wavelength_nm,
            radius_um=float(ring["radius_um"]),
            n_eff=mode["n_eff"],
            n_g=mode["n_g"],
            kappa=kappa,
            config=ring.get("config", "all-pass"),
            ring=ring_result,
            critical_kappa=critical["kappa"] if critical else None,
            coupler=coupler_results[0] if coupler_results else None,
            coupler_length_um=float(c0_dev["length_um"]) if c0_dev else None,
        )

    return {
        "mode": mode,
        "s_parameters": s,
        "t_through": t_through,
        "t_drop": t_drop if t_drop > 0 else None,
        "t_through_db": power_db(t_through),
        "t_drop_db": power_db(t_drop) if t_drop > 0 else None,
        "kappa": kappa if (ring or coupler_results) else None,
        "wg_loss_db": loss_db,
        "device_count": len(devices),
        "detectors": detector_out,
        "radius_for_laser_um": radius_tune,
        "analyte_sweep": sweep,
        "critical": critical,
        "theory": theory,
        "fdtd": describe_fdtd(
            width_nm,
            height_nm,
            wavelength_nm,
            n_clad,
            polarization,
            devices,
            run_mode=False,
        ),
        **core,
    }


def _analyte_sweep(
    width_nm: float,
    height_nm: float,
    wavelength_nm: float,
    n_clad: float,
    polarization: str,
    ring: dict,
    kappa: float,
) -> dict:
    span = 0.02 if n_clad >= 1.2 else 0.04
    n_lo = max(1.0, n_clad - span)
    n_hi = min(1.6, n_clad + span)
    ns = [n_lo + i * (n_hi - n_lo) / 20 for i in range(21)]
    n_list = []
    res_nm = []
    t_th = []
    t_dr = []
    radius = float(ring["radius_um"])
    config = ring.get("config", "all-pass")
    for n in ns:
        mode = analyze_waveguide(
            width_nm,
            height_nm,
            wavelength_nm,
            n,
            polarization,
            length_um=100.0,
            include_field=False,
        )
        if mode is None:
            continue
        rr = analyze_ring(
            radius_um=radius,
            wavelength_nm=wavelength_nm,
            n_eff=mode["n_eff"],
            n_g=mode["n_g"],
            kappa=kappa,
            loss_db_per_cm=mode["loss_db_per_cm"],
            dn_eff_dn=mode["dn_eff_dn"],
            config=config,
            width_nm=width_nm,
            polarization=polarization,
        )
        tt, td = transmission_at_laser(
            radius,
            wavelength_nm,
            mode["n_eff"],
            kappa,
            mode["loss_db_per_cm"],
            config,
            width_nm,
            polarization,
        )
        n_list.append(n)
        res_nm.append(rr["resonance_nm"])
        t_th.append(tt)
        t_dr.append(td if config == "add-drop" else None)

    shift = []
    if res_nm:
        i0 = min(range(len(n_list)), key=lambda i: abs(n_list[i] - n_clad))
        lam0 = res_nm[i0]
        shift = [r - lam0 for r in res_nm]
        dn = n_list[-1] - n_list[0]
        dlam = res_nm[-1] - res_nm[0]
        slope = dlam / dn if dn else 0.0
    else:
        slope = 0.0

    drop = [v for v in t_dr if v is not None]
    return {
        "n_clad": n_list,
        "resonance_nm": res_nm,
        "shift_nm": shift,
        "t_through": t_th,
        "t_drop": drop if len(drop) == len(n_list) else None,
        "sensitivity_nm_per_riu": slope,
        "n_design": n_clad,
    }


def _kappa_for_ring(ring: dict, rings: list, couplers: list, results: list[dict]) -> float:
    rx, ry = float(ring.get("x", 0)), float(ring.get("y", 0))
    best_i = 0
    best_d = 1e18
    for i, c in enumerate(couplers):
        d = (float(c.get("x", 0)) - rx) ** 2 + (float(c.get("y", 0)) - ry) ** 2
        if d < best_d:
            best_d = d
            best_i = i
    return results[best_i]["kappa"] if results else 0.05


def _detector_port(det: dict, ring: dict | None, wgs: list[dict]) -> str:
    if not ring or ring.get("config") != "add-drop" or len(wgs) < 2:
        return "through"
    dy = float(det.get("y", 0))
    ry = float(ring.get("y", 0))
    return "drop" if dy < ry else "through"
