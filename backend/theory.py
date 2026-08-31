"""Analytic checks against the compact-model circuit numbers."""

from __future__ import annotations

import math


def _rel_error(model: float | None, theory: float | None) -> float | None:
    if model is None or theory is None:
        return None
    denom = max(abs(theory), 1e-15)
    return abs(model - theory) / denom


def _row(
    id_: str,
    name: str,
    formula: str,
    model: float | None,
    theory: float | None,
    unit: str,
    note: str,
    tol: float = 0.05,
) -> dict:
    err = _rel_error(model, theory)
    ok = None if err is None else bool(err <= tol)
    return {
        "id": id_,
        "name": name,
        "formula": formula,
        "model": model,
        "theory": theory,
        "unit": unit,
        "rel_error": err,
        "ok": ok,
        "tol": tol,
        "note": note,
    }


def circuit_theory(
    *,
    wavelength_nm: float,
    radius_um: float,
    n_eff: float,
    n_g: float,
    kappa: float,
    config: str,
    ring: dict,
    critical_kappa: float | None,
    coupler: dict | None = None,
    coupler_length_um: float | None = None,
) -> dict:
    """Compare compact-model outputs to textbook resonator / coupler formulas."""
    L_um = 2 * math.pi * radius_um
    lam0 = float(ring["resonance_nm"])
    fsr_model = float(ring["fsr_nm"])
    # Textbook FSR at the resonance wavelength (model FSR in ring.py uses laser λ).
    fsr_theory = ((lam0 * 1e-3) ** 2) / (n_g * L_um) * 1e3

    m = int(ring["mode_order"])
    # Resonance condition: m · λ = n_eff · 2πR  →  λ0 = n_eff · L / m
    lam0_theory = (n_eff * L_um / m) * 1e3 if m else None

    a = float(ring["round_trip_amplitude"])
    # Round-trip power loss 1 − a²; for all-pass critical κ = 1 − a².
    power_rt_loss = 1.0 - a * a
    k_crit = critical_kappa
    if k_crit is None:
        from ring import kappa_for_critical

        k_crit = kappa_for_critical(a, config)

    q_analytic = float(ring["q"]) if ring.get("q") is not None else None
    extracted = ring.get("extracted") or {}
    q_spectrum = extracted.get("q_loaded")
    if q_spectrum is not None:
        q_spectrum = float(q_spectrum)

    checks = [
        _row(
            "fsr",
            "FSR",
            "λ₀² / (n_g · 2πR)",
            fsr_model,
            fsr_theory,
            "nm",
            f"Model FSR uses laser λ = {wavelength_nm:.1f} nm; theory uses resonance λ₀. "
            "Small Δ is expected.",
            tol=0.02,
        ),
        _row(
            "resonance",
            "Resonance λ₀",
            "λ₀ = n_eff · 2πR / m",
            lam0,
            lam0_theory,
            "nm",
            "Nearest longitudinal mode to the laser wavelength.",
            tol=1e-6,
        ),
        _row(
            "kappa_crit",
            "κ vs κ_crit",
            "κ_crit = 1 − a² (all-pass) or 1 − a (add-drop)",
            float(kappa),
            float(k_crit),
            "",
            f"Round-trip amplitude a = {a:.4f}; power loss 1−a² = {power_rt_loss:.4f}. "
            f"Regime from model: {ring.get('coupling_regime', '—')}.",
            tol=0.15,
        ),
        _row(
            "q_loaded",
            "Q_loaded",
            "π n_g L √r / (λ (1−r)) vs spectrum notch",
            q_spectrum,
            q_analytic,
            "",
            "Spectrum fit (model column) vs closed-form loaded Q (theory column).",
            tol=0.08,
        ),
    ]

    if coupler and coupler.get("kappa0_per_um") and coupler.get("beat_length_um"):
        kappa0 = float(coupler["kappa0_per_um"])
        beat_model = float(coupler["beat_length_um"])
        # L_π = π / (2 κ₀) for power transfer length of a symmetric DC.
        beat_theory = (math.pi / 2) / kappa0 if kappa0 > 1e-12 else None
        # Consistency: invert measured power split κ = sin²(κ₀ L).
        k_pow = float(coupler.get("kappa") or coupler.get("t_drop") or 0.0)
        raw_len = coupler_length_um if coupler_length_um is not None else coupler.get("length_um")
        length = float(raw_len) if isinstance(raw_len, (int, float)) else 0.0
        beat_from_split = None
        if 0 < k_pow < 1 and length > 0:
            phase = math.asin(min(1.0, math.sqrt(k_pow)))
            k0_eff = phase / length
            if k0_eff > 1e-12:
                beat_from_split = (math.pi / 2) / k0_eff
        note = "From FDTD-fitted κ₀."
        if beat_from_split is not None:
            note = f"From FDTD-fitted κ₀. Split-inverted L_π = {beat_from_split:.3f} μm."
        checks.append(
            _row(
                "beat_length",
                "Coupler beat length",
                "L_π = π / (2 κ₀)",
                beat_model,
                beat_theory,
                "μm",
                note,
                tol=1e-6,
            )
        )

    n_ok = sum(1 for c in checks if c["ok"] is True)
    n_bad = sum(1 for c in checks if c["ok"] is False)
    n_na = sum(1 for c in checks if c["ok"] is None)

    return {
        "source": "compact-model vs analytic",
        "checks": checks,
        "summary": {"ok": n_ok, "fail": n_bad, "na": n_na},
        "note": "Textbook formulas on the same n_eff / n_g / a — not a full-wave 3D check.",
    }
