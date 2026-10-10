"""Uniform-ΔT corners, heater heat solve, and thermo-optic TO."""

from __future__ import annotations

import unittest

import numpy as np

from circuit import analyze_circuit
from thermo import (
    DN_DT_SI,
    G_SINK,
    apply_heat,
    apply_heat_transpose,
    adjoint_heat,
    default_coupler_heater,
    dn_eff_dt,
    dsheet_k_drho,
    heat_kappa_gradient,
    heater_layout,
    kappa0_at,
    neighbor_sum,
    paint_heaters,
    bilinear_sample,
    bilinear_scatter,
    coupled_window_heat,
    mean_on_mask,
    resonance_shift_nm,
    sheet_k,
    solve_circuit_heat,
    solve_heat,
    window_heat_gradient,
)
from thermo_adjoint import (
    optical_adjoint_field,
    permittivity_gradient,
    port_field_grad,
    solve_helmholtz,
    te_apply,
    te_permittivity_gradient,
    thermo_optic_terms,
)
from topology import (
    _desensitization_sensitivity,
    _eps_pml,
    _etch_vjp,
    _fdfd_split,
    _grayscale_sensitivity,
    _jsrc_jmon,
    _port_masks,
    _project_design,
    _state_weight,
    helmholtz_filter,
    litho_etch_surrogate,
    run_topology,
    tanh_project,
)


def _devices(heater: bool = False) -> list[dict]:
    out = [
        {"type": "waveguide", "length_um": 100.0, "x": 260, "y": 210},
        {"type": "coupler", "gap_nm": 200.0, "length_um": 12.0, "x": 250, "y": 210},
        {"type": "ring", "radius_um": 10.0, "config": "all-pass", "x": 250, "y": 134},
    ]
    if heater:
        out.append(
            {
                "type": "heater",
                "x": 235,
                "y": 134,
                "power_mw": 10.0,
                "width_um": 2.0,
                "length_um": 40.0,
            }
        )
    return out


def _circuit(heater: bool = False) -> dict:
    return analyze_circuit(
        width_nm=450.0,
        height_nm=220.0,
        wavelength_nm=1550.0,
        n_clad=1.33,
        polarization="TE",
        devices=_devices(heater),
    )


class ThermoCornersTest(unittest.TestCase):
    def test_dn_eff_dt_is_between_clad_and_silicon(self):
        dndt = dn_eff_dt(0.81, 0.10, 1.444)
        self.assertGreater(dndt, 0.0)
        self.assertLess(dndt, DN_DT_SI)
        self.assertAlmostEqual(dndt, 0.81 * DN_DT_SI + 0.10 * 1.0e-5, places=9)

    def test_resonance_redshifts_and_kappa_drops_when_hot(self):
        result = _circuit()
        thermal = result["thermal_corners"]
        self.assertIsNotNone(thermal)
        by_dt = {p["delta_t_k"]: p for p in thermal["points"]}
        cold = by_dt[-20.0]
        hot = by_dt[40.0]
        self.assertGreater(hot["resonance_nm"], cold["resonance_nm"])
        self.assertLess(hot["kappa"], cold["kappa"])
        self.assertAlmostEqual(by_dt[0.0]["shift_nm"], 0.0, places=9)
        self.assertAlmostEqual(
            thermal["dlambda_dt_nm_per_k"],
            resonance_shift_nm(result["resonance_nm"], result["mode"]["n_g"], thermal["dn_eff_dt"], 1.0),
            places=9,
        )

    def test_worst_thermal_corners_are_extreme_delta_t(self):
        thermal = _circuit()["thermal_corners"]
        self.assertIn(thermal["worst_lambda"]["delta_t_k"], (-20.0, 40.0))
        self.assertIn(thermal["worst_kappa"]["delta_t_k"], (-20.0, 40.0))
        self.assertGreater(thermal["worst_lambda"]["abs_dlambda_nm"], 0.0)

    def test_hotter_core_weakens_kappa0(self):
        self.assertLess(kappa0_at(0.05, 1.5e-4, 1.444, 20.0), 0.05)
        self.assertGreater(kappa0_at(0.05, 1.5e-4, 1.444, -20.0), 0.05)


class HeatSolveTest(unittest.TestCase):
    def test_temperature_peaks_under_the_heater(self):
        xs = np.linspace(-4.0, 4.0, 41)
        ys = np.linspace(-4.0, 4.0, 41)
        xx, yy = np.meshgrid(xs, ys, indexing="ij")
        q = np.where((np.abs(xx) < 1.0) & (np.abs(yy) < 2.0), 8.0e7, 0.0)
        temp = solve_heat(xs, ys, q, sheet_k(np.zeros_like(q)), G_SINK)
        self.assertGreater(temp.max(), 2.0)
        self.assertGreater(temp[20, 20], temp[2, 2])

    def test_heat_is_linear_in_power(self):
        xs = np.linspace(-3.0, 3.0, 31)
        ys = np.linspace(-3.0, 3.0, 31)
        heaters = [default_coupler_heater(0.45, 3.0, 1.0, power_mw=10.0)]
        k = sheet_k(np.zeros((xs.size, ys.size)))
        t1 = solve_heat(xs, ys, paint_heaters(xs, ys, heaters, 1.0), k)
        t2 = solve_heat(xs, ys, paint_heaters(xs, ys, heaters, 0.5), k)
        self.assertGreater(t1.max(), 0.5)
        self.assertAlmostEqual(t1.max() / (t2.max() + 1e-12), 2.0, places=1)

    def test_heater_layout_sits_by_the_ring(self):
        layout = heater_layout(_devices(heater=True), 450.0)
        self.assertEqual(len(layout), 1)
        self.assertGreater(layout[0]["power_mw"], 0.0)
        self.assertLess(layout[0]["width"], 5.0)

    def test_heater_outline_sits_on_the_hot_spot(self):
        heat = solve_circuit_heat(_devices(heater=True), 450.0)
        self.assertIsNotNone(heat)
        field = heat["field"]
        self.assertTrue(field["polygons"])
        ys = [p["y"] for p in field["polygons"][0]]
        temp = np.asarray(field["intensity"])
        ix = int(np.unravel_index(int(temp.argmax()), temp.shape)[0])
        x_hot = field["z_um"][ix]
        self.assertGreaterEqual(max(ys) + 1.0, x_hot)
        self.assertLessEqual(min(ys) - 1.0, x_hot)
        guides = field["guides"]
        self.assertTrue(any(g["height"] < 3.5 for g in guides))

    def test_circuit_heater_redshifts_more_than_heater_off(self):
        heat = solve_circuit_heat(_devices(heater=True), 450.0)
        self.assertIsNotNone(heat)
        self.assertGreater(heat["dt_max_k"], 1.0)
        thermal = _circuit(heater=True)["thermal_corners"]
        self.assertIsNotNone(thermal["heater"])
        by_s = {p["scale"]: p for p in thermal["heater"]["points"]}
        self.assertAlmostEqual(by_s[0.0]["shift_nm"], 0.0, places=9)
        self.assertGreater(by_s[1.0]["shift_nm"], by_s[0.0]["shift_nm"])
        self.assertGreater(thermal["heater"]["field"]["dt_max_k"], 0.0)


class ThermoOpticLoopTest(unittest.TestCase):
    def test_thermo_optic_to_scores_cold_and_hot(self):
        result = run_topology(
            450.0,
            220.0,
            1550.0,
            1.33,
            "TE",
            _devices(heater=True),
            steps=1,
            kappa_target=0.12,
            thermo_optic=True,
        )
        self.assertTrue(result["thermo_optic"])
        self.assertIsNotNone(result["thermal"])
        self.assertIsNotNone(result["field_t"])
        chip = solve_circuit_heat(_devices(heater=True), 450.0)
        self.assertIsNotNone(chip)
        self.assertAlmostEqual(
            result["thermal"]["dt_max_k"], chip["dt_max_k"], delta=0.2 * chip["dt_max_k"]
        )
        self.assertGreater(result["thermal"]["dt_ring_k"], 0.0)
        self.assertGreater(result["thermal"]["shift_nm"], 0.0)
        self.assertIn("t_drop_off", result["thermal"])
        self.assertIn("t_drop_hot", result["thermal"])
        if result["history"]:
            step = result["history"][0]
            self.assertIn(step.get("thermal"), ("off", "hot"))
            self.assertIn(step.get("gradient"), ("optical", "optical+heat"))
            self.assertIn("drift", step)
            self.assertGreater(step.get("shift_nm", 0.0), 0.0)
            if step.get("t_drop_hot") is not None:
                self.assertEqual(step.get("gradient"), "optical+heat")


def _rel(actual, expected) -> float:
    scale = max(abs(expected), abs(actual), 1e-12)
    return abs(actual - expected) / scale


class AdjointGradientTest(unittest.TestCase):
    def test_heat_operator_matches_its_transpose(self):
        rng = np.random.default_rng(0)
        kappa = rng.random((5, 6)) + 0.2
        left = rng.random((5, 6))
        right = rng.random((5, 6))
        dx = 2e-7
        applied = apply_heat(left, kappa, dx, G_SINK)
        transposed = apply_heat_transpose(right, kappa, dx, G_SINK)
        self.assertLess(_rel(float(np.vdot(right, applied)), float(np.vdot(transposed, left))), 1e-9)

    def test_constant_conductivity_is_self_adjoint(self):
        kappa = np.full((6, 5), 1.7e-6)
        field = np.arange(30, dtype=float).reshape(6, 5)
        dx = 1.5e-7
        forward = apply_heat(field, kappa, dx)
        self.assertLess(
            np.max(np.abs(forward - apply_heat_transpose(field, kappa, dx))) / (np.max(np.abs(forward)) + 1e-30),
            1e-12,
        )

    def test_screened_poisson_stencil_is_symmetric(self):
        r2 = 1.7
        field = np.linspace(-0.2, 0.8, 20).reshape(4, 5)
        probe = np.linspace(0.1, 1.0, 20).reshape(4, 5)

        def apply(grid):
            return (1.0 + 4.0 * r2) * grid - r2 * neighbor_sum(grid)

        self.assertAlmostEqual(float(np.vdot(probe, apply(field))), float(np.vdot(field, apply(probe))), places=9)

    def test_heat_adjoint_matches_kappa_finite_difference(self):
        xs = np.linspace(-1.6, 1.6, 15)
        ys = np.linspace(-1.2, 1.2, 13)
        xx, yy = np.meshgrid(xs, ys, indexing="ij")
        kappa = sheet_k(np.full(xx.shape, 0.25)) * (1.0 + 0.4 * np.exp(-(xx**2 + yy**2) / 0.8))
        q = np.exp(-(xx**2) / 0.35 - yy**2) * 5.0e7

        def objective(k_sheet):
            temp = solve_heat(xs, ys, q, k_sheet)
            return float(np.sum(temp**2)), temp

        base, temp = objective(kappa)
        residual = apply_heat(temp, kappa, abs(xs[1] - xs[0]) * 1e-6) - q
        self.assertLess(np.linalg.norm(residual) / (np.linalg.norm(q) + 1e-30), 1e-6)
        lam = adjoint_heat(xs, kappa, 2.0 * temp)
        grad = heat_kappa_gradient(lam, temp, xs)
        i, j = 7, 6
        delta = 1e-5 * kappa[i, j]
        plus, _ = objective(_bump(kappa, i, j, delta))
        minus, _ = objective(_bump(kappa, i, j, -delta))
        fd = (plus - minus) / (2.0 * delta)
        self.assertLess(_rel(grad[i, j], fd), 2e-3)
        self.assertGreater(base, 0.0)

    def test_permittivity_chain_matches_density_finite_difference(self):
        xs = np.linspace(-1.5, 1.5, 13)
        ys = np.linspace(-1.2, 1.2, 11)
        xx, yy = np.meshgrid(xs, ys, indexing="ij")
        rho = np.full(xx.shape, 0.35)
        rho[3:9, 2:9] = 0.72
        q = np.exp(-(xx**2) / 0.4 - (yy**2) / 0.5) * 8.0e7
        n_core, n_bg = 2.5, 1.45
        dndt_core, dndt_clad = 0.2, 0.02

        def objective(density):
            temp = solve_heat(xs, ys, q, sheet_k(density))
            n_c = n_core + dndt_core * temp
            n_b = n_bg + dndt_clad * temp
            eps = (1.0 - density) * n_b**2 + density * n_c**2
            return float(np.sum(eps)), temp

        value, temp = objective(rho)
        direct, thermal = thermo_optic_terms(
            np.ones_like(rho), rho, temp, rho, xs, n_core, n_bg, dndt_core, dndt_clad
        )
        grad = direct + thermal
        i, j = 5, 4
        delta = 1e-4
        plus, _ = objective(_bump(rho, i, j, delta))
        minus, _ = objective(_bump(rho, i, j, -delta))
        fd = (plus - minus) / (2.0 * delta)
        self.assertLess(_rel(grad[i, j], fd), 2e-3)
        self.assertGreater(abs(thermal[i, j]), 1e-4 * abs(direct[i, j]))
        self.assertGreater(value, 0.0)
        self.assertAlmostEqual(
            float(dsheet_k_drho(rho)[i, j]),
            float((sheet_k(_bump(rho, i, j, delta)) - sheet_k(_bump(rho, i, j, -delta)))[i, j] / (2 * delta)),
            places=8,
        )

    def test_optical_adjoint_matches_port_finite_difference(self):
        ny = 22
        w_um, gap_um = 0.40, 0.28
        xs = np.linspace(-0.75, 1.15, 20)
        ys = np.linspace(0.0, 2.1, ny)
        dx = float(xs[1] - xs[0])
        rho = np.zeros((xs.size, ny))
        rho[(xs >= -w_um) & (xs <= 0.0), :] = 1.0
        rho[(xs >= gap_um) & (xs <= gap_um + w_um), :] = 1.0
        wavelength_nm = 1550.0
        n_core, n_bg = 2.45, 1.44
        npml = 3
        eps, k0, npml = _eps_pml(rho, xs, ys, dx, n_core, n_bg, wavelength_nm, npml)
        jsrc, jmon = _jsrc_jmon(ny, npml, dx)
        through, drop = _port_masks(xs, w_um, gap_um)
        target = 0.25

        def objective(perm):
            rhs = np.zeros_like(perm, dtype=np.complex128)
            rhs[:, jsrc] = -k0**2 * np.exp(-((xs - 0.5 * w_um) / (0.45 * w_um)) ** 2)
            field, ok = solve_helmholtz(perm, k0, dx, rhs, tol=1e-12, maxiter=2000)
            self.assertTrue(ok)
            p_th = float(np.sum(np.abs(field[through, jmon]) ** 2))
            p_dr = float(np.sum(np.abs(field[drop, jmon]) ** 2))
            t_dr = p_dr / (p_th + p_dr + 1e-30)
            return (t_dr - target) ** 2, field

        base, field = objective(eps)
        grad_field = port_field_grad(field, through, drop, jmon, target)
        adjoint, ok = optical_adjoint_field(eps, k0, dx, grad_field, tol=1e-12, maxiter=2000)
        self.assertTrue(ok)
        sens = permittivity_gradient(field, adjoint, k0)
        i, j = 4, 10
        delta = 1e-4
        plus, _ = objective(_bump(eps, i, j, delta))
        minus, _ = objective(_bump(eps, i, j, -delta))
        fd = (plus - minus) / (2.0 * delta)
        self.assertLess(_rel(sens[i, j], fd), 2e-2)
        self.assertNotAlmostEqual(base, 0.0, places=6)

    def test_state_weight_tracks_target_and_drift(self):
        off = {"thermal": "off", "t_drop": 0.20}
        hot = {"thermal": "hot", "t_drop": 0.50}
        target = 0.10
        self.assertAlmostEqual(_state_weight(hot, off, target, True), 2.0 * 0.40 + 2.0 * 0.30)
        self.assertAlmostEqual(_state_weight(off, hot, target, False), -2.0 * 0.30)

    def test_both_temperatures_match_density_finite_difference(self):
        ny = 18
        w_um, gap_um = 0.40, 0.28
        xs = np.linspace(-0.75, 1.05, 16)
        ys = np.linspace(0.0, 1.7, ny)
        dx = float(xs[1] - xs[0])
        rho = np.zeros((xs.size, ny))
        rho[(xs >= -w_um) & (xs <= 0.0), :] = 1.0
        rho[(xs >= gap_um) & (xs <= gap_um + w_um), :] = 1.0
        xx, yy = np.meshgrid(xs, ys, indexing="ij")
        design = (xx > 0.02) & (xx < gap_um - 0.02) & (yy > 0.85) & (yy < 1.25)
        rho[design] = 0.45
        self.assertGreater(int(design.sum()), 2)
        beta, eta, r_px = 4.0, 0.5, 0.9
        n_core, n_bg = 2.45, 1.44
        dndt_core, dndt_clad = 0.04, 0.0
        wavelength_nm, npml, target = 1550.0, 3, 0.15
        heater = [
            {
                "x0": float(xs[0] + 0.02),
                "y0": 0.55,
                "width": 0.22,
                "height": 0.45,
                "power_w": 8.0e-5,
                "power_mw": 0.08,
            }
        ]

        def forward(density):
            rho_f = helmholtz_filter(density, r_px)
            rho_p = _project_design(rho_f, design, density, beta, eta)
            temp = solve_heat(xs, ys, paint_heaters(xs, ys, heater), sheet_k(rho_p))
            cold, e_off, ok_off = _fdfd_split(
                rho_p, xs, ys, dx, n_core, n_bg, wavelength_nm, w_um, gap_um, npml,
                tol=1e-8, maxiter=800,
            )
            hot, e_hot, ok_hot = _fdfd_split(
                rho_p, xs, ys, dx, n_core, n_bg, wavelength_nm, w_um, gap_um, npml,
                None, temp, dndt_core, dndt_clad, tol=1e-8, maxiter=800,
            )
            self.assertTrue(ok_off and ok_hot and cold is not None and hot is not None)
            t_off, t_hot = float(cold[1]), float(hot[1])
            return {
                "J": (t_off - target) ** 2 + (t_hot - target) ** 2 + (t_hot - t_off) ** 2,
                "rho_f": rho_f,
                "rho_p": rho_p,
                "temp": temp,
                "t_off": t_off,
                "t_hot": t_hot,
                "e_off": e_off,
                "e_hot": e_hot,
                "split_off": cold,
                "split_hot": hot,
            }

        base = forward(rho)
        self.assertGreater(abs(base["t_hot"] - base["t_off"]), 1e-4)

        def corner(name, thermal, temp, split, field):
            return {
                "name": name,
                "eta": eta,
                "thermal": thermal,
                "temp": temp,
                "rho_p": base["rho_p"],
                "split": split,
                "e_fwd": field,
                "t_drop": float(split[1]),
            }

        off_c = corner("mid", "off", None, base["split_off"], base["e_off"])
        hot_c = corner("mid", "hot", base["temp"], base["split_hot"], base["e_hot"])
        sens = _desensitization_sensitivity(
            off_c, hot_c, base["rho_f"], design, xs, ys, dx, r_px, beta,
            n_core, n_bg, wavelength_nm, w_um, gap_um, target, npml, dndt_core, dndt_clad,
            adj_tol=1e-8,
        )
        self.assertIsNotNone(sens)
        free = np.argwhere(design & (rho > 0.2) & (rho < 0.8))
        i, j = max(free, key=lambda p: abs(sens[int(p[0]), int(p[1])]))
        i, j = int(i), int(j)
        delta = 5e-4
        plus = forward(_bump(rho, i, j, delta))
        minus = forward(_bump(rho, i, j, -delta))
        fd = (plus["J"] - minus["J"]) / (2.0 * delta)
        self.assertLess(_rel(float(sens[i, j]), fd), 0.05)

    def test_grayscale_matches_density_finite_difference(self):
        rng = np.random.default_rng(2)
        rho = 0.25 + 0.5 * rng.random((12, 10))
        design = np.zeros(rho.shape, dtype=bool)
        design[2:10, 2:8] = True
        beta, weight, r_px = 6.0, 0.02, 1.1
        rho_f = helmholtz_filter(rho, r_px)

        def objective(density):
            filtered = helmholtz_filter(density, r_px)
            projected_rho = tanh_project(filtered, beta, 0.5)
            return weight * float(np.mean(4.0 * projected_rho[design] * (1.0 - projected_rho[design])))

        sens = _grayscale_sensitivity(rho_f, design, beta, weight, r_px)
        free = np.argwhere(design)
        i, j = max(free, key=lambda p: abs(sens[int(p[0]), int(p[1])]))
        i, j = int(i), int(j)
        delta = 1e-4
        fd = (objective(_bump(rho, i, j, delta)) - objective(_bump(rho, i, j, -delta))) / (2.0 * delta)
        self.assertLess(_rel(float(sens[i, j]), fd), 0.02)

    def test_etch_vjp_matches_finite_difference(self):
        rng = np.random.default_rng(3)
        rho = 0.3 + 0.4 * rng.random((11, 9))
        etch_px, round_px = 0.35, 0.8

        def etched(density):
            return litho_etch_surrogate(density, etch_px, round_px)

        base = etched(rho)
        sens = _etch_vjp(2.0 * base, rho, etch_px, round_px)
        i, j = 5, 4
        delta = 1e-4
        plus = etched(_bump(rho, i, j, delta))
        minus = etched(_bump(rho, i, j, -delta))
        fd = float(np.sum(plus**2) - np.sum(minus**2)) / (2.0 * delta)
        self.assertLess(_rel(float(sens[i, j]), fd), 0.02)

    def test_te_adjoint_matches_permittivity_finite_difference(self):
        rng = np.random.default_rng(4)
        eps = (1.6 + 0.5 * rng.random((9, 8))) + 1j * 0.02
        dx, k0 = 0.05, 4.0
        u = rng.normal(size=eps.shape) + 1j * rng.normal(size=eps.shape)
        v = rng.normal(size=eps.shape) + 1j * rng.normal(size=eps.shape)
        left = np.vdot(v, te_apply(u, eps, k0, dx))
        right = np.vdot(te_apply(v, np.conjugate(eps), k0, dx), u)
        self.assertLess(abs(left - right) / abs(left), 1e-10)

        rhs = np.zeros_like(eps, dtype=np.complex128)
        rhs[4, 3] = 1.0
        from topology import _cocg

        def te_solve(perm):
            field, ok = _cocg(
                lambda guess: te_apply(guess, perm, k0, dx),
                rhs,
                tol=1e-10,
                maxiter=400,
            )
            self.assertTrue(ok)
            return field

        field = te_solve(eps)
        adjoint, ok = optical_adjoint_field(
            eps, k0, dx, field, tol=1e-10, maxiter=400, apply=te_apply
        )
        self.assertTrue(ok)
        sens = te_permittivity_gradient(field, adjoint, eps, k0, dx)
        i, j = 3, 5
        delta = 1e-6
        plus = float(np.sum(np.abs(te_solve(_bump(eps, i, j, delta))) ** 2))
        minus = float(np.sum(np.abs(te_solve(_bump(eps, i, j, -delta))) ** 2))
        fd = (plus - minus) / (2.0 * delta)
        self.assertLess(_rel(float(sens[i, j]), fd), 1e-4)

    def test_bilinear_sample_matches_its_scatter(self):
        rng = np.random.default_rng(5)
        field = rng.normal(size=(8, 6))
        xs = np.linspace(-1.0, 1.4, 8)
        ys = np.linspace(0.0, 1.0, 6)
        xq = rng.uniform(xs[0], xs[-1], size=(5, 4))
        yq = rng.uniform(ys[0], ys[-1], size=(5, 4))
        sampled = bilinear_sample(xs, ys, field, xq, yq)
        weights = rng.normal(size=sampled.shape)
        scattered = bilinear_scatter(xs, ys, xq, yq, weights, field.shape)
        self.assertLess(
            _rel(float(np.sum(sampled * weights)), float(np.sum(field * scattered))),
            1e-12,
        )

    def test_chip_heat_matches_window_density_finite_difference(self):
        xs_c = np.linspace(-4.0, 4.0, 41)
        ys_c = np.linspace(-3.0, 3.0, 31)
        xs_w = np.linspace(-1.0, 1.0, 15)
        ys_w = np.linspace(-0.8, 0.8, 13)
        rho_fixed = np.zeros((xs_c.size, ys_c.size))
        rho_w = np.full((xs_w.size, ys_w.size), 0.55)
        xx, yy = np.meshgrid(xs_c, ys_c, indexing="ij")
        q = np.where((np.abs(xx + 2.2) < 0.6) & (np.abs(yy) < 0.8), 4.0e7, 0.0)
        ring = (np.abs(xx - 1.6) < 0.5) & (np.abs(yy) < 0.6)
        dldt = 1.0

        def solve(density):
            rho_c, inside, temp_c, temp_w = coupled_window_heat(
                xs_c, ys_c, rho_fixed, q, density, xs_w, ys_w
            )
            t_ring = mean_on_mask(temp_c, ring)
            return rho_c, inside, temp_c, temp_w, dldt * t_ring

        rho_c, inside, temp_c, temp_w, shift = solve(rho_w)
        window_grad = window_heat_gradient(
            xs_c, ys_c, rho_c, inside, temp_c, xs_w, ys_w, 2.0 * temp_w
        )
        ring_grad = window_heat_gradient(
            xs_c, ys_c, rho_c, inside, temp_c, xs_w, ys_w,
            np.zeros_like(temp_w), ring, 2.0 * shift * dldt,
        )
        self.assertGreater(abs(shift), 0.1)

        def window_J(density):
            _rho, _inside, _temp, sampled, _shift = solve(density)
            return float(np.sum(sampled**2))

        def ring_J(density):
            *_, hot_shift = solve(density)
            return float(hot_shift**2)

        i, j = 7, 6
        delta = 1e-3
        fd = (window_J(_bump(rho_w, i, j, delta)) - window_J(_bump(rho_w, i, j, -delta))) / (2.0 * delta)
        self.assertLess(_rel(float(window_grad[i, j]), fd), 0.02)
        ri, rj = np.unravel_index(int(np.argmax(np.abs(ring_grad))), ring_grad.shape)
        delta = 0.01
        fd = (ring_J(_bump(rho_w, ri, rj, delta)) - ring_J(_bump(rho_w, ri, rj, -delta))) / (2.0 * delta)
        self.assertLess(_rel(float(ring_grad[ri, rj]), fd), 0.02)


def _bump(grid, i, j, delta):
    out = np.array(grid, copy=True)
    out[i, j] = out[i, j] + delta
    return out


if __name__ == "__main__":
    unittest.main()
