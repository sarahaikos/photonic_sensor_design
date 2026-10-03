"""Exact thermo-optic sensitivities for desensitization topology optimization.

The optical adjoint field solves Aᴴ λ = ∂J/∂E*. TM uses the scalar Helmholtz
operator and TE uses ∇ · ε⁻¹ ∇ + k0². Both are complex symmetric, so Aᴴ is
the same apply with conjugated permittivity. When permittivity depends on the
heater temperature, ∂J/∂T drives the adjoint BOX-sink heat equation.
"""

from __future__ import annotations

import numpy as np

from thermo import adjoint_heat, bicgstab_field, dsheet_k_drho, heat_kappa_gradient, sheet_k


def helmholtz_apply(field: np.ndarray, eps: np.ndarray, k0: float, dx_um: float) -> np.ndarray:
    """(∇² + k0² ε) E with zero Dirichlet edges. Δx is in μm, matching the TO grid."""
    f = np.asarray(field, dtype=np.complex128)
    pad = np.pad(f, 1, mode="constant")
    lap = pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:] - 4.0 * pad[1:-1, 1:-1]
    dx = float(dx_um)
    return lap / (dx * dx) + (float(k0) ** 2) * np.asarray(eps, dtype=np.complex128) * f


def solve_helmholtz(
    eps: np.ndarray,
    k0: float,
    dx_um: float,
    rhs: np.ndarray,
    tol: float = 1e-8,
    maxiter: int = 500,
):
    """Solve (∇² + k0² ε) E = rhs. Returns ``(E, converged)``."""
    field, ok = bicgstab_field(
        lambda guess: helmholtz_apply(guess, eps, k0, dx_um),
        np.asarray(rhs, dtype=np.complex128),
        tol=tol,
        maxiter=maxiter,
    )
    return field, bool(ok and np.isfinite(field).all())


def optical_adjoint_field(
    eps: np.ndarray,
    k0: float,
    dx_um: float,
    field_grad: np.ndarray,
    tol: float = 1e-6,
    maxiter: int = 500,
    solve=None,
    apply=None,
):
    """Adjoint field λ with Aᴴ λ = field_grad.

    Both the TM Helmholtz operator and the TE operator are complex symmetric,
    so Aᴴ is the same apply with conjugated ε. ``field_grad`` is ∂J/∂E* for a
    real objective, the g in dJ = 2 Re(gᴴ dE). ``solve(op, rhs)`` may replace
    the default BiCGSTAB and must return ``(λ, ok)``.
    """
    eps_h = np.conjugate(np.asarray(eps, dtype=np.complex128))
    rhs = np.asarray(field_grad, dtype=np.complex128)
    op_apply = helmholtz_apply if apply is None else apply

    def op(field):
        return op_apply(field, eps_h, k0, dx_um)

    if solve is None:
        lam, ok = bicgstab_field(op, rhs, tol=tol, maxiter=maxiter)
        lam = np.asarray(lam, dtype=np.complex128)
        return lam, bool(ok and np.isfinite(lam).all())
    lam, ok = solve(op, rhs)
    lam = np.asarray(lam, dtype=np.complex128)
    return lam, bool(ok and np.isfinite(lam).all())


def permittivity_gradient(e_fwd: np.ndarray, adjoint: np.ndarray, k0: float, loss: float = 0.0) -> np.ndarray:
    """∂J/∂ε_material for the TM Helmholtz operator.

    A real material change δ enters the complex permittivity as (1 + i loss) δ.
    ``loss=0`` is a real perturbation of the permittivity already stored in A.
    """
    weight = 1.0 + 1j * float(loss)
    return -2.0 * (float(k0) ** 2) * np.real(np.conjugate(adjoint) * weight * np.asarray(e_fwd))


def te_apply(field: np.ndarray, eps: np.ndarray, k0: float, dx_um: float) -> np.ndarray:
    """(∇ · ε⁻¹ ∇ + k0²) Hz with zero Dirichlet edges.

    Face conductance is the harmonic mean of the two neighboring permittivities.
    A missing neighbor is a zero field and reuses the boundary permittivity.
    """
    f = np.asarray(field, dtype=np.complex128)
    inv = 1.0 / (np.asarray(eps, dtype=np.complex128) + 1e-6)
    dx2 = float(dx_um) ** 2
    fp = np.pad(f, 1, mode="constant")
    ip = np.pad(inv, 1, mode="edge")
    ic = ip[1:-1, 1:-1]
    acc = np.zeros_like(f)
    for inv_n, f_n in (
        (ip[:-2, 1:-1], fp[:-2, 1:-1]),
        (ip[2:, 1:-1], fp[2:, 1:-1]),
        (ip[1:-1, :-2], fp[1:-1, :-2]),
        (ip[1:-1, 2:], fp[1:-1, 2:]),
    ):
        acc += (2.0 / (ic + inv_n)) * (f_n - f)
    return acc / dx2 + (float(k0) ** 2) * f


def te_permittivity_gradient(
    field: np.ndarray,
    adjoint: np.ndarray,
    eps: np.ndarray,
    k0: float,
    dx_um: float,
    loss: float = 0.0,
) -> np.ndarray:
    """∂J/∂ε_material for the TE operator. Same loss chain as the TM gradient."""
    f = np.asarray(field, dtype=np.complex128)
    lam = np.asarray(adjoint, dtype=np.complex128)
    inv = 1.0 / (np.asarray(eps, dtype=np.complex128) + 1e-6)
    dx2 = float(dx_um) ** 2
    chain = 1.0 + 1j * float(loss)
    dot = np.zeros(f.shape, dtype=np.complex128)

    def face(f_a, f_b, lam_a, lam_b, inv_a, inv_b):
        g = 2.0 / (inv_a + inv_b)
        s = (f_b - f_a) / dx2
        return (np.conjugate(lam_a) - np.conjugate(lam_b)) * s * (g**2 / 2.0) * chain

    factor = face(f[:-1, :], f[1:, :], lam[:-1, :], lam[1:, :], inv[:-1, :], inv[1:, :])
    dot[:-1, :] += factor * (inv[:-1, :] ** 2)
    dot[1:, :] += factor * (inv[1:, :] ** 2)
    factor = face(f[:, :-1], f[:, 1:], lam[:, :-1], lam[:, 1:], inv[:, :-1], inv[:, 1:])
    dot[:, :-1] += factor * (inv[:, :-1] ** 2)
    dot[:, 1:] += factor * (inv[:, 1:] ** 2)

    zero_f = np.zeros(1, dtype=np.complex128)
    zero_l = zero_f
    for index, sign in ((0, 1), (-1, -1)):
        inv_b = inv[index, :]
        factor_b = face(
            zero_f if sign > 0 else f[index, :],
            f[index, :] if sign > 0 else zero_f,
            zero_l if sign > 0 else lam[index, :],
            lam[index, :] if sign > 0 else zero_l,
            inv_b,
            inv_b,
        )
        dot[index, :] += factor_b * (2.0 * inv_b**2)
        inv_b = inv[:, index]
        factor_b = face(
            zero_f if sign > 0 else f[:, index],
            f[:, index] if sign > 0 else zero_f,
            zero_l if sign > 0 else lam[:, index],
            lam[:, index] if sign > 0 else zero_l,
            inv_b,
            inv_b,
        )
        dot[:, index] += factor_b * (2.0 * inv_b**2)
    return -2.0 * np.real(dot)


def port_field_grad(
    e_fwd: np.ndarray,
    through: np.ndarray,
    drop: np.ndarray,
    jmon: int,
    target: float,
    dJ_dt: float | None = None,
) -> np.ndarray:
    """∂J/∂E* for a power split on one monitor column.

    T_drop = P_drop / (P_through + P_drop). Pass ∂J/∂T_drop, or omit it to use
    J = (T_drop − target)².
    """
    p_th = float(np.sum(np.abs(e_fwd[through, jmon]) ** 2))
    p_dr = float(np.sum(np.abs(e_fwd[drop, jmon]) ** 2))
    total = p_th + p_dr + 1e-30
    if dJ_dt is None:
        dJ_dt = 2.0 * (p_dr / total - float(target))
    grad = np.zeros_like(e_fwd)
    grad[drop, jmon] += float(dJ_dt) * (p_th / total**2) * e_fwd[drop, jmon]
    grad[through, jmon] += float(dJ_dt) * (-p_dr / total**2) * e_fwd[through, jmon]
    return grad


def permittivity_derivatives(
    rho: np.ndarray,
    temp: np.ndarray | None,
    n_core: float,
    n_bg: float,
    dndt_core: float,
    dndt_clad: float,
):
    """∂ε/∂ρ and ∂ε/∂T for ε = (1−ρ) n_clad(T)² + ρ n_core(T)².

    ∂ε/∂T is None when ``temp`` is None (heater off, indices frozen).
    """
    density = np.clip(np.asarray(rho, dtype=float), 0.0, 1.0)
    if temp is None:
        n_c = float(n_core)
        n_b = float(n_bg)
        d_eps_dt = None
    else:
        t = np.asarray(temp, dtype=float)
        n_c = float(n_core) + float(dndt_core) * t
        n_b = float(n_bg) + float(dndt_clad) * t
        d_eps_dt = 2.0 * (1.0 - density) * n_b * float(dndt_clad) + 2.0 * density * n_c * float(dndt_core)
    d_eps_drho = np.where(
        (np.asarray(rho, dtype=float) >= 0.0) & (np.asarray(rho, dtype=float) <= 1.0),
        n_c**2 - n_b**2,
        0.0,
    )
    return d_eps_drho, d_eps_dt


def thermo_optic_terms(
    dJ_deps: np.ndarray,
    rho_eps: np.ndarray,
    temp: np.ndarray | None,
    rho_kappa: np.ndarray,
    xs: np.ndarray,
    n_core: float,
    n_bg: float,
    dndt_core: float,
    dndt_clad: float,
):
    """Split ∂J/∂ρ into the direct ε(ρ) piece and the adjoint-heat piece.

    ``rho_eps`` is the density inside the permittivity.
    ``rho_kappa`` is the density inside the sheet conductivity. For a projected
    blueprint these are the same array, and ``temp`` is the heat solution for
    ``sheet_k(rho_kappa)``.
    Returns ``(direct, thermal)`` with those two shapes.
    """
    d_rho, d_t = permittivity_derivatives(rho_eps, temp, n_core, n_bg, dndt_core, dndt_clad)
    direct = np.asarray(dJ_deps, dtype=float) * d_rho
    thermal = np.zeros(np.shape(rho_kappa), dtype=float)
    if temp is None or d_t is None:
        return direct, thermal
    dJ_dT = np.asarray(dJ_deps, dtype=float) * d_t
    lam = adjoint_heat(xs, sheet_k(rho_kappa), dJ_dT)
    thermal = heat_kappa_gradient(lam, temp, xs) * dsheet_k_drho(rho_kappa)
    return direct, thermal
