"""2D FDFD for Ez (TMz) and Hz (TEz) polarizations.
The Ez system is the ceviche discretization of
    ∇ × μ⁻¹ ∇ × E − ω² ε E = i ω J
with Yee differences and a stretched-coordinate PML. Time convention is e^{-iωt}.
    Hx, Hy, Ez = fdfd_ez(omega, dL, eps_r, npml).solve(Jz)
"""

from __future__ import annotations
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from ceviche.constants import EPSILON_0, MU_0
from ceviche.derivatives import compute_derivative_matrices
from ceviche.solvers import solve_linear

class fdfd:
    """Shared grid, PML derivatives, and field packing."""
    def __init__(self, omega, dL, eps_r, npml, bloch_phases=None, iterative=False, method="bicgstab"):
        self.omega = float(omega)
        if self.omega == 0.0:
            raise ValueError("omega must be nonzero")
        self.dL = dL
        self.npml = npml
        self.bloch_x, self.bloch_y = _bloch(bloch_phases)
        self.iterative = iterative
        self.method = method
        self.eps_r = eps_r

    @property
    def eps_r(self):
        return self._eps_r

    @eps_r.setter
    def eps_r(self, new_eps):
        grid = np.asarray(new_eps)
        if grid.ndim != 2:
            raise ValueError("eps_r must be a 2D array of shape (Nx, Ny)")
        shape = (int(grid.shape[0]), int(grid.shape[1]))
        if not hasattr(self, "shape") or shape != getattr(self, "shape", None):
            self.shape = shape
            self.Nx, self.Ny = shape
            self.N = self.Nx * self.Ny
            self._setup_derivatives()
        self._eps_r = np.asarray(grid, dtype=np.complex128)

    def solve(self, source_z):
        raise NotImplementedError

    def system_matrix(self):
        """Sparse Maxwell operator for the current ``eps_r``."""
        raise NotImplementedError

    def _setup_derivatives(self):
        self.Dxf, self.Dxb, self.Dyf, self.Dyb = compute_derivative_matrices(
            self.omega,
            self.shape,
            self.npml,
            self.dL,
            bloch_x=self.bloch_x,
            bloch_y=self.bloch_y,
        )

    def _solve_vec(self, A, b):
        if self.iterative:
            return solve_linear(A, b, iterative=True, method=self.method)
        return solve_linear(A, b, iterative=False)

    def _grid(self, vec):
        return np.asarray(vec).reshape(self.shape)

    def _vec(self, grid):
        array = np.asarray(grid)
        if array.shape != self.shape:
            raise ValueError(f"expected shape {self.shape}, got {array.shape}")
        return array.reshape(self.N)


class fdfd_ez(fdfd):
    """Linear Ez polarization. ``solve`` returns ``(Hx, Hy, Ez)``."""

    def system_matrix(self):
        return _ez_matrix(self.omega, self._vec(self.eps_r), self.Dxf, self.Dxb, self.Dyf, self.Dyb)

    def solve(self, source_z):
        eps_vec = self._vec(self.eps_r)
        jz = self._vec(source_z)
        A = _ez_matrix(self.omega, eps_vec, self.Dxf, self.Dxb, self.Dyf, self.Dyb)
        ez = self._solve_vec(A, 1j * self.omega * jz)
        hx, hy = self._ez_to_h(ez)
        return self._grid(hx), self._grid(hy), self._grid(ez)

    def grad_eps(self, source_z, field_grad):
        """Gradient of a real objective w.r.t. real ``eps_r``.

        ``field_grad`` is the Wirtinger derivative ``∂J/∂Ez*``, same shape as ``Ez``.
        For ``J = sum(|Ez|**2)`` pass ``Ez`` itself. The chain rule uses one adjoint solve:

            dJ = 2 Re(gᴴ dEz),   Aᴴ λ = g,   A = ∇×μ⁻¹∇× − ω²ε₀ diag(ε).
        """
        eps_vec = self._vec(self.eps_r)
        jz = self._vec(source_z)
        g = self._vec(field_grad)
        A = _ez_matrix(self.omega, eps_vec, self.Dxf, self.Dxb, self.Dyf, self.Dyb)
        lu = splu(A.tocsc())
        ez = lu.solve(1j * self.omega * jz)
        lam = lu.solve(np.asarray(g, dtype=np.complex128), trans="H")
        # dA = −ω²ε₀ diag(δ), δ real → dJ = 2 ω² ε₀ Re(conj(λ) * Ez) · δ
        sens = 2.0 * (self.omega**2) * EPSILON_0 * np.real(np.conjugate(lam) * ez)
        return sens.reshape(self.shape)

    def _ez_to_h(self, ez_vec):
        scale = 1.0 / (1j * self.omega * MU_0)
        hx = -scale * (self.Dyb @ ez_vec)
        hy = scale * (self.Dxb @ ez_vec)
        return hx, hy


class fdfd_hz(fdfd):
    """Linear Hz polarization. ``solve`` returns ``(Ex, Ey, Hz)``."""

    def system_matrix(self):
        return _hz_matrix(self.omega, self._averaged_eps(self._vec(self.eps_r)), self.Dxf, self.Dxb, self.Dyf, self.Dyb)

    def solve(self, source_z):
        eps_vec = self._vec(self.eps_r)
        mz = self._vec(source_z)
        eps_xx, eps_yy = self._averaged_eps(eps_vec)
        A = _hz_matrix(self.omega, (eps_xx, eps_yy), self.Dxf, self.Dxb, self.Dyf, self.Dyb)
        hz = self._solve_vec(A, 1j * self.omega * mz)
        ex, ey = self._hz_to_e(hz, eps_xx, eps_yy)
        return self._grid(ex), self._grid(ey), self._grid(hz)

    def grad_eps(self, source_z, field_grad):
        """Gradient of a real objective w.r.t. real ``eps_r``.

        ``field_grad`` is ``∂J/∂Hz*``. For ``J = sum(|Hz|**2)`` pass ``Hz``.
        Permittivity is Yee-averaged onto the Ex and Ey edges before it enters ``A``.
        """
        eps_vec = self._vec(self.eps_r)
        mz = self._vec(source_z)
        g = np.asarray(self._vec(field_grad), dtype=np.complex128)
        eps_xx, eps_yy = self._averaged_eps(eps_vec)
        A = _hz_matrix(self.omega, (eps_xx, eps_yy), self.Dxf, self.Dxb, self.Dyf, self.Dyb)
        lu = splu(A.tocsc())
        hz = lu.solve(1j * self.omega * mz)
        lam = lu.solve(g, trans="H")

        inv_xx = 1.0 / (eps_xx + 1e-5)
        inv_yy = 1.0 / (eps_yy + 1e-5)
        # λᴴ (Dxb diag(δinv_yy) Dxf) Hz = sum conj((Dxbᴴ λ)) * δinv * (Dxf Hz)
        sens_inv_yy = np.conjugate(self.Dxb.getH() @ lam) * (self.Dxf @ hz) / EPSILON_0
        sens_inv_xx = np.conjugate(self.Dyb.getH() @ lam) * (self.Dyf @ hz) / EPSILON_0
        # δinv = −inv² δe_avg, and dJ = −2 Re(sens_inv · δinv)
        g_yy = -2.0 * np.real(sens_inv_yy * (-inv_yy**2))
        g_xx = -2.0 * np.real(sens_inv_xx * (-inv_xx**2))
        g_yy = g_yy.reshape(self.shape)
        g_xx = g_xx.reshape(self.shape)
        # eps_yy = ½(eps + roll(eps, +x)), eps_xx = ½(eps + roll(eps, +y))
        grad = 0.5 * g_yy + 0.5 * np.roll(g_yy, -1, axis=0)
        grad = grad + 0.5 * g_xx + 0.5 * np.roll(g_xx, -1, axis=1)
        return np.real(grad)

    def _averaged_eps(self, eps_vec):
        grid = self._grid(eps_vec)
        eps_xx = 0.5 * (grid + np.roll(grid, shift=1, axis=1))
        eps_yy = 0.5 * (grid + np.roll(grid, shift=1, axis=0))
        return eps_xx.ravel(), eps_yy.ravel()

    def _hz_to_e(self, hz_vec, eps_xx, eps_yy):
        scale = 1.0 / (1j * self.omega * EPSILON_0)
        ex = scale * (self.Dyf @ hz_vec) / (eps_xx + 1e-5)
        ey = -scale * (self.Dxf @ hz_vec) / (eps_yy + 1e-5)
        return ex, ey

def _ez_matrix(omega, eps_vec, dxf, dxb, dyf, dyb):
    curl = -(dxf @ dxb + dyf @ dyb) / MU_0
    diag = -(omega**2) * EPSILON_0 * np.asarray(eps_vec, dtype=np.complex128)
    return (curl + sparse.diags(diag, format="csr")).tocsr()

def _hz_matrix(omega, averaged, dxf, dxb, dyf, dyb):
    eps_xx, eps_yy = averaged
    inv_xx = 1.0 / (np.asarray(eps_xx, dtype=np.complex128) + 1e-5)
    inv_yy = 1.0 / (np.asarray(eps_yy, dtype=np.complex128) + 1e-5)
    curl = (dxb @ sparse.diags(inv_yy) @ dxf + dyb @ sparse.diags(inv_xx) @ dyf) / EPSILON_0
    diag = (MU_0 * omega**2) * np.ones(inv_xx.size, dtype=np.complex128)
    return (curl + sparse.diags(diag, format="csr")).tocsr()

def _bloch(phases):
    if phases is None:
        return 0.0, 0.0
    if np.isscalar(phases):
        return float(phases), 0.0
    if len(phases) == 1:
        return float(phases[0]), 0.0
    return float(phases[0]), float(phases[1])