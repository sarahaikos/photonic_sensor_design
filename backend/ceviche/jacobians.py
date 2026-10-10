"""Dense Jacobian of a numpy function by forward differences.

Upstream ceviche differentiates through autograd. Here the Maxwell solve is
differentiated with one adjoint system, ``fdfd_ez.grad_eps`` / ``fdfd_hz.grad_eps``.
``jacobian`` itself is the dense numerical Jacobian (one extra solve per input).
``mode`` accepts ``'numerical'``, ``'forward'``, and ``'reverse'``; all three
build that same matrix. Use ``grad_eps`` when the input is a permittivity grid.
"""

from __future__ import annotations

import numpy as np


def jacobian(fun, argnum=0, mode="numerical", step_size=1e-6):
    """Return a function that evaluates the Jacobian of ``fun`` w.r.t. one argument.

    The Jacobian has shape ``(out_size, in_size)`` and is built with one-sided
    differences. ``mode='reverse'`` and ``mode='forward'`` accept the same
    signature; both evaluate this dense numerical Jacobian. Use ``grad_eps``
    on an FDFD object for a single reverse-mode solve against permittivity.
    """
    if mode not in ("numerical", "forward", "reverse"):
        raise ValueError("mode must be 'numerical', 'forward', or 'reverse'")

    def jac(*args):
        values = list(args)
        base_in = np.asarray(values[argnum])
        flat = np.array(base_in, dtype=float, copy=True).ravel()
        base_out = np.asarray(fun(*values)).ravel()
        columns = np.empty((base_out.size, flat.size), dtype=np.result_type(base_out, float))
        for i in range(flat.size):
            bumped = flat.copy()
            bumped[i] += step_size
            values[argnum] = bumped.reshape(base_in.shape)
            columns[:, i] = (np.asarray(fun(*values)).ravel() - base_out) / step_size
        return columns

    return jac
