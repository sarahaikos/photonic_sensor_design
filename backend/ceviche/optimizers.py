"""Adam updates for inverse design. Bounds are applied by clipping after the step."""

from __future__ import annotations

import numpy as np


def adam_optimize(
    objective,
    params,
    jac,
    step_size=1e-2,
    Nsteps=100,
    bounds=None,
    direction="min",
    beta1=0.9,
    beta2=0.999,
    callback=None,
    verbose=False,
):
    """Minimize ``objective`` for ``Nsteps`` steps.

    If ``jac is True``, ``objective(params)`` must return ``(value, gradient)``.
    Otherwise ``jac(params)`` supplies the gradient.
    """
    if direction not in ("min", "max"):
        raise ValueError("direction must be 'min' or 'max'")
    params = np.array(params, dtype=float, copy=True)
    history = []
    mopt = None
    vopt = None
    sign = -1.0 if direction == "min" else 1.0
    for iteration in range(int(Nsteps)):
        if callback is not None:
            callback(iteration, history, params)
        if jac is True:
            value, grad = objective(params)
        else:
            value = objective(params)
            grad = jac(params)
        grad = np.asarray(grad, dtype=float)
        history.append(float(np.real(value)))
        if verbose:
            print(f"Epoch: {iteration + 1:3d}/{int(Nsteps):3d} | Value: {history[-1]:.6e}")
        if iteration == 0:
            mopt = np.zeros(grad.shape)
            vopt = np.zeros(grad.shape)
        grad_adam, mopt, vopt = step_adam(grad, mopt, vopt, iteration, beta1, beta2)
        params = params + sign * step_size * grad_adam
        if bounds is not None:
            params = np.clip(params, bounds[0], bounds[1])
    return params, history


def step_adam(gradient, mopt_old, vopt_old, iteration, beta1, beta2, epsilon=1e-8):
    """One Adam moment update. Returns ``(step, m, v)``."""
    grad = np.asarray(gradient, dtype=float)
    mopt = beta1 * mopt_old + (1.0 - beta1) * grad
    vopt = beta2 * vopt_old + (1.0 - beta2) * np.square(grad)
    mhat = mopt / (1.0 - beta1 ** (iteration + 1))
    vhat = vopt / (1.0 - beta2 ** (iteration + 1))
    return mhat / (np.sqrt(vhat) + epsilon), mopt, vopt
