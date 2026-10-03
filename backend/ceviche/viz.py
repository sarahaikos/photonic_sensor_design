"""Field plots. Matplotlib is imported when a plot is requested."""

from __future__ import annotations

import numpy as np


def real(val, outline=None, ax=None, cbar=False, cmap="RdBu", outline_alpha=0.5):
    """Plot ``Re(val)`` with a symmetric color scale. ``val`` is shaped ``(Nx, Ny)``."""
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(1, 1, constrained_layout=True)
    shown = np.real(np.asarray(val).T)
    limit = np.max(np.abs(shown))
    image = ax.imshow(shown, cmap=cmap, origin="lower", vmin=-limit, vmax=limit)
    if outline is not None:
        ax.contour(np.asarray(outline).T, levels=[0], colors="k", alpha=outline_alpha)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if cbar:
        plt.colorbar(image, ax=ax)
    return ax


def abs(val, outline=None, ax=None, cbar=False, cmap="magma", outline_alpha=0.5, outline_val=None):
    """Plot ``|val|``. An outline is the contour of ``outline`` at its midpoint, or at ``outline_val``."""
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(1, 1, constrained_layout=True)
    shown = np.abs(np.asarray(val).T)
    image = ax.imshow(shown, cmap=cmap, origin="lower", vmin=0, vmax=np.max(shown))
    if outline is not None:
        level = outline_val
        if level is None:
            data = np.asarray(outline)
            level = 0.5 * (data.min() + data.max())
        ax.contour(np.asarray(outline).T, levels=[level], colors="w", alpha=outline_alpha)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if cbar:
        plt.colorbar(image, ax=ax)
    return ax


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("viz.real / viz.abs need matplotlib") from exc
    return plt
