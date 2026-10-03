"""SI constants used by the FDFD and FDTD solvers.

Same values as [ceviche](https://github.com/fancompute/ceviche/) (Hughes et al., ACS Photonics 2019).
"""

from __future__ import annotations

import numpy as np

EPSILON_0 = 8.85418782e-12
MU_0 = 1.25663706e-6
C_0 = 1.0 / np.sqrt(EPSILON_0 * MU_0)
ETA_0 = np.sqrt(MU_0 / EPSILON_0)
Q_e = 1.602176634e-19
Q_E = Q_e