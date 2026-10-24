"""Nucleation-site generation for the 2-D evolutionary-selection model.

The paper (Wild, Herres & Koidl 1990, Sec. IV) used square nuclei with random
orientation and random position, infinite surface diffusion, and a growth rate
perpendicular to each face that is constant and independent of orientation.

Square orientation has period pi/2, so orientations are drawn from [0, pi/2).
"""

from __future__ import annotations

import numpy as np


def generate_nuclei(
    width: float,
    density: float = 1.0,
    pad: float = 0.0,
    rng: np.random.Generator | None = None,
    substrate_y: float = 0.0,
    orientation_mode: str = "continuous",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (x, y, phi_rad) for nuclei on a substrate line.

    Parameters
    ----------
    width
        Width of the window of interest, in units of d0 (mean nucleus spacing).
    density
        Nuclei per unit length; ``d0 = 1 / density``.
    pad
        Extra length generated on both sides. Points outside the window still
        compete for territory inside it, so ``pad`` must be at least the
        maximum film thickness for the interior to be exact.
    rng
        Random generator; ``None`` means a fresh default generator.
    orientation_mode
        ``"continuous"`` draws phi uniformly in [0, pi/2) (the literal reading of
        "square nuclei with random orientation").  ``"two-class"`` draws only
        corner-up (pi/4) or face-up (0) squares, which is a stricter idealisation
        useful for testing how much selection the model can produce.
    """
    if width <= 0:
        raise ValueError("width must be positive")
    if density <= 0:
        raise ValueError("density must be positive")
    if pad < 0:
        raise ValueError("pad must be non-negative")

    rng = np.random.default_rng() if rng is None else rng
    total = width + 2.0 * pad
    n = int(round(total * density))
    x = rng.uniform(-pad, width + pad, n)
    if orientation_mode == "continuous":
        phi = rng.uniform(0.0, 0.5 * np.pi, n)
    elif orientation_mode == "two-class":
        phi = np.where(rng.random(n) < 0.5, 0.0, 0.25 * np.pi)
    else:
        raise ValueError(f"unknown orientation_mode: {orientation_mode!r}")
    y = np.full(n, float(substrate_y))
    return x, y, phi


def spacing(density: float) -> float:
    """Mean nucleus spacing d0 for a given linear density."""
    return 1.0 / density
