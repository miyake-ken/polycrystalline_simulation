"""Rendering of the simulated microstructure and of the texture-versus-thickness scan."""

from __future__ import annotations

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import hsv_to_rgb  # noqa: E402

__all__ = ["structure_rgb", "plot_structure", "plot_scan"]


def structure_rgb(labels: np.ndarray, mask: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Colour the film by orientation; uncovered space is white."""
    hue = (np.asarray(phi, dtype=float) % (0.5 * np.pi)) / (0.5 * np.pi) * 0.83
    sat = np.full_like(hue, 0.62)
    val = np.full_like(hue, 0.92)
    rgb = hsv_to_rgb(np.stack([hue[labels], sat[labels], val[labels]], axis=-1))
    rgb[~mask] = 1.0
    return rgb


def plot_structure(
    labels: np.ndarray,
    best: np.ndarray,
    phi: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    thickness: float,
    *,
    px_per_d0: float,
    title: str = "",
    path: str | None = None,
):
    """Draw the polycrystal at a given thickness (Fig. 8 analogue)."""
    mask = best <= float(thickness)
    rgb = structure_rgb(labels, mask, phi)
    ny, nx = mask.shape
    extent = (0.0, nx / px_per_d0, 0.0, ny / px_per_d0)

    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=160)
    ax.imshow(rgb, origin="lower", extent=extent, interpolation="nearest")
    inside = (cx >= 0) & (cx <= extent[1])
    ax.plot(cx[inside], cy[inside], "s", ms=1.2, color="black", alpha=0.75, lw=0)
    heights = ny - 1 - np.argmax(mask[::-1], axis=0)
    top = (heights.max() + 1) / px_per_d0
    ax.set_xlim(0, extent[1])
    ax.set_ylim(0, max(top, 1.0))
    ax.set_xlabel("x (d$_0$)")
    ax.set_ylabel("height (d$_0$)")
    ax.set_title(title or f"thickness = {thickness:g} d$_0$")
    ax.set_aspect("equal")
    fig.tight_layout()
    if path:
        fig.savefig(path)
    return fig


def plot_scan(rows: dict[str, np.ndarray], params: dict, path: str | None = None):
    """Draw the thickness scan: orientation ratio, fractions and grain width."""
    t = rows["thickness_d0"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), dpi=160)

    ax = axes[0]
    ax.plot(t, rows["ratio_11_10"], "o-", ms=3, color="C3")
    ax.set_yscale("log")
    ax.set_xlabel("thickness (d$_0$)")
    ax.set_ylabel("$I_{11}/I_{10}$ (area ratio)")
    ax.set_title("evolutionary selection")
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    ax.plot(t, rows["corner"], "o-", ms=3, label="corner-up (11)")
    ax.plot(t, rows["face"], "s-", ms=3, label="face-up (10)")
    ax.plot(t, rows["other"], "^-", ms=3, label="other")
    ax.plot(t, rows["corner_weight"], "k--", lw=1, label=r"$\langle|\sin 2\phi|\rangle$")
    ax.set_xlabel("thickness (d$_0$)")
    ax.set_ylabel("area fraction")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(t, rows["surface_width_d0"], "o-", ms=3, color="C0")
    ax.set_xlabel("thickness (d$_0$)")
    ax.set_ylabel("mean surface crystal width (d$_0$)")
    ax.set_title("surviving crystals widen")
    ax.grid(alpha=0.3)

    fig.suptitle(
        "2-D evolutionary selection, Wild/Herres/Koidl (1990) model — "
        f"{params['width']:g} d$_0$ wide, {params['seeds']} seed(s), tol = {params['tol_deg']:g}°",
        fontsize=9,
    )
    fig.tight_layout()
    if path:
        fig.savefig(path)
    return fig
