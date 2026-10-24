"""Texture statistics of the simulated polycrystal."""

from __future__ import annotations

import numpy as np

__all__ = [
    "orientation_classes",
    "corner_weight",
    "top_surface",
    "area_fractions",
    "mean_surface_width",
    "scan_thicknesses",
]


def corner_weight(phi: np.ndarray) -> np.ndarray:
    """Continuous "corner-upness" of a square, 1 for corner up, 0 for face up."""
    return np.abs(np.sin(2.0 * np.asarray(phi, dtype=float)))


def orientation_classes(phi: np.ndarray, tol_deg: float = 10.0) -> dict[str, np.ndarray]:
    """Split orientations into corner-up, face-up and other.

    Square symmetry makes phi periodic in 90 deg.  ``corner`` means a corner
    points up (what the paper calls an 11-oriented crystal), ``face`` means a
    face is parallel to the substrate (10-oriented).
    """
    p = np.degrees(np.asarray(phi, dtype=float)) % 90.0
    dist_corner = np.abs(p - 45.0)
    dist_face = np.minimum(p, 90.0 - p)
    corner = dist_corner <= tol_deg
    face = dist_face <= tol_deg
    return {"corner": corner, "face": face, "other": ~(corner | face)}


def top_surface(mask: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Owning nucleus and height of the topmost covered pixel of each column.

    Returns ``(owner, height)`` with ``owner = -1`` for empty columns and
    ``height`` in pixels.
    """
    ny, nx = mask.shape
    filled = mask.any(axis=0)
    from_top = np.argmax(mask[::-1], axis=0)
    rows = ny - 1 - from_top
    cols = np.arange(nx)
    owner = np.where(filled, labels[rows, cols], -1).astype(np.int32)
    height = np.where(filled, rows + 1, 0).astype(np.int32)
    return owner, height


def area_fractions(
    mask: np.ndarray, labels: np.ndarray, phi: np.ndarray, tol_deg: float = 10.0
) -> dict[str, float]:
    """Area fraction of the film owned by each orientation class."""
    counts = np.bincount(labels[mask], minlength=phi.size).astype(float)
    total = counts.sum()
    if total == 0:
        return {"corner": np.nan, "face": np.nan, "other": np.nan, "corner_weight": np.nan}
    cls = orientation_classes(phi, tol_deg)
    out = {
        "corner": float(counts[cls["corner"]].sum() / total),
        "face": float(counts[cls["face"]].sum() / total),
        "other": float(counts[cls["other"]].sum() / total),
        "corner_weight": float((counts * corner_weight(phi)).sum() / total),
    }
    return out


def mean_surface_width(owner: np.ndarray) -> float:
    """Mean width (in pixels) of the crystals reaching the top surface."""
    live = owner[owner >= 0]
    if live.size == 0:
        return float("nan")
    breaks = np.count_nonzero(np.diff(live)) + 1
    return float(live.size / breaks)


def scan_thicknesses(
    best: np.ndarray,
    labels: np.ndarray,
    phi: np.ndarray,
    thicknesses: np.ndarray,
    *,
    tol_deg: float = 10.0,
    pixel_size: float = 1.0,
) -> list[dict[str, float]]:
    """Texture statistics of the same simulation sliced at several thicknesses."""
    rows: list[dict[str, float]] = []
    for t in thicknesses:
        mask = best <= float(t)
        frac = area_fractions(mask, labels, phi, tol_deg)
        owner, _ = top_surface(mask, labels)
        ratio = frac["corner"] / frac["face"] if frac["face"] > 0 else float("inf")
        rows.append(
            {
                "thickness_d0": float(t),
                "corner": frac["corner"],
                "face": frac["face"],
                "other": frac["other"],
                "corner_weight": frac["corner_weight"],
                "ratio_11_10": ratio,
                "surface_width_d0": mean_surface_width(owner) * pixel_size,
            }
        )
    return rows
