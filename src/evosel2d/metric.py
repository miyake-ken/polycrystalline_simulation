"""Arrival-time field and orientation partition for the 1990 model.

Model (Wild, Herres & Koidl, J. Appl. Phys. 68, 973 (1990), Sec. IV):

* nuclei are squares with random position and random orientation on a line,
* every face advances at the same normal velocity ``v`` (infinite surface
  diffusion), independent of orientation,
* a grain is overgrown where a neighbour reaches the point first.

With those assumptions a grain grown for a time ``t`` from a point nucleus at
``c`` with orientation ``phi`` occupies

    {x : max(|u|, |w|) <= v t},   (u, w) = R(-phi) (x - c)

i.e. a square (in the grain frame) of inradius ``v t``.  The polygon boundary is
exactly the L-infinity sphere in the rotated frame, so the arrival time is

    tau_i(x) = max(|u_i|, |w_i|) / v

and the polycrystal at time ``t`` is the set of points with ``min_i tau_i <= t``.
No time stepping is needed: the partition is an anisotropic Voronoi diagram and
is computed once, then sliced at any thickness.
"""

from __future__ import annotations

import numpy as np

__all__ = ["arrival_time", "label_field", "top_height"]


def arrival_time(
    x: np.ndarray | float,
    y: np.ndarray | float,
    cx: np.ndarray | float,
    cy: np.ndarray | float,
    phi: np.ndarray | float,
) -> np.ndarray:
    """Arrival time (with v = 1) of each nucleus at the points (x, y).

    All arguments broadcast against each other, so passing scalars for the
    point and arrays for the nuclei returns one arrival time per nucleus.
    """
    dx = np.asarray(x, dtype=float) - np.asarray(cx, dtype=float)
    dy = np.asarray(y, dtype=float) - np.asarray(cy, dtype=float)
    c = np.cos(phi)
    s = np.sin(phi)
    u = c * dx + s * dy
    w = -s * dx + c * dy
    return np.maximum(np.abs(u), np.abs(w))


def top_height(phi: float, t: float) -> float:
    """Height of the topmost point of a single grain after time ``t``.

    A corner-up grain (``phi = pi/4``) gains height at ``sqrt(2) v``, a
    face-up grain (``phi = 0``) at ``v``.
    """
    c, s = abs(np.cos(phi)), abs(np.sin(phi))
    return t / max(c, s)


def label_field(
    xs: np.ndarray,
    ys: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    phi: np.ndarray,
    *,
    max_work_bytes: int = 256 << 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-arrival partition on the grid ``xs`` x ``ys``.

    Returns ``(best, labels)`` where ``best[j, i]`` is the arrival time of the
    winning nucleus at ``(xs[i], ys[j])`` and ``labels[j, i]`` is its index.
    The film at thickness ``d`` (with ``v = 1``) is ``best <= d``.

    Nuclei outside the window must be supplied by the caller (see
    :func:`evosel2d.nuclei.generate_nuclei`'s ``pad``).
    """
    xs = np.asarray(xs, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)
    cx = np.asarray(cx, dtype=np.float32)
    cy = np.asarray(cy, dtype=np.float32)
    phi = np.asarray(phi, dtype=np.float32)

    nx, ny, ng = xs.size, ys.size, cx.size
    if nx == 0 or ny == 0:
        raise ValueError("empty grid")
    if ng == 0:
        raise ValueError("no nuclei supplied")

    c = np.cos(phi)
    s = np.sin(phi)
    best = np.empty((ny, nx), np.float32)
    labels = np.empty((ny, nx), np.int32)
    cols = np.arange(nx)

    # three temporaries of nx * m float32 elements are alive at once
    m = int(max(1, min(ng, max_work_bytes // (16 * nx))))

    for j in range(ny):
        y = ys[j]
        row_best = np.full(nx, np.inf, np.float32)
        row_lab = np.zeros(nx, np.int32)
        for g0 in range(0, ng, m):
            g1 = min(ng, g0 + m)
            dx = xs[None, :] - cx[g0:g1, None]
            dy = (y - cy[g0:g1])[:, None]
            cc = c[g0:g1, None]
            ss = s[g0:g1, None]
            u = cc * dx + ss * dy
            w = cc * dy - ss * dx
            np.abs(u, out=u)
            np.abs(w, out=w)
            g = np.maximum(u, w, out=u)
            k = g.argmin(axis=0)
            v = g[k, cols]
            upd = v < row_best
            row_best[upd] = v[upd]
            row_lab[upd] = k[upd] + g0
        best[j] = row_best
        labels[j] = row_lab

    return best, labels
