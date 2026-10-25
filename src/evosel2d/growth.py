"""Growth simulation with the connectivity constraint that a crystal cannot reappear.

``metric.label_field`` solves the *unconstrained* first-arrival problem: each
point goes to the nucleus with the smallest arrival time.  For anisotropic
growth that answer is not physical.  A crystal grows only by advancing its own
surface, so its material must be connected to the nucleus: once a neighbour has
covered a point, the buried crystal can never reach anything behind it, and it
must not show up again further away.

The figure in the paper (Fig. 8) shows connected columnar crystals, so the model
has to be simulated rather than evaluated as a closed-form partition.

Implementation: a monotone flood (Dijkstra) driven by the exact arrival time of
each grain.  A pixel may be claimed by grain ``g`` at time ``tau_g`` only if a
neighbouring pixel is already owned by ``g`` and was claimed no later, i.e. the
front propagates from the nucleus through material that grain already owns.
With a single grain this reproduces ``metric.label_field`` exactly; with many
grains it forbids the disconnected fragments that the plain argmin produces.
"""

from __future__ import annotations

import heapq

import numpy as np

from evosel2d.metric import label_field

__all__ = ["growth_field"]

_NEIGHBOURS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def growth_field(
    xs: np.ndarray,
    ys: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    phi: np.ndarray,
    *,
    method: str = "connected",
    t_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(time, labels)`` for the film grown from the given nuclei.

    ``time[j, i]`` is the arrival time of the winning crystal at the grid point
    and ``labels[j, i]`` its index.  Points that no crystal reaches (or that are
    not reached before ``t_max``) keep ``inf`` and ``-1``.

    ``method="analytic"`` gives the unconstrained first-arrival partition; it is
    kept for cross-checks and single-grain tests only.
    """
    if method == "analytic":
        best, labels = label_field(xs, ys, cx, cy, phi)
        if t_max is not None:
            labels = np.where(best <= t_max, labels, -1)
        return best, labels
    if method != "connected":
        raise ValueError(f"unknown method: {method!r}")
    return _connected_field(xs, ys, cx, cy, phi, t_max)


def _connected_field(
    xs: np.ndarray,
    ys: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    phi: np.ndarray,
    t_max: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    nx, ny = xs.size, ys.size
    ng = np.asarray(cx).size
    if ng == 0:
        raise ValueError("no nuclei supplied")

    time = np.full((ny, nx), np.inf, dtype=np.float32)
    labels = np.full((ny, nx), -1, dtype=np.int32)

    # plain Python sequences make the inner loop several times faster
    cxl = np.asarray(cx, dtype=float).tolist()
    cyl = np.asarray(cy, dtype=float).tolist()
    ccl = np.cos(phi).astype(float).tolist()
    ssl = np.sin(phi).astype(float).tolist()
    xsl = xs.tolist()
    ysl = ys.tolist()
    limit = np.inf if t_max is None else float(t_max)

    heap: list[tuple[float, int, int]] = []

    def push(g: int, j: int, i: int) -> None:
        dx = xsl[i] - cxl[g]
        dy = ysl[j] - cyl[g]
        u = ccl[g] * dx + ssl[g] * dy
        w = ccl[g] * dy - ssl[g] * dx
        u = u if u >= 0.0 else -u
        w = w if w >= 0.0 else -w
        tau = u if u > w else w
        if tau <= limit:
            heapq.heappush(heap, (tau, g, j * nx + i))

    # every nucleus seeds the pixel it falls into
    for g in range(ng):
        i = int(np.clip(np.searchsorted(xs, cxl[g]), 0, nx - 1))
        j = int(np.clip(np.searchsorted(ys, cyl[g]), 0, ny - 1))
        if labels[j, i] != -1:
            continue
        labels[j, i] = g
        time[j, i] = 0.0
        for dj, di in _NEIGHBOURS:
            jj, ii = j + dj, i + di
            if 0 <= jj < ny and 0 <= ii < nx and labels[jj, ii] == -1:
                push(g, jj, ii)

    flat_time = time.reshape(-1)
    flat_lab = labels.reshape(-1)
    pop = heapq.heappop
    while heap:
        tau, g, p = pop(heap)
        if flat_lab[p] != -1:
            continue
        j, i = divmod(p, nx)
        # the front must arrive here from material this grain already owns
        reached = False
        for dj, di in _NEIGHBOURS:
            jj, ii = j + dj, i + di
            if 0 <= jj < ny and 0 <= ii < nx:
                q = jj * nx + ii
                if flat_lab[q] == g and flat_time[q] <= tau:
                    reached = True
                    break
        if not reached:
            continue
        flat_lab[p] = g
        flat_time[p] = tau
        for dj, di in _NEIGHBOURS:
            jj, ii = j + dj, i + di
            if 0 <= jj < ny and 0 <= ii < nx and flat_lab[jj * nx + ii] == -1:
                push(g, jj, ii)

    return time, labels
