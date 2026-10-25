"""Cubic kinetic Wulff geometry and specimen-Z inverse-pole-figure colours.

R maps crystal coordinates to specimen coordinates. Length and time units are
arbitrary; V111=1 unless specified. No material parameters are fitted to images.
"""
from itertools import combinations, product

import numpy as np
from scipy.spatial.transform import Rotation


def facets(alpha: float, v111: float = 1.0):
    if alpha <= 0 or v111 <= 0:
        raise ValueError("alpha and v111 must be positive")
    normals = np.concatenate((np.eye(3), -np.eye(3),
                              np.array(list(product((-1., 1.), repeat=3))) / np.sqrt(3)))
    speeds = np.r_[np.full(6, alpha * v111 / np.sqrt(3)), np.full(8, v111)]
    return normals, speeds


def wulff_vertices(alpha: float, v111: float = 1.0):
    """Vertices of intersection n_f.x <= V_f (unit-time isolated crystal)."""
    normals, speeds = facets(alpha, v111)
    vertices = []
    for indices in combinations(range(14), 3):
        a = normals[list(indices)]
        if abs(np.linalg.det(a)) < 1e-10:
            continue
        v = np.linalg.solve(a, speeds[list(indices)])
        if np.all(normals @ v <= speeds + 1e-10):
            vertices.append(v)
    return np.unique(np.round(vertices, 12), axis=0)


def ipf_z(rotations):
    """Explicit cubic RGB convention, not a claim of vendor/MTEX colour parity.

    Fold R.T @ Z to 0<=a<=b<=c. RGB before gamma/normalization is
    (c-b, sqrt(2)*(b-a), sqrt(3)*a): 001 red, 011 green, 111 blue.
    Different grain IDs can have identical IPF colours.
    """
    abc = np.sort(np.abs(np.asarray(rotations)[..., 2, :]), axis=-1)
    a, b, c = np.moveaxis(abc, -1, 0)
    rgb = np.stack((c-b, np.sqrt(2)*(b-a), np.sqrt(3)*a), axis=-1)
    rgb = np.sqrt(np.maximum(rgb, 0))
    return rgb / np.maximum(rgb.max(axis=-1, keepdims=True), 1e-15)


def theta_001(rotations):
    """Angle of specimen Z to the nearest symmetry-equivalent <001> axis."""
    return np.degrees(np.arccos(np.clip(np.abs(rotations[:, 2, :]).max(axis=1), 0, 1)))


def initial_nuclei(width, count, seed, min_separation):
    """Uniform SO(3) orientations, periodic hard-core substrate seed positions.

    The exclusion avoids overlapping finite seeds; it is an explicit departure
    from unconstrained independent point positions. No nuclei are silently lost.
    """
    rng = np.random.default_rng(seed)
    xy = []
    for _ in range(100000):
        p = rng.uniform(0, width, 2)
        if xy:
            delta = np.abs(np.array(xy) - p)
            delta = np.minimum(delta, width - delta)
            if np.any(np.linalg.norm(delta, axis=1) < min_separation):
                continue
        xy.append(p)
        if len(xy) == count:
            break
    if len(xy) != count:
        raise ValueError("cannot pack requested seeds; reduce count/radius")
    return np.c_[np.array(xy), np.zeros(count)], Rotation.random(count, random_state=rng).as_matrix()
