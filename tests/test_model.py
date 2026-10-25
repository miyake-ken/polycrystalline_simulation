"""Correctness tests for the arrival-time field and texture statistics."""

from __future__ import annotations

import numpy as np
import pytest

from evosel2d import (
    arrival_time,
    generate_nuclei,
    growth_field,
    label_field,
    mean_surface_width,
    orientation_classes,
    top_height,
    top_surface,
)


def test_corner_up_grain_grows_at_sqrt2():
    t = 3.0
    assert top_height(np.pi / 4, t) == pytest.approx(np.sqrt(2.0) * t)
    assert top_height(0.0, t) == pytest.approx(t)


def test_arrival_time_boundary_is_inradius():
    for phi in (0.0, 0.3, np.pi / 4, 1.2):
        t = 2.5
        y = top_height(phi, t)
        assert arrival_time(0.0, y, 0.0, 0.0, phi) == pytest.approx(t, rel=1e-12)
        assert arrival_time(0.0, 1.2 * y, 0.0, 0.0, phi) > t


def test_single_grain_covers_its_square():
    t, phi = 1.0, np.pi / 4
    d = 0.5 * np.sqrt(2.0)
    assert arrival_time(d, d, 0.0, 0.0, phi) == pytest.approx(t)
    assert arrival_time(0.9, 0.9, 0.0, 0.0, phi) > t


def test_partition_is_complete_and_consistent():
    rng = np.random.default_rng(5)
    cx, cy, phi = generate_nuclei(20.0, density=1.0, pad=8.0, rng=rng)
    xs = np.arange(0.0, 20.0, 0.25)
    ys = np.arange(0.0, 4.0, 0.25)
    best, labels = label_field(xs, ys, cx, cy, phi)

    assert np.isfinite(best).all()
    assert labels.min() >= 0 and labels.max() < cx.size

    ref = arrival_time(
        xs[None, :], ys[:, None], cx[:, None, None], cy[:, None, None], phi[:, None, None]
    )
    assert np.allclose(best, ref.min(axis=0), atol=1e-5)
    assert np.array_equal(labels, ref.argmin(axis=0))


def test_padding_makes_the_interior_exact():
    """Once the padding exceeds the max film height, the window is exact."""
    xs = np.arange(0.5, 30.0, 0.5)
    ys = np.arange(0.5, 8.0, 0.5)
    rng = np.random.default_rng(3)
    cx_in = rng.uniform(0.0, 30.0, 30)
    phi_in = rng.uniform(0.0, 0.5 * np.pi, 30)

    results = []
    cx_a = cx_b = phi_a = phi_b = None
    for pad in (16.0, 32.0):
        # nuclei outside the window, with orientation a function of x so the two
        # runs share identical nuclei inside [-16, 46]
        extra_x = np.concatenate([np.arange(-pad, 0.0, 1.0), np.arange(30.0, 30.0 + pad, 1.0)])
        extra_phi = (extra_x * 0.37) % (0.5 * np.pi)
        cx = np.concatenate([cx_in, extra_x])
        phi = np.concatenate([phi_in, extra_phi])
        cy = np.zeros_like(cx)
        best, lab = label_field(xs, ys, cx, cy, phi)
        results.append((best, lab))
        if pad == 16.0:
            cx_a, phi_a = cx, phi
        else:
            cx_b, phi_b = cx, phi

    (best_a, lab_a), (best_b, lab_b) = results
    # label indices shift with the nucleus count, so compare the owners themselves
    assert np.allclose(best_a, best_b, atol=1e-6)
    assert np.allclose(cx_a[lab_a], cx_b[lab_b])
    assert np.allclose(phi_a[lab_a], phi_b[lab_b])


def test_orientation_classes_are_symmetric():
    phi = np.radians([0.0, 45.0, 90.0, 30.0])
    cls = orientation_classes(phi, tol_deg=10.0)
    assert list(cls["corner"]) == [False, True, False, False]
    assert list(cls["face"]) == [True, False, True, False]
    assert list(cls["other"]) == [False, False, False, True]


def test_top_surface_picks_highest_pixel_owner():
    mask = np.zeros((4, 5), dtype=bool)
    mask[0, :] = True
    mask[1, :3] = True
    labels = np.zeros((4, 5), dtype=np.int32)
    labels[0, :] = 7
    labels[1, :3] = 9
    owner, height = top_surface(mask, labels)
    assert list(owner) == [9, 9, 9, 7, 7]
    assert list(height) == [2, 2, 2, 1, 1]
    assert mean_surface_width(owner) == pytest.approx(5 / 2)


def test_selection_grows_with_thickness():
    """Corner-up crystals take over as the film thickens (paper's Fig. 9 trend)."""
    width, tmax, res = 200.0, 20.0, 1.0
    xs = (np.arange(int(width * res)) + 0.5) / res
    ys = (np.arange(int(tmax * res)) + 0.5) / res
    rng = np.random.default_rng(42)
    cx, cy, phi = generate_nuclei(width, density=1.0, pad=2.0 * tmax + 2.0, rng=rng)
    best, labels = label_field(xs, ys, cx, cy, phi)
    cls = orientation_classes(phi, tol_deg=10.0)

    def corner_share(t: float) -> float:
        counts = np.bincount(labels[best <= t], minlength=phi.size)
        c, f = counts[cls["corner"]].sum(), counts[cls["face"]].sum()
        return float(c / max(c + f, 1))

    share_1 = corner_share(1.0)
    share_20 = corner_share(20.0)
    assert 0.2 < share_1 < 0.8  # no selection yet: close to the 1:1 window ratio
    assert share_20 > share_1
    assert share_20 > 0.6

    owner_1, _ = top_surface(best <= 1.0, labels)
    owner_20, _ = top_surface(best <= 20.0, labels)
    assert mean_surface_width(owner_20) > 1.5 * mean_surface_width(owner_1)


def test_connected_growth_matches_the_closed_form_for_one_grain():
    xs = (np.arange(160) + 0.5) / 8.0
    ys = (np.arange(160) + 0.5) / 8.0
    cx, cy, phi = np.array([10.0]), np.array([0.0]), np.array([0.4])
    analytic, _ = label_field(xs, ys, cx, cy, phi)
    grown, _ = growth_field(xs, ys, cx, cy, phi, t_max=12.0)
    # the flood seeds the nucleus pixel at t = 0, so skip its one-pixel neighbourhood
    compared = np.isfinite(grown) & (analytic >= 0.5)
    assert compared.sum() > 1000
    assert np.max(np.abs(grown[compared] - analytic[compared])) < 1e-3


def test_connected_growth_leaves_no_floating_crystal():
    """Each crystal must stay connected to its nucleus (no island above a neighbour)."""
    width, res, t = 10.0, 8.0, 6.0
    xs = (np.arange(int(width * res)) + 0.5) / res
    ys = (np.arange(int(1.6 * t * res)) + 0.5) / res
    cx, cy, phi = generate_nuclei(width, 1.0, 2.0 * t, np.random.default_rng(2))
    time, labels = growth_field(xs, ys, cx, cy, phi, t_max=t)
    film = time <= t
    assert film.any()

    def n_pieces(mask: np.ndarray) -> int:
        seen = np.zeros_like(mask, dtype=bool)
        pieces = 0
        for j0, i0 in np.argwhere(mask):
            if seen[j0, i0]:
                continue
            pieces += 1
            stack = [(j0, i0)]
            seen[j0, i0] = True
            while stack:
                j, i = stack.pop()
                for dj in (-1, 0, 1):
                    for di in (-1, 0, 1):
                        jj, ii = j + dj, i + di
                        if 0 <= jj < mask.shape[0] and 0 <= ii < mask.shape[1]:
                            if mask[jj, ii] and not seen[jj, ii]:
                                seen[jj, ii] = True
                                stack.append((jj, ii))
        return pieces

    for label in np.unique(labels[film]):
        region = (labels == label) & film
        if region.sum() > 20:
            assert n_pieces(region) == 1, f"crystal {label} is split into islands"
