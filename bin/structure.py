#!/usr/bin/env python3
"""Draw the polycrystal at one thickness (Fig. 8 analogue).

    uv run bin/structure.py --width 12 --thickness 6 --out results/structure.png

Units: lengths in d0 (mean nucleus spacing), time with v = 1, so the film
thickness equals the growth time.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from evosel2d import area_fractions, generate_nuclei, label_field, mean_surface_width, top_surface
from evosel2d.plots import plot_structure


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--width", type=float, default=12.0, help="window width in d0")
    p.add_argument("--thickness", type=float, default=6.0, help="film thickness in d0")
    p.add_argument("--density", type=float, default=1.0, help="nuclei per d0")
    p.add_argument("--px-per-d0", type=float, default=24.0, help="grid resolution")
    p.add_argument("--height-factor", type=float, default=1.6,
                   help="grid height / thickness (a corner-up crystal reaches sqrt(2) x thickness)")
    p.add_argument("--angles", choices=("continuous", "two-class"), default="continuous")
    p.add_argument("--tol", type=float, default=10.0, help="corner/face window in degrees")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("results/structure.png"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    res = args.px_per_d0
    nx = int(round(args.width * res))
    ny = int(round(args.height_factor * args.thickness * res))
    xs = (np.arange(nx) + 0.5) / res
    ys = (np.arange(ny) + 0.5) / res

    rng = np.random.default_rng(args.seed)
    cx, cy, phi = generate_nuclei(args.width, density=args.density, pad=2.0 * args.thickness, rng=rng,
                                  orientation_mode=args.angles)
    best, labels = label_field(xs, ys, cx, cy, phi)

    mask = best <= args.thickness
    frac = area_fractions(mask, labels, phi, args.tol)
    owner, _ = top_surface(mask, labels)
    width = mean_surface_width(owner) / res
    ratio = frac["corner"] / frac["face"] if frac["face"] > 0 else float("inf")

    title = (
        f"thickness = {args.thickness:g} d$_0$, {int(args.width):g} d$_0$ wide, seed {args.seed}"
        f"   corner {frac['corner']:.2f} | face {frac['face']:.2f} | other {frac['other']:.2f}"
        f"   $I_{{11}}/I_{{10}}$ = {ratio:.1f}"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plot_structure(
        labels,
        best,
        phi,
        cx,
        cy,
        args.thickness,
        px_per_d0=res,
        title=title,
        path=str(args.out),
    )
    print(f"wrote {args.out}")
    print(
        f"thickness={args.thickness:g} d0  corner={frac['corner']:.3f}  face={frac['face']:.3f}  "
        f"other={frac['other']:.3f}  <|sin2phi|>={frac['corner_weight']:.3f}  "
        f"I11/I10={ratio:.2f}  mean surface width={width:.2f} d0"
    )


if __name__ == "__main__":
    main()
