#!/usr/bin/env python3
"""Thickness scan: texture ratio, orientation fractions and crystal width (Fig. 9 analogue).

    uv run bin/thickness_scan.py --width 600 --tmax 25 --seeds 3

The arrival-time partition is computed once per seed and then sliced at every
thickness, so the cost does not grow with the number of thicknesses.
"""

from __future__ import annotations

import argparse
import json
import csv
from pathlib import Path

import numpy as np

from evosel2d import generate_nuclei, label_field, scan_thicknesses
from evosel2d.plots import plot_scan

FIELDS = [
    "thickness_d0",
    "corner",
    "face",
    "other",
    "corner_weight",
    "ratio_11_10",
    "surface_width_d0",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--width", type=float, default=600.0, help="window width in d0")
    p.add_argument("--tmax", type=float, default=25.0, help="largest thickness in d0")
    p.add_argument("--thicknesses", type=float, nargs="*", default=None,
                   help="explicit thickness list (default: 1..tmax)")
    p.add_argument("--density", type=float, default=1.0, help="nuclei per d0")
    p.add_argument("--px-per-d0", type=float, default=1.5, help="grid resolution")
    p.add_argument("--height-factor", type=float, default=1.6,
                   help="grid height / tmax (a corner-up crystal reaches sqrt(2) x thickness)")
    p.add_argument("--angles", choices=("continuous", "two-class"), default="continuous")
    p.add_argument("--tol", type=float, default=10.0, help="corner/face window in degrees")
    p.add_argument("--seeds", type=int, default=3, help="independent realisations to average")
    p.add_argument("--seed0", type=int, default=0, help="first seed")
    p.add_argument("--out-prefix", type=Path, default=Path("results/scan"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    res = args.px_per_d0
    nx = int(round(args.width * res))
    ny = int(round(args.height_factor * args.tmax * res))
    xs = (np.arange(nx) + 0.5) / res
    ys = (np.arange(ny) + 0.5) / res
    thicknesses = (
        np.array(args.thicknesses, dtype=float)
        if args.thicknesses
        else np.arange(1.0, args.tmax + 1e-9, 1.0)
    )

    runs: list[list[dict[str, float]]] = []
    for seed in range(args.seed0, args.seed0 + args.seeds):
        rng = np.random.default_rng(seed)
        cx, cy, phi = generate_nuclei(args.width, density=args.density, pad=2.0 * args.tmax + 2.0, rng=rng,
                                  orientation_mode=args.angles)
        best, labels = label_field(xs, ys, cx, cy, phi)
        runs.append(
            scan_thicknesses(best, labels, phi, thicknesses, tol_deg=args.tol, pixel_size=1.0 / res)
        )
        print(f"seed {seed}: {cx.size} nuclei, {labels.shape[1]}x{labels.shape[0]} grid")

    # average area fractions across seeds, then form the ratio (more stable than
    # averaging ratios when the face fraction is tiny)
    rows: dict[str, np.ndarray] = {}
    for key in FIELDS:
        rows[key] = np.array([np.mean([r[i][key] for r in runs]) for i in range(len(thicknesses))])
    rows["ratio_11_10"] = np.where(rows["face"] > 0, rows["corner"] / rows["face"], np.inf)
    spread = {
        "corner_std": np.array([np.std([r[i]["corner"] for r in runs]) for i in range(len(thicknesses))]),
        "ratio_std": np.array([np.std([r[i]["ratio_11_10"] for r in runs]) for i in range(len(thicknesses))]),
    }

    params = {
        "width": args.width,
        "tmax": args.tmax,
        "density": args.density,
        "px_per_d0": res,
        "tol_deg": args.tol,
        "seeds": args.seeds,
        "seed0": args.seed0,
        "model": "Wild, Herres & Koidl, J. Appl. Phys. 68, 973 (1990), Sec. IV (2-D, v=1)",
        "angles": args.angles,
        "height_factor": args.height_factor,
    }

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_prefix.with_suffix(".csv")
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(FIELDS + ["corner_std", "ratio_std"])
        for i, t in enumerate(rows["thickness_d0"]):
            w.writerow([f"{rows[k][i]:.6g}" for k in FIELDS] + [
                f"{spread['corner_std'][i]:.6g}", f"{spread['ratio_std'][i]:.6g}"
            ])

    png_path = args.out_prefix.with_suffix(".png")
    plot_scan(rows, params, path=str(png_path))
    summary = {
        "params": params,
        "at": {f"{t:g}": {"corner": float(rows["corner"][i]), "face": float(rows["face"][i]),
                          "ratio_11_10": float(rows["ratio_11_10"][i]),
                          "surface_width_d0": float(rows["surface_width_d0"][i])}
               for i, t in enumerate(rows["thickness_d0"])
               if t in (1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0)},
    }
    json_path = args.out_prefix.with_suffix(".json")
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print()
    print(f"{'d/d0':>6} {'corner':>7} {'face':>7} {'other':>7} {'I11/I10':>9} {'width':>8}")
    for i, t in enumerate(rows["thickness_d0"]):
        print(
            f"{t:6.1f} {rows['corner'][i]:7.3f} {rows['face'][i]:7.3f} {rows['other'][i]:7.3f} "
            f"{rows['ratio_11_10'][i]:9.2f} {rows['surface_width_d0'][i]:8.2f}"
        )
    print()
    print(f"wrote {csv_path}\nwrote {png_path}\nwrote {json_path}")


if __name__ == "__main__":
    main()
