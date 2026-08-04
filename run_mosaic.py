#!/usr/bin/env python
"""JX-3D -- reconstruct a whole Matrigel dome from a tiled brightfield scan.

    ./.venv/bin/python run_mosaic.py BK52_WT_9805_B

A tiled acquisition is a grid of overlapping Z-stacks covering one specimen.
This assembles them into a single frame, fits the droplet across all of them,
merges the detections that appear in more than one tile, and writes one feature
matrix with one row per organoid.

For a single field of view, use run.py instead; that path is unchanged.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from jx3d.config import Params
from jx3d.mosaic_pipeline import run


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Whole-dome reconstruction from a tiled brightfield scan",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("dataset", nargs="?", default="BK52_WT_9805_B",
                   help="folder holding the tile subfolders and the .gci")
    p.add_argument("-o", "--out", default=None,
                   help="output folder (default: output/<dataset name>_mosaic)")

    g = p.add_argument_group("segmentation")
    g.add_argument("--mode", choices=["both", "edf", "slices"], default="edf",
                   help="edf = one pass on each tile's all-in-focus projection; "
                        "slices = per-slice plus Z linking, fifteen times more "
                        "segmenter calls; both = the union")
    g.add_argument("--detector", choices=["cellpose", "classical"],
                   default="cellpose")
    g.add_argument("--diameter", type=float, default=40.0, help="pixels")
    g.add_argument("--min-diameter", type=float, default=8.0, help="pixels")
    g.add_argument("--max-diameter", type=float, default=160.0, help="pixels")
    g.add_argument("--min-sharpness", type=float, default=0.25,
                   help="focus-peak prominence threshold (0-1)")
    g.add_argument("--no-gpu", action="store_true")

    gc = p.add_argument_group(
        "calibration",
        "Measurements are in pixels and slices. Micrometre columns appear only "
        "when a real scale is known, from the .gci or from these options.")
    gc.add_argument("--px-size", type=float, default=None, help="µm per pixel")
    gc.add_argument("--z-step-um", type=float, default=None, help="µm per slice")

    g3 = p.add_argument_group("output")
    g3.add_argument("--no-cache", action="store_true",
                    help="re-measure every tile instead of reusing its result")

    a = p.parse_args(argv)

    dataset = Path(a.dataset)
    if not dataset.is_dir():
        print(f"no such dataset: {dataset}", file=sys.stderr)
        return 1
    outdir = Path(a.out) if a.out else Path("output") / f"{dataset.name}_mosaic"

    params = Params(mode=a.mode, detector=a.detector,
                    expected_diameter_px=a.diameter,
                    min_diameter_px=a.min_diameter,
                    max_diameter_px=a.max_diameter,
                    fit_dome=False)

    overrides = {}
    if a.px_size is not None:
        overrides["px_um"] = a.px_size
    if a.z_step_um is not None:
        overrides["z_um"] = a.z_step_um

    result = run(dataset, outdir, params=params, gpu=not a.no_gpu,
                 use_cache=not a.no_cache, min_sharpness=a.min_sharpness,
                 calibration=overrides)
    print(f"\nFeature matrix:  {(result.outdir / 'features.csv').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
