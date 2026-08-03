#!/usr/bin/env python
"""JX-3D -- 3D organoid reconstruction from a brightfield Z-stack.

Examples:
    ./.venv/bin/python run.py BK52_WT_9805_B/4x_00009
    ./.venv/bin/python run.py BK52_WT_9805_B/4x_00009 --mode edf --open
"""
from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

from jx3d.config import Params
from jx3d.pipeline import run


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="3D organoid reconstruction from a brightfield Z-stack "
                    "(shape-from-focus + per-slice segmentation)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("folder", help="Z-stack folder containing *_Z###_*.tif")
    p.add_argument("-o", "--out", default=None,
                   help="output folder (default: output/<folder name>)")

    g = p.add_argument_group("segmentation")
    g.add_argument("--mode", choices=["both", "edf", "slices"], default="both",
                   help="edf = single pass on the all-in-focus projection (best "
                        "recall, fast); slices = per-slice + Z linking (separates "
                        "organoids stacked in Z); both = union of the two")
    g.add_argument("--detector", choices=["cellpose", "classical"], default="cellpose",
                   help="cellpose = Cellpose-SAM (GPU); classical = edge+watershed")
    g.add_argument("--z-step", type=int, default=1,
                   help="segment every Nth slice (2 = twice as fast)")
    g.add_argument("--diameter", type=float, default=150.0,
                   help="expected organoid diameter (µm)")
    g.add_argument("--min-diameter", type=float, default=30.0, help="µm")
    g.add_argument("--max-diameter", type=float, default=600.0, help="µm")
    g.add_argument("--min-circularity", type=float, default=0.55)
    g.add_argument("--no-gpu", action="store_true")

    g2 = p.add_argument_group("reconstruction")
    g2.add_argument("--min-sharpness", type=float, default=0.25,
                    help="focus-peak prominence threshold (0-1); lowering it "
                         "returns more, but less trustworthy, objects")
    g2.add_argument("--axial-ratio", type=float, default=1.0,
                    help="rz/rxy -- 1.0 assumes spherical")
    g2.add_argument("--min-track-slices", type=int, default=2)

    g3 = p.add_argument_group("output")
    g3.add_argument("--quality", type=int, default=72, help="viewer JPEG quality")
    g3.add_argument("--no-viewer", action="store_true", help="skip the HTML viewer")
    g3.add_argument("--no-cache", action="store_true",
                    help="ignore the segmentation cache")
    g3.add_argument("--open", action="store_true", help="open the viewer in a browser when done")

    a = p.parse_args(argv)

    folder = Path(a.folder)
    if not folder.is_dir():
        print(f"No such folder: {folder}", file=sys.stderr)
        return 1
    outdir = Path(a.out) if a.out else Path("output") / folder.name

    params = Params(
        mode=a.mode,
        detector=a.detector,
        z_step=max(1, a.z_step),
        expected_diameter_um=a.diameter,
        min_diameter_um=a.min_diameter,
        max_diameter_um=a.max_diameter,
        min_circularity=a.min_circularity,
        min_track_slices=a.min_track_slices,
        axial_ratio=a.axial_ratio,
    )

    res = run(folder, outdir, params=params, gpu=not a.no_gpu,
              use_cache=not a.no_cache, jpeg_quality=a.quality,
              min_sharpness=a.min_sharpness, build_html=not a.no_viewer)

    html = res.outdir / "viewer.html"
    if a.open and html.exists():
        webbrowser.open(html.resolve().as_uri())
    elif html.exists():
        print(f"\nViewer:  {html.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
