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
from jx3d.stack import DEFAULT_HINT, default_stack, discover_stacks


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="3D organoid reconstruction from a brightfield Z-stack "
                    "(shape-from-focus + per-slice segmentation)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("folder", nargs="?", default=None,
                   help=f"Z-stack folder containing *_Z###_*.tif. Omit it and the "
                        f"nearest {DEFAULT_HINT} is used, or the first stack found.")
    p.add_argument("--menu", action="store_true",
                   help="list the stacks found here and pick one interactively")
    p.add_argument("--list", action="store_true",
                   help="list the stacks found here and exit")
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
    g.add_argument("--diameter", type=float, default=40.0,
                   help="expected organoid diameter, in PIXELS")
    g.add_argument("--min-diameter", type=float, default=8.0, help="pixels")
    g.add_argument("--max-diameter", type=float, default=160.0, help="pixels")
    g.add_argument("--min-circularity", type=float, default=0.55)
    g.add_argument("--no-gpu", action="store_true")

    gc = p.add_argument_group(
        "calibration",
        "Sizes are measured in pixels and slices. Micrometre columns are only "
        "written when a real scale is known -- from the Keyence .gci, or from "
        "these options. Nothing is ever converted with a guessed scale.")
    gc.add_argument("--px-size", type=float, default=None,
                    help="lateral scale, µm per pixel (overrides the .gci)")
    gc.add_argument("--z-step-um", type=float, default=None,
                    help="axial scale, µm per slice (overrides the .gci)")
    gc.add_argument("--anisotropy", type=float, default=None,
                    help="slice spacing / pixel spacing, when the absolute "
                         "scale is unknown but the ratio is not")

    g2 = p.add_argument_group("reconstruction")
    g2.add_argument("--min-sharpness", type=float, default=0.25,
                    help="focus-peak prominence threshold (0-1); lowering it "
                         "returns more, but less trustworthy, objects")
    g2.add_argument("--axial-ratio", type=float, default=1.0,
                    help="rz/rxy -- 1.0 assumes spherical")
    g2.add_argument("--min-track-slices", type=int, default=2)
    g2.add_argument("--no-dome", action="store_true",
                    help="skip fitting the Matrigel dome surface")

    g3 = p.add_argument_group("output")
    g3.add_argument("--quality", type=int, default=72, help="viewer JPEG quality")
    g3.add_argument("--no-viewer", action="store_true", help="skip the HTML viewer")
    g3.add_argument("--no-cache", action="store_true",
                    help="ignore the segmentation cache")
    g3.add_argument("--open", action="store_true", help="open the viewer in a browser when done")

    a = p.parse_args(argv)

    if a.list or a.menu:
        stacks = discover_stacks(".")
        if not stacks:
            print("No Z-stacks found below the current folder.", file=sys.stderr)
            return 1
        for i, st in enumerate(stacks, 1):
            n = len([q for q in st.glob("*.tif")])
            mark = "  <- default" if DEFAULT_HINT in st.name else ""
            print(f"  {i:2d}  {st}  ({n} slices){mark}")
        if a.list:
            return 0
        try:
            choice = input(f"\nstack [1-{len(stacks)}, Enter = default]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 1
        if not choice:
            folder = default_stack(".") or stacks[0]
        elif choice.isdigit() and 1 <= int(choice) <= len(stacks):
            folder = stacks[int(choice) - 1]
        else:
            print(f"Not a valid choice: {choice}", file=sys.stderr)
            return 1
    elif a.folder:
        folder = Path(a.folder)
    else:
        found = default_stack(".")
        if found is None:
            print("No Z-stack found below the current folder. Pass one "
                  "explicitly, or run serve.py to browse for it.", file=sys.stderr)
            return 1
        folder = found
        print(f"No folder given; using {folder}")

    if not folder.is_dir():
        print(f"No such folder: {folder}", file=sys.stderr)
        return 1
    outdir = Path(a.out) if a.out else Path("output") / folder.name

    params = Params(
        mode=a.mode,
        detector=a.detector,
        z_step=max(1, a.z_step),
        expected_diameter_px=a.diameter,
        min_diameter_px=a.min_diameter,
        max_diameter_px=a.max_diameter,
        min_circularity=a.min_circularity,
        min_track_slices=a.min_track_slices,
        axial_ratio=a.axial_ratio,
        fit_dome=not a.no_dome,
    )

    overrides = {}
    if a.px_size is not None:
        overrides["px_um"] = a.px_size
    if a.z_step_um is not None:
        overrides["z_um"] = a.z_step_um
    if a.anisotropy is not None:
        overrides["assumed_anisotropy"] = a.anisotropy

    res = run(folder, outdir, params=params, gpu=not a.no_gpu,
              calibration=overrides,
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
