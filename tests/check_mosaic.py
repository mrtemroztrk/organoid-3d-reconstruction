#!/usr/bin/env python
"""Check that the tile grid describes the specimen that was actually imaged.

    ./.venv/bin/python tests/check_mosaic.py BK52_WT_9805_B

A mosaic can be self-consistent and still be wrong. If the camera is mounted so
that the stage runs the opposite way to the image, every tile lands on the
correct side of a perfectly regular grid that happens to be mirrored, and no
amount of checking the metadata against itself will notice. The layout in the
group file and the stage log in the frames come from the same instrument and
share the same convention, so they cannot catch each other out either.

So the last few checks here leave the metadata behind and ask the pixels. Two
tiles that overlap show the same specimen twice; if the grid places them
correctly their overlapping strips correlate, and if it has them mirrored the
correlation collapses. That is a test the metadata cannot pass by agreeing with
itself.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jx3d import blend as blendmod
from jx3d import mosaic as mosaicmod
from jx3d.keyence import read_group_metadata, read_tile_layout

_RESULTS: list[tuple[bool, str]] = []


def ok(passed: bool, message: str) -> bool:
    _RESULTS.append((bool(passed), message))
    print(("PASS  " if passed else "FAIL  ") + message)
    return bool(passed)


def _mid_slice(folder: Path) -> np.ndarray:
    import tifffile as tiff

    files = sorted(folder.glob("*.tif"))
    return tiff.imread(files[len(files) // 2]).astype(np.float32)


def _correlation_peak(a: np.ndarray, b: np.ndarray) -> float:
    """Sharpness of the phase-correlation peak between two equal-sized patches.

    Normalising away the magnitude leaves only the phase agreement, so the value
    reflects how well the structures line up rather than how bright either patch
    happens to be -- which matters here, because illumination differs measurably
    from one tile to the next.
    """
    a = a - a.mean()
    b = b - b.mean()
    window = np.hanning(a.shape[0])[:, None] * np.hanning(a.shape[1])[None, :]
    fa = np.fft.rfft2(a * window)
    fb = np.fft.rfft2(b * window)
    cross = fa * np.conj(fb)
    cross /= np.abs(cross) + 1e-12
    return float(np.fft.irfft2(cross, a.shape).max())


def _seam_step(image: np.ndarray, m) -> float:
    """Mean brightness step across the vertical tile boundaries in a composite.

    A seam is a discontinuity, so it shows up as a jump between the column just
    inside a tile's edge and the column just outside it. Comparing that jump to
    the jump one would find at an arbitrary column is what separates a real seam
    from ordinary specimen contrast.
    """
    steps = []
    for t in m:
        for x in (int(round(t.x0)), int(round(t.x0 + t.width))):
            if 2 <= x < image.shape[1] - 2:
                a = image[:, x - 2:x].mean(axis=1).astype(np.float32)
                b = image[:, x:x + 2].mean(axis=1).astype(np.float32)
                steps.append(np.abs(a - b).mean())
    return float(np.mean(steps)) if steps else 0.0


def run(dataset: Path) -> int:
    acq, _ = read_group_metadata(dataset)
    layout, info = read_tile_layout(dataset)

    if not ok(layout is not None, "the acquisition declares a tile layout"):
        return 1
    ok(layout.n_rows * layout.n_cols == len(layout.cells),
       f"{layout.n_rows} x {layout.n_cols} grid names "
       f"{len(layout.cells)} stacks, one per cell")
    ok(sorted(layout.cells.values()) ==
       sorted((r, c) for r in range(layout.n_rows) for c in range(layout.n_cols)),
       "every cell of the grid is filled exactly once")

    m = mosaicmod.build(dataset, acq)
    if not ok(m is not None, "the grid assembles into a mosaic"):
        return 1
    ok(m.offset_source == mosaicmod.STAGE,
       f"tile offsets come from the stage log, not an assumption "
       f"({m.offset_source})")
    ok(all(t.stage_nm is not None for t in m),
       "every tile carries the stage position it was captured at")

    # --- the grid is regular, and the overlap is a sane fraction of a frame ---
    ox, oy = m.overlap_px()
    ok(0.05 * m.tile_width < ox < 0.60 * m.tile_width,
       f"horizontal overlap {ox:.1f} px = {100 * ox / m.tile_width:.1f}% of a frame")
    ok(0.05 * m.tile_height < oy < 0.60 * m.tile_height,
       f"vertical overlap {oy:.1f} px = {100 * oy / m.tile_height:.1f}% of a frame")

    dx = [b.x0 - a.x0 for a, b, ax in m.neighbour_pairs() if ax == "x"]
    dy = [b.y0 - a.y0 for a, b, ax in m.neighbour_pairs() if ax == "y"]
    ok(float(np.std(dx)) < 1.0 and float(np.std(dy)) < 1.0,
       f"the stage stepped regularly (spread {np.std(dx):.2f} px across columns, "
       f"{np.std(dy):.2f} px across rows)")

    depths = {t.stage_nm[2] for t in m}
    ok(len(depths) == 1,
       f"all tiles share one stage Z, so a slice index means one depth "
       f"({len(depths)} distinct value(s))")

    # --- the canvas covers every tile, with no gap between neighbours ---
    coverage = np.zeros((m.height, m.width), dtype=np.uint8)
    for t in m:
        x0, y0 = int(round(t.x0)), int(round(t.y0))
        coverage[y0:y0 + t.height, x0:x0 + t.width] += 1
    hull = coverage[int(round(min(t.y0 for t in m))):
                    int(round(max(t.y0 + t.height for t in m))),
                    int(round(min(t.x0 for t in m))):
                    int(round(max(t.x0 + t.width for t in m)))]
    ok((hull == 0).sum() == 0,
       f"the tiles tile: {(hull == 0).sum()} uncovered pixels inside the mosaic")
    ok(coverage.max() == 4,
       f"corners are seen by up to {coverage.max()} tiles, as a 2D overlap implies")

    # --- and now the part the metadata cannot fake ---
    print("\n  correlating the overlapping strips against the placement:")
    strips_ok = 0
    strips_total = 0
    for a, b, axis in m.neighbour_pairs():
        ia, ib = _mid_slice(a.folder), _mid_slice(b.folder)
        if axis == "x":
            w = int(round(a.width - (b.x0 - a.x0)))
            placed = _correlation_peak(ia[:, -w:], ib[:, :w])
            mirrored = _correlation_peak(ia[:, :w], ib[:, -w:])
        else:
            h = int(round(a.height - (b.y0 - a.y0)))
            placed = _correlation_peak(ia[-h:, :], ib[:h, :])
            mirrored = _correlation_peak(ia[:h, :], ib[-h:, :])
        strips_total += 1
        good = placed > 3.0 * mirrored
        strips_ok += good
        print(f"    {'PASS' if good else 'FAIL'}  {a.index:>3d}-{b.index:<3d} {axis}  "
              f"as placed {placed:.4f}   mirrored {mirrored:.4f}   "
              f"x{placed / max(mirrored, 1e-9):.0f}")
    ok(strips_ok == strips_total,
       f"{strips_ok}/{strips_total} neighbour pairs correlate as the grid places "
       f"them, and not as their mirror image")

    # --- flat-fielding removes the fixed pattern, and blending hides the seams ---
    print("\n  illumination and seams:")
    flat = blendmod.estimate_flat_field(m)
    ok(flat.span_after < 0.25 * flat.span_before,
       f"flat-fielding flattens the illumination from {flat.span_before:.1f} to "
       f"{flat.span_after:.1f} grey levels across a frame")
    ok(0.5 < flat.gain.min() and flat.gain.max() < 2.0,
       f"the correction stays gentle (gain {flat.gain.min():.2f} to "
       f"{flat.gain.max():.2f}), so it cannot invent contrast")

    z = min(89, blendmod.MosaicSlices(m, flat).depth - 1)
    plain = blendmod.MosaicSlices(m, None, blend=False).slice(z)
    fused = blendmod.MosaicSlices(m, flat, blend=True).slice(z)
    ok(_seam_step(fused, m) < _seam_step(plain, m),
       f"blending softens the seams: mean step across a tile edge falls from "
       f"{_seam_step(plain, m):.2f} to {_seam_step(fused, m):.2f} grey levels")

    failed = sum(1 for good, _ in _RESULTS if not good)
    print(f"\n{len(_RESULTS) - failed} passed, {failed} failed")
    if not failed:
        print("\n" + m.describe(acq))
    return 1 if failed else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Check a tiled acquisition's geometry")
    ap.add_argument("dataset", nargs="?", default="BK52_WT_9805_B")
    a = ap.parse_args(argv)
    d = Path(a.dataset)
    if not d.is_dir():
        print(f"no such dataset: {d}", file=sys.stderr)
        return 2
    return run(d)


if __name__ == "__main__":
    raise SystemExit(main())
