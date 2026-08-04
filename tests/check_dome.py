#!/usr/bin/env python
"""Check the Matrigel dome fitted across the whole mosaic.

    ./.venv/bin/python tests/check_dome.py BK52_WT_9805_B

A fitted surface is easy to believe and hard to check, so the checks here are
chosen to be ones a wrong fit would fail. Three of them are worth naming.

The droplet's cross-section must *widen* with depth, monotonically, because that
is what a spherical cap is; a fit that has locked onto something else -- the
well wall, a band of debris, the frame edge -- has no reason to produce a
monotone sequence and generally does not.

Each slice is an independent measurement of the droplet's axis, so the spread of
the per-slice centres is a real error bar rather than a bootstrap of one fit
resampling its own assumptions.

And the measured rim has to sit where the picture says the interface is. The
gel/medium boundary leaves a ridge of texture, so the radius the fit returns
should coincide with a maximum of the raw radial texture profile. That check
goes back to the pixels and cannot be satisfied by a self-consistent fit to the
wrong thing.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jx3d import dome_global as dg
from jx3d.blend import MosaicSlices, estimate_flat_field
from jx3d.focus import tenengrad
from jx3d.keyence import read_edge_points, read_group_metadata
from jx3d.mosaic import build
from jx3d.register import register

_RESULTS: list[tuple[bool, str]] = []


def ok(passed: bool, message: str) -> bool:
    _RESULTS.append((bool(passed), message))
    print(("PASS  " if passed else "FAIL  ") + message)
    return bool(passed)


def _texture_ridge_radius(slices, mosaic, centre, z: int, r_max: float,
                          sigma: float = 28.0) -> float:
    """Radius of peak texture on one slice, measured straight from the pixels."""
    import cv2

    cx, cy = centre
    az = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    rr = np.arange(0.0, r_max, 4.0, dtype=np.float32)
    gx = (cx + rr[None, :] * np.cos(az)[:, None]).astype(np.float32)
    gy = (cy + rr[None, :] * np.sin(az)[:, None]).astype(np.float32)

    total = np.zeros((az.size, rr.size), dtype=np.float32)
    count = np.zeros_like(total)
    for tile in mosaic:
        t = np.log1p(cv2.GaussianBlur(tenengrad(slices.tile_slice(tile, z)),
                                      (0, 0), sigma))
        sx, sy = gx - np.float32(tile.x0), gy - np.float32(tile.y0)
        inside = ((sx >= 0) & (sx < tile.width - 1) &
                  (sy >= 0) & (sy < tile.height - 1))
        total += np.where(inside, cv2.remap(t, sx, sy, cv2.INTER_LINEAR,
                                            borderMode=cv2.BORDER_CONSTANT,
                                            borderValue=0.0), 0.0)
        count += inside
    profile = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    return float(rr[int(np.argmax(np.median(profile, axis=0)))])


def run(dataset: Path) -> int:
    acq, _ = read_group_metadata(dataset)
    m = build(dataset, acq)
    flat = estimate_flat_field(m)
    tiles = MosaicSlices(m, flat, blend=False)
    m, _ = register(m, tiles)
    tiles = MosaicSlices(m, flat, blend=False)

    # --- the glass is one plane, and the tiles that cannot see it are excluded ---
    print("  the glass plane:")
    sub = dg.find_substrate(m, tiles)
    ok(len(sub.votes) >= 0.6 * len(m),
       f"{len(sub.votes)} of {len(m)} tiles saw the glass clearly enough to vote")
    ok(len(sub.rejected) > 0 or len(sub.votes) == len(m),
       f"{len(sub.rejected)} tile(s) held back for want of contrast: "
       + (", ".join(sorted(sub.rejected)) or "none"))
    ok(sub.spread_slices <= 10.0,
       f"the voting tiles agree to {sub.spread_slices:.1f} slices "
       f"({sub.spread_slices * (acq.z_um or 1):.0f} um)")
    for name in sub.rejected:
        ok(sub.contrast[name] < min(sub.contrast[v] for v in sub.votes),
           f"{name}'s focus peak ({sub.contrast[name]:.2f}) really is flatter than "
           f"every accepted tile's")

    # --- the fit ---
    print("\n  the cap:")
    fit = dg.fit(m, tiles, sub, acq.anisotropy)
    if not ok(fit.dome is not None, "a cap was fitted"):
        return 1
    d = fit.dome
    used = [r for r in fit.rings if r.used]
    ok(len(used) >= 8, f"{len(used)} of {len(fit.rings)} slices produced a usable ring")

    radii = np.array([r.radius_px for r in used])
    zs = np.array([r.z for r in used])
    ok(bool(np.all(np.diff(radii) > 0)),
       f"the cross-section widens with depth at every step, "
       f"{radii[0]:.0f} px at Z{zs[0] + 1:03d} to {radii[-1]:.0f} px at Z{zs[-1] + 1:03d}")

    ok(max(fit.axis_scatter_px) < 8.0,
       f"the {len(used)} slices agree on the axis to "
       f"({fit.axis_scatter_px[0]:.1f}, {fit.axis_scatter_px[1]:.1f}) px")
    ok(fit.cap_residual_px < 0.02 * d.radius_px,
       f"the cap explains the measured radii to {fit.cap_residual_px:.2f} px rms "
       f"= {100 * fit.cap_residual_px / d.radius_px:.2f}% of the sphere radius")

    # --- against the pixels, not against itself ---
    print("\n  the rim, checked against the raw texture ridge:")
    r_max = 0.62 * float(np.hypot(m.width, m.height))
    agree = []
    for ring in used[::max(1, len(used) // 4)]:
        measured = _texture_ridge_radius(tiles, m, (d.cx_px, d.cy_px), ring.z, r_max)
        gap = abs(ring.radius_px - measured)
        agree.append(gap)
        print(f"    Z{ring.z + 1:03d}: fit says {ring.radius_px:6.0f} px, "
              f"the texture ridge peaks at {measured:6.0f} px  ({gap:5.0f} px apart)")
    ok(float(np.median(agree)) < 120.0,
       f"the fitted rim tracks the texture ridge to a median {np.median(agree):.0f} px, "
       f"which is inside the width of the interface band itself")

    # --- physical plausibility, stated in units a bench scientist recognises ---
    print("\n  is it a droplet someone could have pipetted?")
    if acq.calibrated:
        a = d.contact_radius_px * acq.px_um
        h = d.height_slices * acq.z_um
        volume = np.pi * h / 6.0 * (3.0 * a * a + h * h) / 1e9
        ok(2.0 <= 2 * a / 1000 <= 12.0,
           f"footprint {2 * a / 1000:.2f} mm across")
        ok(5.0 <= volume <= 150.0,
           f"volume {volume:.0f} ul, in the range a Matrigel dome is pipetted at")
        ok(abs(a * a - h * (2 * d.radius_px * acq.px_um - h)) / (a * a) < 0.02,
           "contact radius, height and sphere radius satisfy the spherical-cap "
           "identity a^2 = h(2R - h)")

    # --- and the metadata claim that is easy to get wrong ---
    print("\n  what the operator's recorded points actually are:")
    eps = read_edge_points(dataset)
    if eps:
        stage = np.array([[t.stage_nm[0], t.stage_nm[1], 1.0] for t in m])
        centres = np.array([[t.x0 + t.width / 2, t.y0 + t.height / 2] for t in m])
        mapping, *_ = np.linalg.lstsq(stage, centres, rcond=None)
        pts = np.array([[x, y, 1.0] for x, y, _ in eps]) @ mapping
        xs = sorted({round(t.x0 + t.width / 2, 2) for t in m})
        ys = sorted({round(t.y0 + t.height / 2, 2) for t in m})
        ok(abs(pts[:, 0].min() - xs[0]) < 3.0 and abs(pts[:, 1].min() - ys[0]) < 3.0,
           f"they are the requested scan bounds: the first tile centre sits "
           f"{abs(pts[:, 0].min() - xs[0]):.1f} px from the first recorded point")
        rim_gap = np.median([abs(np.hypot(p[0] - d.cx_px, p[1] - d.cy_px)
                                 - d.contact_radius_px) for p in pts])
        ok(rim_gap > 100.0,
           f"and they are NOT rim points: they sit {rim_gap:.0f} px "
           f"({rim_gap * (acq.px_um or 0) / 1000:.2f} mm) off the fitted contact "
           f"circle, so they must not be used to validate it")

    failed = sum(1 for good, _ in _RESULTS if not good)
    print(f"\n{len(_RESULTS) - failed} passed, {failed} failed")
    if not failed:
        print(f"\n{sub.describe()}")
        print(f"axis            ({d.cx_px:.1f}, {d.cy_px:.1f}) px")
        print(f"sphere radius   {d.radius_px:.0f} px")
        print(f"contact radius  {d.contact_radius_px:.0f} px")
        print(f"height          {d.height_slices:.0f} slices, apex at slice "
              f"{d.apex_slice + 1:.0f}")
    return 1 if failed else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Check the global dome fit")
    ap.add_argument("dataset", nargs="?", default="BK52_WT_9805_B")
    a = ap.parse_args(argv)
    d = Path(a.dataset)
    if not d.is_dir():
        print(f"no such dataset: {d}", file=sys.stderr)
        return 2
    return run(d)


if __name__ == "__main__":
    raise SystemExit(main())
