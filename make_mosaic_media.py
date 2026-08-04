#!/usr/bin/env python
"""Build the animated figures for the whole-dome section of README.md.

    ./.venv/bin/python make_mosaic_media.py output/BK52_WT_9805_B_mosaic

Every frame here is rendered from the real run -- the same geometry the viewer
draws and the same numbers in the feature matrix. Nothing is illustrative.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from jx3d.blend import MosaicSlices, estimate_flat_field
from jx3d.keyence import read_group_metadata
from jx3d.mosaic import build
from jx3d.register import register

GOLD = (60, 200, 255)
GREEN = (110, 235, 130)
AMBER = (84, 180, 255)


def save_gif(path: Path, frames, delay_ms: int, palettesize: int = 128) -> None:
    """Write a GIF with one shared, undithered palette.

    The delay is in milliseconds; passing seconds rounds it to zero and every
    browser then falls back to its own minimum, which is why the value is read
    back out of the finished file and checked. Dithering is off because
    Floyd-Steinberg scatters different noise into every frame, which defeats
    compression on a smooth background and roughly doubles the file for no
    visible gain.
    """
    from PIL import Image

    pil = [Image.fromarray(f) for f in frames]
    sample = pil[:: max(1, len(pil) // 12)] or pil[:1]
    w, h = pil[0].size
    strip = Image.new("RGB", (w, h * len(sample)))
    for i, im in enumerate(sample):
        strip.paste(im, (0, i * h))
    pal = strip.quantize(colors=palettesize, method=Image.Quantize.MEDIANCUT)
    quant = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in pil]
    quant[0].save(path, save_all=True, append_images=quant[1:],
                  duration=int(delay_ms), loop=0, disposal=1, optimize=True)
    with Image.open(path) as im:
        im.seek(min(1, im.n_frames - 1))
        written = im.info.get("duration", 0)
        n = im.n_frames
    if not written:
        raise RuntimeError(f"{path}: frame delay was not stored")
    print(f"  {path}  ({path.stat().st_size / 1e6:.1f} MB, {n} frames, "
          f"{written} ms/frame)")


def label(img, text, sub=""):
    h = img.shape[0]
    cv2.rectangle(img, (0, h - (40 if sub else 26)), (img.shape[1], h), (16, 11, 7), -1)
    cv2.putText(img, text, (12, h - (25 if sub else 8)), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (240, 245, 255), 1, cv2.LINE_AA)
    if sub:
        cv2.putText(img, sub, (12, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (150, 170, 200), 1, cv2.LINE_AA)
    return img


def fit_width(img, width):
    return cv2.resize(img, (width, int(img.shape[0] * width / img.shape[1])),
                      interpolation=cv2.INTER_AREA)


def assembly_gif(mosaic, slices, docs, z, width=760):
    """Fifteen fields arriving one at a time, in the order they were captured.

    The order matters to the picture: the stage snakes along each row, so the
    mosaic fills left to right, then right to left, which is exactly what the
    serpentine numbering in the group file records.
    """
    canvas = np.zeros((mosaic.height, mosaic.width), np.uint8)
    order = sorted(mosaic.tiles, key=lambda t: t.index)
    frames = []
    for i, tile in enumerate(order, start=1):
        img = slices.tile_slice(tile, z).astype(np.uint8)
        x0, y0 = int(round(tile.x0)), int(round(tile.y0))
        canvas[y0:y0 + tile.height, x0:x0 + tile.width] = img
        vis = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        for t in order[:i]:
            tx, ty = int(round(t.x0)), int(round(t.y0))
            cv2.rectangle(vis, (tx, ty), (tx + t.width, ty + t.height), GOLD, 3)
            cv2.putText(vis, str(t.index), (tx + 20, ty + 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, GOLD, 4, cv2.LINE_AA)
        vis = fit_width(vis, width)
        vis = label(vis, f"field {tile.index} of 15  ·  row {tile.row + 1}, "
                         f"column {tile.col + 1}",
                    "30% overlap; placement from the stage log, then refined "
                    "against the pixels")
        frames.append(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    for _ in range(12):
        frames.append(frames[-1])
    save_gif(docs / "mosaic_assembly.gif", frames, 240)
    cv2.imwrite(str(docs / "mosaic.png"),
                cv2.cvtColor(frames[-1], cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_PNG_COMPRESSION, 9])


def dome_gif(mosaic, slices, meta, docs, width=760, n=26):
    """The droplet's cross-section widening as the focus descends.

    This is the whole basis of the dome fit, and it is a figure only the mosaic
    can produce: in a single field the interface is a short arc crossing one
    corner, and nothing about it looks like a circle that grows.
    """
    dome = meta["dome"]["dome"]
    rings = {r["z"]: r for r in meta["dome"]["rings"] if r["used"]}
    if not rings:
        return
    zs = sorted(rings)
    picks = np.linspace(zs[0], zs[-1], n).round().astype(int)
    aniso = meta["units"]["anisotropy"]
    px_um = meta["units"]["px_um"]

    frames = []
    fused = MosaicSlices(mosaic, slices.flat, blend=True)
    for z in picks:
        near = min(zs, key=lambda k: abs(k - z))
        r = rings[near]
        vis = cv2.cvtColor(fused.slice(int(z)), cv2.COLOR_GRAY2BGR)
        cv2.circle(vis, (int(dome["cx_px"]), int(dome["cy_px"])),
                   int(dome["contact_radius_px"]), (90, 150, 90), 2)
        cv2.circle(vis, (int(r["cx_px"]), int(r["cy_px"])),
                   int(r["radius_px"]), GREEN, 5)
        vis = fit_width(vis, width)
        label(vis, f"slice Z{int(z) + 1:03d}   ·   droplet cross-section "
                   f"{2 * r['radius_px'] * px_um / 1000:.2f} mm across",
              "faint ring = where the gel meets the glass, the widest it ever gets")
        frames.append(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    frames = frames + frames[::-1]
    save_gif(docs / "dome_rings.gif", frames, 130)


def fov_gif(mosaic, slices, rows, docs, z, width=760):
    """Switching fields on and off, which is how the viewer is actually used."""
    canvas = {}
    for tile in mosaic:
        canvas[tile.index] = slices.tile_slice(tile, z).astype(np.uint8)

    def compose(active, caption, sub):
        img = np.zeros((mosaic.height, mosaic.width), np.uint8)
        for tile in mosaic:
            if tile.index not in active:
                continue
            x0, y0 = int(round(tile.x0)), int(round(tile.y0))
            img[y0:y0 + tile.height, x0:x0 + tile.width] = canvas[tile.index]
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for tile in mosaic:
            x0, y0 = int(round(tile.x0)), int(round(tile.y0))
            on = tile.index in active
            cv2.rectangle(vis, (x0, y0), (x0 + tile.width, y0 + tile.height),
                          GOLD if on else (60, 60, 70), 3 if on else 1)
        for row in rows:
            if row.get("tile") and tileindex(mosaic, row["tile"]) in active:
                cv2.circle(vis, (int(row["x_mosaic_px"]), int(row["y_mosaic_px"])),
                           max(3, int(row.get("radius_px", 6))), AMBER, 2)
        vis = fit_width(vis, width)
        return cv2.cvtColor(label(vis, caption, sub), cv2.COLOR_BGR2RGB)

    everything = {t.index for t in mosaic}
    steps = [(everything, "all 15 fields", "the whole droplet, counted once"),
             ({4}, "field 4 alone", "shift-click a cell in the viewer's map"),
             ({1, 2}, "fields 1 and 2", "neighbours, with their overlap merged"),
             ({7, 8, 9}, "row 3", "the middle band of the dome"),
             ({5, 8, 11}, "the centre column", "the deepest gel, where the glass is hidden"),
             (everything, "all 15 fields", "the whole droplet, counted once")]
    frames = []
    for active, cap, sub in steps:
        frame = compose(active, cap, sub)
        frames.extend([frame] * 8)
    save_gif(docs / "fov_toggle.gif", frames, 180)


def tileindex(mosaic, name):
    t = mosaic.by_name(name)
    return t.index if t else -1


def border_figure(mosaic, slices, meta, rows, docs, width=900):
    """A few organoids with a line drawn to their nearest point on the gel edge."""
    dome = meta["dome"]["dome"]
    px_um = meta["units"]["px_um"]
    candidates = [r for r in rows if r.get("nearest_border_px") is not None]
    if not candidates:
        return
    candidates.sort(key=lambda r: r["nearest_border_px"])
    picks = candidates[:3] + candidates[len(candidates) // 2:len(candidates) // 2 + 2]

    z = int(round(np.median([r["z_slice"] for r in picks])))
    fused = MosaicSlices(mosaic, slices.flat, blend=True)
    vis = cv2.cvtColor(fused.slice(z), cv2.COLOR_GRAY2BGR)
    cv2.circle(vis, (int(dome["cx_px"]), int(dome["cy_px"])),
               int(dome["contact_radius_px"]), (90, 160, 90), 3)
    for r in picks:
        x, y = r["x_mosaic_px"], r["y_mosaic_px"]
        dx, dy = x - dome["cx_px"], y - dome["cy_px"]
        rad = max(np.hypot(dx, dy), 1e-6)
        rz = np.sqrt(max(dome["radius_px"] ** 2 -
                         ((r["z_slice"] - dome["cz_slice"]) *
                          meta["units"]["anisotropy"]) ** 2, 0.0))
        ex, ey = dome["cx_px"] + dx / rad * rz, dome["cy_px"] + dy / rad * rz
        cv2.line(vis, (int(x), int(y)), (int(ex), int(ey)), AMBER, 3, cv2.LINE_AA)
        cv2.circle(vis, (int(x), int(y)), max(6, int(r.get("radius_px", 8))),
                   (255, 255, 255), 3)
        cv2.putText(vis, f"{r['nearest_border_px'] * px_um / 1000:.2f} mm",
                    (int(x) + 16, int(y) - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, AMBER, 3, cv2.LINE_AA)
    vis = fit_width(vis, width)
    label(vis, "distance from each organoid's surface to the nearest gel boundary",
          "the same number the feature matrix reports as nearest_border_px")
    cv2.imwrite(str(docs / "border_distance.png"), vis,
                [cv2.IMWRITE_PNG_COMPRESSION, 9])
    print(f"  {docs / 'border_distance.png'}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the whole-dome README figures")
    ap.add_argument("outdir", nargs="?", default="output/BK52_WT_9805_B_mosaic")
    ap.add_argument("--dataset", default="BK52_WT_9805_B")
    ap.add_argument("--docs", default="docs")
    a = ap.parse_args(argv)

    outdir, docs = Path(a.outdir), Path(a.docs)
    docs.mkdir(parents=True, exist_ok=True)
    meta = json.loads((outdir / "mosaic.json").read_text())
    rows = list(__import__("csv").DictReader(
        (outdir / "features.csv").open(encoding="utf-8")))
    for r in rows:
        for k, v in list(r.items()):
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass

    acq, _ = read_group_metadata(a.dataset)
    mosaic = build(a.dataset, acq)
    flat = estimate_flat_field(mosaic)
    slices = MosaicSlices(mosaic, flat, blend=False)
    slices.flat = flat
    mosaic, _ = register(mosaic, slices)
    slices = MosaicSlices(mosaic, flat, blend=False)
    slices.flat = flat

    z = int(round(meta["dome"]["substrate"]["slice_index"] * 0.62))
    print("building figures:")
    assembly_gif(mosaic, slices, docs, z)
    dome_gif(mosaic, slices, meta, docs)
    fov_gif(mosaic, slices, rows, docs, z)
    border_figure(mosaic, slices, meta, rows, docs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
