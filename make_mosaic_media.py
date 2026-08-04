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

from jx3d.blend import MosaicSlices, estimate_flat_field, feather_weights
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

    Each field is laid down feathered against the ones already there, the same
    way the viewer composites, so the figure shows the mosaic that the code
    actually produces. Pasting raw tiles instead leaves a visible grid of seams
    and brightness steps, which advertises a problem that was solved two
    versions ago.
    """
    weight = feather_weights(mosaic.tile_height, mosaic.tile_width,
                             *mosaic.overlap_px())
    acc = np.zeros((mosaic.height, mosaic.width), np.float32)
    wsum = np.zeros((mosaic.height, mosaic.width), np.float32)
    order = sorted(mosaic.tiles, key=lambda t: t.index)
    frames = []
    for i, tile in enumerate(order, start=1):
        img = slices.tile_slice(tile, z)
        x0, y0 = int(round(tile.x0)), int(round(tile.y0))
        h, w = img.shape
        acc[y0:y0 + h, x0:x0 + w] += img * weight
        wsum[y0:y0 + h, x0:x0 + w] += weight
        blended = np.divide(acc, wsum, out=np.zeros_like(acc), where=wsum > 0)
        vis = cv2.cvtColor(np.clip(blended, 0, 255).astype(np.uint8),
                           cv2.COLOR_GRAY2BGR)
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

    # The still is the finished mosaic with nothing drawn on it, because the
    # point of that figure is that you cannot see where the joins are.
    clean = np.divide(acc, wsum, out=np.zeros_like(acc), where=wsum > 0)
    cv2.imwrite(str(docs / "mosaic.png"),
                fit_width(np.clip(clean, 0, 255).astype(np.uint8), 1400),
                [cv2.IMWRITE_PNG_COMPRESSION, 9])


def stitching_figure(mosaic, slices, meta, docs, z, width=1500):
    """How fifteen photographs become one, in four panels.

    Written because the assembled mosaic is the one figure where the work is
    invisible by design: if the stitching is right, the joins cannot be seen,
    and a reader has no way to tell that anything happened at all.
    """
    reg = meta["registration"]
    ox, oy = mosaic.overlap_px()
    a = mosaic.at(0, 0); b = mosaic.at(0, 1)
    ia, ib = slices.tile_slice(a, z), slices.tile_slice(b, z)

    cell = (width // 2, int(width // 2 * 0.78))

    def panel(img, title, sub, colour=False):
        """One quarter of the figure, letterboxed into a fixed box.

        Fixed, because the four things being shown have wildly different shapes
        -- a whole mosaic, a single field, a pair of tall thin strips -- and
        stacking them at their natural sizes leaves the figure ragged and full
        of dead space.
        """
        v = img if colour else cv2.cvtColor(
            np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        k = min(cell[0] / v.shape[1], cell[1] / v.shape[0])
        v = cv2.resize(v, (max(1, int(v.shape[1] * k)), max(1, int(v.shape[0] * k))),
                       interpolation=cv2.INTER_AREA)
        box = np.full((cell[1], cell[0], 3), 22, np.uint8)
        oy, ox_ = (cell[1] - v.shape[0]) // 2, (cell[0] - v.shape[1]) // 2
        box[oy:oy + v.shape[0], ox_:ox_ + v.shape[1]] = v
        return label(box, title, sub)

    # 1. one field, and how little of the droplet it holds
    one = np.zeros((mosaic.height, mosaic.width), np.float32)
    x0, y0 = int(round(a.x0)), int(round(a.y0))
    one[y0:y0 + a.height, x0:x0 + a.width] = ia
    o = cv2.cvtColor(np.clip(one, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.rectangle(o, (0, 0), (mosaic.width - 1, mosaic.height - 1), (70, 80, 100), 6)
    cv2.rectangle(o, (x0, y0), (x0 + a.width, y0 + a.height), GOLD, 6)
    p1 = panel(o, "1. one field of view",
               f"{a.width}x{a.height} px of a droplet {mosaic.width}x{mosaic.height} px across",
               colour=True)

    # 2. the grid the microscope actually recorded
    grid = np.zeros((mosaic.height, mosaic.width), np.float32)
    for t in mosaic:
        tx, ty = int(round(t.x0)), int(round(t.y0))
        grid[ty:ty + t.height, tx:tx + t.width] = slices.tile_slice(t, z)
    g = cv2.cvtColor(np.clip(grid, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for t in mosaic:
        tx, ty = int(round(t.x0)), int(round(t.y0))
        cv2.rectangle(g, (tx, ty), (tx + t.width, ty + t.height), GOLD, 3)
        cv2.putText(g, str(t.index), (tx + 18, ty + 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, GOLD, 4, cv2.LINE_AA)
    for t in mosaic:                                    # shade the shared strips
        r = mosaic.at(t.row, t.col + 1)
        if r is not None:
            xs = int(round(r.x0)); xe = int(round(t.x0 + t.width))
            cv2.rectangle(g, (xs, int(round(t.y0))), (xe, int(round(t.y0 + t.height))),
                          (80, 90, 255), 2)
    p2 = panel(g, f"2. fifteen fields, overlapping {100 * ox / a.width:.0f}%",
               "red = the strips two fields both photographed", colour=True)

    # 3. what the overlap is for: the same specimen, twice
    w = int(round(ox))
    strip_a, strip_b = ia[:, -w:], ib[:, :w]
    pair = np.hstack([strip_a, np.full((strip_a.shape[0], 12), 255.0), strip_b])
    p3 = panel(pair, f"3. the shared strip, from each field",
               f"correlating these placed the fields to {reg['residual_px']:.2f} px")

    # 4. the join, feathered away
    weight = feather_weights(mosaic.tile_height, mosaic.tile_width, ox, oy)
    acc = np.zeros((mosaic.height, mosaic.width), np.float32)
    wsum = np.zeros_like(acc)
    for t in mosaic:
        img = slices.tile_slice(t, z); tx, ty = int(round(t.x0)), int(round(t.y0))
        h, wd = img.shape
        acc[ty:ty + h, tx:tx + wd] += img * weight
        wsum[ty:ty + h, tx:tx + wd] += weight
    fused = np.divide(acc, wsum, out=np.zeros_like(acc), where=wsum > 0)
    p4 = panel(fused, "4. one photograph of the whole droplet",
               "illumination flattened first, then the seams feathered")

    cv2.imwrite(str(docs / "stitching.png"),
                np.vstack([np.hstack([p1, p2]), np.hstack([p3, p4])]),
                [cv2.IMWRITE_PNG_COMPRESSION, 9])
    print(f"  {docs / 'stitching.png'}")


def dome_gif(mosaic, slices, meta, docs, width=560, n=20):
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
    # Down and back up, so the widening reads as a sweep rather than a jump cut.
    # The palette is deliberately small: this frame is a grey photograph with
    # two coloured circles on it, and spending colours on the photograph's
    # gradient costs megabytes without showing anything.
    frames = frames + frames[-2:0:-1]
    save_gif(docs / "dome_rings.gif", frames, 150, palettesize=64)


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
    stitching_figure(mosaic, slices, meta, docs, z)
    dome_gif(mosaic, slices, meta, docs)
    fov_gif(mosaic, slices, rows, docs, z)
    border_figure(mosaic, slices, meta, rows, docs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
