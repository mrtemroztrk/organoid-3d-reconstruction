"""Build the whole-dome viewer: one file, fifteen switchable fields, one 3D scene.

The single-field viewer inlines every raw slice as a JPEG. That costs about
sixteen megabytes for one tile, and fifteen tiles would be a quarter of a
gigabyte in a single HTML file -- which no browser will open pleasantly and
nobody will send anyone. The budget has to be spent somewhere else.

It is spent on the all-in-focus projections. A brightfield Z-stack of a dome is
mostly out-of-focus haze: on any given slice a handful of organoids are crisp
and the rest are grey discs. The projection is the one image where every
organoid in that field is sharp at once, which is exactly the image someone
wants when they are looking at a field and asking what is in it. Sixteen of them
-- one per tile, plus the assembled mosaic -- come to a few megabytes, and the
per-slice scrub is kept as a coarse mosaic stack rather than dropped entirely.

What is not compromised is the pairing. Every organoid drawn in 3D can be
clicked and found in the photograph it was measured in, at the depth it was
measured at, which is the only way any of these numbers can be checked.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

_TEMPLATE = Path(__file__).with_name("mosaic_viewer.html")
_THREE = Path(__file__).parent / "assets" / "three.min.js"


def template_version() -> str:
    import hashlib

    return hashlib.sha256(_TEMPLATE.read_bytes()).hexdigest()[:12]


def _jpeg_uri(image: np.ndarray, quality: int = 78,
              max_width: int | None = None) -> str:
    import cv2

    img = image
    if max_width and img.shape[1] > max_width:
        scale = max_width / img.shape[1]
        img = cv2.resize(img, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _tile_projection(tile, slices, z_from: int, z_to: int, step: int = 3
                     ) -> np.ndarray:
    """All-in-focus projection of one tile: the sharpest pixel over depth.

    Sampled every few slices rather than every one. The depth of field spans
    several slices, so consecutive frames are nearly the same image and the
    projection is indistinguishable at a third of the cost.
    """
    import cv2

    from .focus import tenengrad

    best_value = None
    best_pixel = None
    for z in range(z_from, z_to, step):
        frame = slices.tile_slice(tile, z)
        sharp = cv2.GaussianBlur(tenengrad(frame), (0, 0), 6.0)
        if best_value is None:
            best_value, best_pixel = sharp, frame.copy()
            continue
        better = sharp > best_value
        best_value[better] = sharp[better]
        best_pixel[better] = frame[better]
    return np.clip(best_pixel, 0, 255).astype(np.uint8)


def build(result, path: str | Path, quality: int = 78,
          scrub_slices: int = 26, scrub_width: int = 1100,
          progress=None) -> Path:
    """Write the viewer for a finished mosaic run."""
    from . import __version__
    from .blend import MosaicSlices, estimate_flat_field

    mosaic = result.mosaic
    acq = result.acq
    dome = result.dome_fit.dome
    substrate = result.dome_fit.substrate.slice_index

    flat = estimate_flat_field(mosaic)
    tiles = MosaicSlices(mosaic, flat, blend=False)
    fused = MosaicSlices(mosaic, flat, blend=True)
    z_stop = int(max(4, substrate - 3))

    steps = len(mosaic) + 1 + scrub_slices
    done = 0

    tile_payload = []
    canvas = np.zeros((mosaic.height, mosaic.width), dtype=np.uint8)
    for tile in mosaic:
        projection = _tile_projection(tile, tiles, 0, z_stop)
        x0, y0 = int(round(tile.x0)), int(round(tile.y0))
        canvas[y0:y0 + tile.height, x0:x0 + tile.width] = np.maximum(
            canvas[y0:y0 + tile.height, x0:x0 + tile.width], projection)
        tile_payload.append({
            "name": tile.name, "index": tile.index,
            "row": tile.row, "col": tile.col,
            "x0": round(tile.x0, 2), "y0": round(tile.y0, 2),
            "w": tile.width, "h": tile.height,
            "edf": _jpeg_uri(projection, quality),
        })
        done += 1
        if progress:
            progress(done, steps)

    mosaic_edf = _jpeg_uri(canvas, quality, max_width=2200)
    done += 1
    if progress:
        progress(done, steps)

    scrub = []
    for z in np.linspace(0, z_stop - 1, scrub_slices).round().astype(int):
        scrub.append({"z": int(z),
                      "img": _jpeg_uri(fused.slice(int(z)), 66,
                                       max_width=scrub_width)})
        done += 1
        if progress:
            progress(done, steps)

    payload = {
        "meta": {
            "version": __version__,
            "viewer_version": template_version(),
            "dataset": Path(mosaic.tiles[0].folder).parent.name,
            "width": mosaic.width, "height": mosaic.height,
            "depth": tiles.depth,
            "n_rows": mosaic.n_rows, "n_cols": mosaic.n_cols,
            "calibrated": acq.calibrated,
            "px_um": acq.px_um, "z_um": acq.z_um,
            "px_um_source": acq.px_um_source, "z_um_source": acq.z_um_source,
            "anisotropy": round(acq.anisotropy, 6),
            "substrate_slice": round(float(substrate), 2),
            "offset_source": mosaic.offset_source,
            "registration_residual_px": round(result.registration.residual_px, 3),
            "n_sightings": result.report.n_sightings,
            "n_merged": result.report.n_merged,
            "dome": dome.to_dict() if dome is not None else None,
            "scrub_width": scrub_width,
        },
        "tiles": tile_payload,
        "organoids": result.rows,
        "mosaic_edf": mosaic_edf,
        "scrub": scrub,
    }

    html = _TEMPLATE.read_text(encoding="utf-8")
    html = html.replace("/*__THREE__*/", _THREE.read_text(encoding="utf-8"))
    html = html.replace("/*__DATA__*/", json.dumps(payload, separators=(",", ":")))

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out
