"""Build the whole-dome viewer: one file, fifteen switchable fields, one 3D scene.

Inlining every slice of every tile at full resolution would be 222 MB, which no
browser opens pleasantly. Inlining almost nothing and relying on the all-in-focus
projection was the other extreme, and it was wrong in a way that mattered: a
Z-stack whose depth you cannot step through is not a Z-stack, and the one thing
this modality actually measures is depth. Stepping down through the gel, watching
each organoid sharpen at its own plane and blur again, *is* the evidence that the
depths in the feature matrix are real.

So the budget goes on the mosaic stack: every analysed slice, assembled and
flat-fielded, at a width that keeps a 93-slice stack near thirteen megabytes. The
per-tile projections stay as well, because when you switch to a single field and
ask what is in it, the image you want is the one where everything is sharp at
once.

The same slice image is used for both panes -- the photograph on the left and
the plane floating at its own depth inside the 3D volume on the right -- so the
two cannot drift apart.
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


def _now_utc() -> str:
    """When this viewer was generated.

    A generated page is a frozen copy of the template, and there is no way to
    tell a stale one apart by looking at it -- which is exactly how a viewer
    several fixes behind ended up being the one on disk while the fixes sat in
    git. Stamping the build makes that visible at a glance instead of after an
    hour of wondering why a change did not appear.
    """
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


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


_KEEP_REDUNDANT = ("radial_profile_um", "outlines_px", "z_extent_slices")
"""Per-organoid fields dropped from the payload. `radial_profile_um` is the
pixel profile times a constant, and at 48 angles for 844 organoids that constant
costs a third of a megabyte to repeat."""


def _trim(row: dict) -> dict:
    """One organoid, small enough to ship 844 of them in a single file.

    Nothing is dropped that the feature table shows -- the whole point of the
    viewer is that every measured number is one click away. What goes is
    precision nobody reads: a texture descriptor quoted to fifteen significant
    figures is fourteen more than the measurement supports, and repeated across
    a hundred and thirty-nine columns and eight hundred rows it costs megabytes.
    """
    out = {}
    for k, v in row.items():
        if k in _KEEP_REDUNDANT:
            continue
        if isinstance(v, float):
            out[k] = round(v, 4)
        else:
            out[k] = v
    profile = out.get("radial_profile_px")
    if profile:
        # The outline is what makes superposition on the photograph honest --
        # a circle would draw a shape that was never measured -- but a tenth of
        # a pixel is already below what the segmentation resolves.
        out["radial_profile_px"] = [round(float(v), 1) for v in profile]
    return out


def build(result, path: str | Path, quality: int = 70,
          slice_width: int = 760, slice_quality: int = 60,
          tile_width: int = 720, progress=None) -> Path:
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

    steps = len(mosaic) + 1 + tiles.depth
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
            "edf": _jpeg_uri(projection, quality, max_width=tile_width),
        })
        done += 1
        if progress:
            progress(done, steps)

    mosaic_edf = _jpeg_uri(canvas, quality, max_width=1800)
    done += 1
    if progress:
        progress(done, steps)

    # The whole stack, not just the part that was analysed. Measurement stops
    # short of the glass because the dish surface is sharper than anything
    # biological and would capture every focus peak, but that is a reason to
    # exclude those slices from the *analysis*, not from the picture. They were
    # photographed, they show the specimen settling onto the glass, and a viewer
    # that silently ends at slice 91 of 119 is hiding a quarter of the data.
    stack = []
    for z in range(tiles.depth):
        stack.append(_jpeg_uri(fused.slice(z), slice_quality,
                               max_width=slice_width))
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
            "built_utc": _now_utc(),
            "z_analysed": z_stop,
            "z_total": tiles.depth,
            "n_theta": len((result.rows[0].get("radial_profile_px") or [])) or None,
        },
        "tiles": tile_payload,
        "organoids": [_trim(r) for r in result.rows],
        "mosaic_edf": mosaic_edf,
        "stack": stack,
    }

    html = _TEMPLATE.read_text(encoding="utf-8")
    html = html.replace("/*__THREE__*/", _THREE.read_text(encoding="utf-8"))
    html = html.replace("/*__DATA__*/", json.dumps(payload, separators=(",", ":")))

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out
