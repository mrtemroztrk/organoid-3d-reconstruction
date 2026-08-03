"""Build the self-contained HTML viewer.

Everything -- three.js, every raw slice as a JPEG data URI, and the measurement
table -- goes into one file you can double-click. No server, no CDN, no network.
At q=72 the 119-slice stack costs about 16 MB, which is the point: the previous
viewers embedded one JSON object per voxel and reached 800 MB.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import cv2
import numpy as np

from .config import Params
from .reconstruct import Organoid

_TEMPLATE = Path(__file__).with_name("viewer_template.html")
_THREE = Path(__file__).parent / "assets" / "three.min.js"


def _encode_slices(volume: np.ndarray, quality: int = 72,
                   max_width: int | None = None, progress=None) -> list[str]:
    uris: list[str] = []
    for z in range(volume.shape[0]):
        img = volume[z]
        if max_width and img.shape[1] > max_width:
            scale = max_width / img.shape[1]
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError(f"JPEG encoding failed on slice {z}")
        uris.append("data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii"))
        if progress:
            progress(z + 1, volume.shape[0])
    return uris


def build_viewer(stack, organoids: list[Organoid], params: Params,
                 focus_profile: np.ndarray, substrate_slice: int,
                 output: str | Path, jpeg_quality: int = 72,
                 dome=None, progress=None) -> Path:
    acq = stack.acq
    meta = {
        "dataset": stack.name,
        "objective": acq.objective,
        "width": stack.width,
        "height": stack.height,
        "depth": stack.depth,
        "calibrated": acq.calibrated,
        "px_um": acq.px_um,
        "z_um": acq.z_um,
        "px_um_source": acq.px_um_source,
        "z_um_source": acq.z_um_source,
        "anisotropy": round(acq.anisotropy, 6),
        "n_theta": params.n_theta,
        "n_phi": params.n_phi,
        "substrate_slice": int(substrate_slice),
        "focus_profile": [round(float(v), 2) for v in focus_profile],
        "depth_of_field_slices": round(acq.depth_of_field_slices, 2),
        "dome": dome.to_dict() if dome is not None else None,
    }
    payload = {"meta": meta, "organoids": [o.to_dict() for o in organoids]}

    slices = _encode_slices(stack.data, quality=jpeg_quality, progress=progress)

    html = _TEMPLATE.read_text(encoding="utf-8")
    html = html.replace("/*__THREE__*/", _THREE.read_text(encoding="utf-8"))
    html = html.replace("/*__DATA__*/", json.dumps(payload, separators=(",", ":")))
    html = html.replace("/*__SLICES__*/", json.dumps(slices, separators=(",", ":")))

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out
